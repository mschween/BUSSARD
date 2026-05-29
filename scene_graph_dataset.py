import gc
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch_geometric.data import Dataset

from dataset_summary import DatasetSummary
from sg_to_data import create_graph_data_glove, create_graph_data_llm
from synonyms import synonyms_dict
from triplets import Triplet, create_triplets, triplets_add_concepts


class SceneGraphDataset(Dataset):
    """
    A PyTorch Geometric dataset for loading scene graph JSON files without caching.
    Data is generated on-the-fly for each access.
    """

    def __init__(
        self,
        config_path: str,
        scene_name: str,
        transform=None,
        pre_transform=None,
        mutation_rate=None,
        word_embed=None,
        embed_type="glove",
        dataset_source="sard",  # "sard" or "visual_genome" or "mit-67"
    ):
        """
        embed_type options: "glove" and llm model names (see config for supported models).
        """
        self.config_path = config_path
        self.scene_name = scene_name
        self.mutation_rate = mutation_rate
        self.word_embed_preloaded = word_embed
        self.embedding_type = embed_type
        print(f"Used embedding type: {self.embedding_type}")
        self.dataset_source = dataset_source

        # Load config
        with open(config_path, "r") as config_file:
            self.all_configs = json.load(config_file)

        # Extract relevant config
        self.dataset_config = self.all_configs.get("dataset", {}).copy()

        if self.dataset_source == "visual_genome":
            # Use VG config
            vg_config = self.all_configs.get("visual_genome", {})
            self.dataset_config.update(vg_config)
        elif self.dataset_source == "mit-67":
            # Use Indoor config
            indoor_config = self.all_configs.get("mit-67", {})
            self.dataset_config.update(indoor_config)

            # Build path to specific scene's scene_graph folder
            base_path = indoor_config.get("sg_path")
            scene_sg_path = f"{base_path}/{scene_name}/scene_graph"
            self.dataset_config["sg_path"] = scene_sg_path
        else:
            # Use SARD scene config
            scene_config = self.all_configs.get("scenes", {}).get(scene_name, {})
            self.dataset_config.update(scene_config)

        # Get Embedding config
        self.embed_config = self.all_configs.get("embeddings", {}).get(embed_type, {})

        if self.embed_config is None:
            available = list(self.all_configs.get("embeddings", {}).keys())
            raise KeyError(
                f"Embedding model '{embed_type}' not found in config. "
                f"Available models: {available}"
            )

        # Max number of loaded triplets per image
        self.max_tpl_img = self.dataset_config.get("max_tpl_img")

        # Load ground truths (VG won't have this)
        gt_path = self.dataset_config.get("gt_path")
        if gt_path:
            with open(gt_path, "r") as file:
                self.ground_truths = json.load(file)
        else:
            self.ground_truths = {}  # Empty dict for VG

        # Set root directory
        self.root = Path(self.dataset_config.get("sg_path"))

        super().__init__(self.root, transform, pre_transform)

        # Process data
        self._process_data()

    def _process_data(self):
        """Process all data (called once during initialization)."""
        # Get file list
        self.file_list, self.normal_idx, self.anomal_idx = self._get_file_list()

        # Load embeddings once
        self.word_embed = self._load_embeddings()

        # Preprocess scene graphs
        self.all_tpls, self.data_summary = self._preprocess_all_images()

        # Create data objects list
        self.data_list, self.feature_info, valid_stems = self._create_data_objects()

        # Rebuild file_list and indices with only valid graphs
        if len(valid_stems) != len(self.file_list):
            stem_to_file = {f.stem: f for f in self.file_list}
            self.file_list = [stem_to_file[stem] for stem in valid_stems]
            self.normal_idx, self.anomal_idx = self._build_indices_from_files(
                self.file_list
            )
            print(f"Updated to {len(self.file_list)} valid graphs")

    def len(self):
        """Return the number of graphs in the dataset."""
        return len(self.data_list)

    def get(self, idx):
        """Get a single graph by index."""
        data = self.data_list[idx]

        # Apply transform if exists
        if self.transform:
            data = self.transform(data)

        return data

    def _load_embeddings(self):
        """Load embedding models."""
        if self.word_embed_preloaded is not None:
            return self.word_embed_preloaded  # skip loading
        if self.embedding_type == "glove":
            # GloVe
            from gensim.models import KeyedVectors

            glove = KeyedVectors.load_word2vec_format(
                "data/dolma_300_2024_1.2M.100_combined.txt",
                binary=False,
                no_header=True,
            )

            return glove
        else:
            from llm_word_embedding import CachedLLMEmbedder

            llm_name = self.embed_config["model_name"]

            llm_embed = CachedLLMEmbedder(
                model_name=llm_name,
                cache_path=f"concept_cache/{self.embedding_type}/embeddings_cache.db",
            )

            return llm_embed

    def _create_data_objects(self):
        """Create Data objects with embeddings."""
        data_list = []
        valid_file_stems = []  # Track which files actually produced data
        feature_info = None
        skipped_count = 0

        for file in self.file_list:
            key = file.stem

            # Skip if no triplets
            if key not in self.all_tpls or len(self.all_tpls[key]) == 0:
                skipped_count += 1
                continue

            # Handle labels based on dataset source
            if (
                self.dataset_source == "visual_genome"
                or self.dataset_source == "mit-67"
            ):
                label = 0  # All VG data is "normal"
            elif self.dataset_source == "sard":
                if key.startswith("0#"):
                    label = 0
                elif key.startswith("1#"):
                    label = 1
                else:
                    raise ValueError(f"Unknown file format: {key}")
            else:
                raise ValueError(f"Unknown dataset_source: {self.dataset_source}")

            try:
                # Convert to Data object
                if self.embedding_type == "glove":
                    data, feat_info = create_graph_data_glove(
                        triplets=self.all_tpls[key],
                        glove=self.word_embed,
                        data_sum=self.data_summary,
                        graph_label=label,
                        ground_truths=self.ground_truths,
                        mutation_rate=self.mutation_rate,
                    )
                else:
                    data, feat_info = create_graph_data_llm(
                        triplets=self.all_tpls[key],
                        llm_embedder=self.word_embed,
                        data_sum=self.data_summary,
                        graph_label=label,
                        ground_truths=self.ground_truths,
                        scene_type=self.scene_name,
                        mutation_rate=self.mutation_rate,
                    )

                # Check for NaN BEFORE adding to list
                if torch.isnan(data.x).any() or torch.isnan(data.edge_attr).any():
                    print(f"Skipping {key}: Contains NaN in embeddings")
                    skipped_count += 1
                    continue

                # Apply pre_transform if exists
                if self.pre_transform:
                    data = self.pre_transform(data)

                data_list.append(data)
                valid_file_stems.append(key)  # Track valid files

                if feature_info is None:
                    feature_info = feat_info

            except (RuntimeError, ValueError) as e:
                print(f"Skipping {key}: {e}")
                skipped_count += 1
                continue

        if skipped_count > 0:
            print(f"Skipped {skipped_count} graphs with empty/invalid data")

        return data_list, feature_info, valid_file_stems

    def _add_ground_truth(self, tpls, objs, preds, edges):
        """Add ground truth triplet if missing."""
        no_gt = True
        src = tpls[0].src if tpls else None
        if src and src in self.ground_truths:
            for gt in self.ground_truths[src]:
                if any(t.name == tuple(gt) for t in tpls):
                    no_gt = False

            if no_gt:
                print(f"Adding missing ground truth to image {src}")
                obj1, pred, obj2 = gt

                tpl = Triplet(
                    name=(obj1, pred, obj2),
                    confidence=1.0,
                    src=src,
                )

                # Update existing
                tpls.append(tpl)
                objs.update([obj1, obj2])
                preds.add(pred)
                edges.append((obj1, obj2))
        return tpls, objs, preds, edges

    def _get_triplet_signature(self, triplet):
        """Get unique signature based on object combination AND relationship"""
        obj1, rel, obj2 = triplet
        return (obj1["name"], rel["name"], obj2["name"])

    def _get_object_pair_signature(self, triplet):
        """Get object pair signature (ignoring relationship)"""
        obj1, rel, obj2 = triplet
        # Sort to make (cup, laptop) same as (laptop, cup)
        objects = sorted([obj1["name"], obj2["name"]])
        return tuple(objects)

    def _greedy_coreset_selection(self, triplets):
        """Prioritize unique object pairs, then unique relationships for same pairs"""
        if len(triplets) <= self.max_tpl_img:
            return triplets

        selected_indices = []
        used_full_signatures = set()
        used_object_pairs = set()

        # Start with highest confidence triplet (index 0)
        selected_indices.append(0)
        first_triplet = triplets[0]
        used_full_signatures.add(self._get_triplet_signature(first_triplet))
        used_object_pairs.add(self._get_object_pair_signature(first_triplet))

        # Greedily select remaining triplets
        for _ in range(self.max_tpl_img - 1):
            best_idx = -1

            # Priority 1: Find triplet with completely new object pair
            for i in range(len(triplets)):
                if i in selected_indices:
                    continue

                obj_pair = self._get_object_pair_signature(triplets[i])
                if obj_pair not in used_object_pairs:
                    best_idx = i
                    break

            # Priority 2: If no new object pairs, find new relationship for existing pair
            if best_idx == -1:
                for i in range(len(triplets)):
                    if i in selected_indices:
                        continue

                    full_sig = self._get_triplet_signature(triplets[i])
                    if full_sig not in used_full_signatures:
                        best_idx = i
                        break

            # Priority 3: Fall back to next highest confidence
            if best_idx == -1:
                for i in range(len(triplets)):
                    if i not in selected_indices:
                        best_idx = i
                        break

            if best_idx != -1:
                selected_indices.append(best_idx)
                selected_triplet = triplets[best_idx]
                used_full_signatures.add(self._get_triplet_signature(selected_triplet))
                used_object_pairs.add(self._get_object_pair_signature(selected_triplet))

        selected_triplets = [triplets[i] for i in selected_indices]
        return selected_triplets

    def _load_tpls(self, file, manual_add=True):
        """
        manual_add: If set to true, manually adds missing ground truths.
        """
        # Load JSON
        with open(file, "r") as f:
            json_data = json.load(f)[: self.max_tpl_img]

            # Skip empty files
            if len(json_data) == 0:
                return [], set(), set(), []

            for item in json_data:
                item[1]["src"] = file.stem

        if len(json_data) == 0:
            print(f"Loaded {len(json_data)} triplets from {file.stem}")  # DEBUG

        # Create triplets
        tpls, objs, preds, edges = create_triplets(
            json_triplets=json_data, filename=file.stem
        )

        # Check if all triplets were filtered out
        if len(tpls) == 0:
            return [], set(), set(), []

        if manual_add and self.ground_truths:
            # Add ground truth if anomalous
            # NOTE: Adding this to SARD as well!
            if file.stem.startswith("1#"):
                tpls, objs, preds, edges = self._add_ground_truth(
                    tpls, objs, preds, edges
                )

        return tpls, objs, preds, edges

    def _preprocess_all_images(self):
        """Process all scene graphs to triplets."""
        all_tpls = {}
        objects = set()
        predicates = set()
        edge_list = []

        def _load_file(file):
            return file.stem, *self._load_tpls(file)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_load_file, self.file_list))

        for stem, tpls, objs, preds, edges in results:
            all_tpls[stem] = tpls
            objects.update(objs)
            predicates.update(preds)
            edge_list.extend(edges)

        synonyms = list(synonyms_dict.values())
        objects.update(synonyms)

        # Initialize concept maps
        obj_to_concept_map = {}
        concept_to_obj_map = {}

        # Add concepts
        for src, tpls in all_tpls.items():
            concept_tpls, obj_to_concept_map, concept_to_obj_map = (
                triplets_add_concepts(
                    config=self.dataset_config, tpls=tpls, all_objects=objects
                )
            )

            # Update all_tpls
            all_tpls[src] = concept_tpls

        # Create mappings
        obj_to_idx = {o: i for i, o in enumerate(objects)}
        pred_to_idx = {p: i for i, p in enumerate(predicates)}
        concept_to_idx = {c: i for i, c in enumerate(concept_to_obj_map.keys())}

        edge_index = [[obj_to_idx[e[0]], obj_to_idx[e[1]]] for e in edge_list]

        data_summary = DatasetSummary(
            all_objects=objects,
            all_predicates=predicates,
            all_concepts=set(concept_to_obj_map.keys()),
            all_edge_index=edge_index,
            obj_to_idx=obj_to_idx,
            pred_to_idx=pred_to_idx,
            concept_to_idx=concept_to_idx,
            obj_to_concept_map=obj_to_concept_map,
            concept_to_obj_map=concept_to_obj_map,
        )

        return all_tpls, data_summary

    def _build_indices_from_files(self, files):
        """Build normal/anomal indices from file list."""
        if self.dataset_source == "visual_genome" or self.dataset_source == "mit-67":
            normal_idx = list(range(len(files)))
            anomal_idx = []
        else:  # SARD
            normal_idx = [i for i, f in enumerate(files) if f.stem.startswith("0#")]
            anomal_idx = [i for i, f in enumerate(files) if f.stem.startswith("1#")]

        return normal_idx, anomal_idx

    def _get_file_list(self):
        """Get sorted file list with indices."""
        all_files = Path(self.root).glob("*.json")

        # Filter out empty files
        all_files = list(Path(self.root).glob("*.json"))
        valid_files = [f for f in all_files if f.stat().st_size > 2]
        empty_files = len(all_files) - len(valid_files)  # <-- count difference

        # Sort files
        if self.dataset_source == "visual_genome" or self.dataset_source == "mit-67":
            # No label prefix, just sort by name
            files = sorted(valid_files, key=lambda x: x.stem)
        else:  # SARD
            files = sorted(valid_files, key=lambda x: int(x.stem.split("#")[1]))

        # Build indices
        normal_idx, anomal_idx = self._build_indices_from_files(files)

        print(f"Skipped {empty_files} empty files.")

        return files, normal_idx, anomal_idx

    def cleanup_embeddings(self):
        """Free memory used by embedding model after data objects are created."""
        if (
            hasattr(self, "word_embed_preloaded")
            and self.word_embed_preloaded is not None
        ):
            return  # Don't delete something we don't own

        if hasattr(self, "word_embed"):
            if hasattr(self.word_embed, "model"):
                del self.word_embed.model
            del self.word_embed
            self.word_embed = None

        gc.collect()
        torch.cuda.empty_cache()
        print("Embedding model memory freed")


if __name__ == "__main__":
    # Example usage
    config_path = "config.json"
    scene_name = "office"

    # Create dataset (always generates fresh)
    dataset = SceneGraphDataset(config_path, scene_name=scene_name)
    print(f"Dataset created: {len(dataset)} samples")
