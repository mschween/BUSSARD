import random
from typing import Dict

import numpy as np
import torch
from gensim.models import KeyedVectors
from torch_geometric.data import Data

from dataset_summary import DatasetSummary
from llm_word_embedding import CachedLLMEmbedder
from synonyms import synonyms_dict

_embed_cache = {}


def _embed_word_avg(glove, name):
    if name not in _embed_cache:
        words = name.split()
        feat = np.zeros(300)
        valid = 0
        for word in words:
            if word in glove:
                feat += glove[word]
                valid += 1
        if valid == 0:
            print(f"Warning: '{name}' has no valid GloVe embeddings, using zero vector")
        else:
            feat /= valid
        _embed_cache[name] = feat
    return _embed_cache[name]


def get_edge_label(triplet, all_ground_truths):
    """
    Use ground truth for anomalies. Normal data has only 0 label. If label not in gts returns 0.
    """
    # Handle list of (list of) triplets
    if isinstance(triplet, list):
        return [get_edge_label(t, all_ground_truths) for t in triplet]

    # Check if ground truths exist for this image
    if triplet.src not in all_ground_truths:
        return 0  # No ground truth = normal (for VG)

    # All images with name starting with 0 are completely normal
    if triplet.src[0] == "0":
        return 0
    # Iterate over all ground truths for the given image
    img_ground_truths = all_ground_truths[triplet.src]
    for gt in img_ground_truths:
        if triplet.name == tuple(gt):
            return 1
    return 0


def _prepare_graph_data(
    triplets,
    data_sum: DatasetSummary,
    ground_truths: Dict,
    mutation_rate: float = None,
):
    """Shared logic for processing triplets into graph structure"""
    # Build edge list and edge features
    edge_index = []
    edge_attr = []
    edge_labels = []
    triplet_data = []

    # Get node indices in this graph
    node_indices = set()
    # Store object names for each global index
    idx_to_obj_local = {}

    # Prepare encoding
    for triplet in triplets:
        obj1, predicate, obj2 = triplet.name

        if mutation_rate:
            # Mutate objects with muation_rate probability
            if obj1 in synonyms_dict:
                if random.random() < mutation_rate:
                    obj1 = synonyms_dict[obj1]
            if obj2 in synonyms_dict:
                if random.random() < mutation_rate:
                    obj2 = synonyms_dict[obj2]

        # Get global node indices
        idx1 = data_sum.obj_to_idx[obj1]
        idx2 = data_sum.obj_to_idx[obj2]

        # Add to node indices set and store object names
        node_indices.add(idx1)
        node_indices.add(idx2)
        idx_to_obj_local[idx1] = obj1
        idx_to_obj_local[idx2] = obj2

        # Store GLOBAL indices temporarily
        edge_index.append([idx1, idx2])

        triplet_data.append((obj1, predicate, obj2))
        edge_labels.append(get_edge_label(triplet, ground_truths))

    # Map to local indices
    node_indices = sorted(list(node_indices))
    global_to_local = {g: l for l, g in enumerate(node_indices)}
    edge_index_local = [[global_to_local[i], global_to_local[j]] for i, j in edge_index]

    return {
        "edge_index_local": edge_index_local,
        "edge_labels": edge_labels,
        "node_indices": node_indices,
        "idx_to_obj_local": idx_to_obj_local,
        "triplet_data": triplet_data,
    }


def create_graph_data_glove(
    triplets,
    glove: KeyedVectors,
    data_sum: DatasetSummary,
    graph_label: int,
    ground_truths: Dict,
    mutation_rate: float = None,
    noise_std: float = 0.00,
):
    """noise_std = 0.0 means normal generation without adding any noise."""
    graph_data = _prepare_graph_data(triplets, data_sum, ground_truths, mutation_rate)

    # Edge features: average GloVe embeddings of predicate words
    edge_attr = []
    for obj1, predicate, obj2 in graph_data["triplet_data"]:
        edge_attr.append(
            torch.tensor(_embed_word_avg(glove, predicate), dtype=torch.float)
        )

    # Node features: GloVe embeddings
    node_features = []
    for global_idx in graph_data["node_indices"]:
        obj_name = graph_data["idx_to_obj_local"][global_idx]
        node_features.append(
            torch.tensor(_embed_word_avg(glove, obj_name), dtype=torch.float)
        )

    # Stack into tensors
    node_features = torch.stack(node_features)
    edge_attr = torch.stack(edge_attr)

    # Add Gaussian noise if requested
    if noise_std > 0:
        node_features = node_features + torch.randn_like(node_features) * noise_std
        edge_attr = edge_attr + torch.randn_like(edge_attr) * noise_std

    feature_info = {
        "num_objects": len(data_sum.all_objects),
        "num_concepts": len(data_sum.all_concepts),
        "node_feature_dim": node_features.shape[1],
        "edge_dim": edge_attr.shape[1],
    }

    data = Data(
        x=node_features,
        edge_index=torch.tensor(graph_data["edge_index_local"], dtype=torch.long)
        .t()
        .contiguous(),
        edge_attr=edge_attr,
        y=torch.tensor([graph_label], dtype=torch.long),
        edge_y=torch.tensor(graph_data["edge_labels"], dtype=torch.float),
        num_nodes=len(node_features),
        global_node_indices=torch.tensor(graph_data["node_indices"]),
    )

    return data, feature_info


def create_graph_data_llm(
    triplets,
    llm_embedder: CachedLLMEmbedder,
    data_sum: DatasetSummary,
    graph_label: int,
    ground_truths: Dict,
    scene_type: str = None,
    mutation_rate: float = None,
):
    graph_data = _prepare_graph_data(triplets, data_sum, ground_truths, mutation_rate)

    # Edge features: embed full triplets
    edge_embeddings = []
    for obj1, predicate, obj2 in graph_data["triplet_data"]:
        emb = llm_embedder.embed_triplet(obj1, predicate, obj2)
        # emb = llm_embedder.embed_triplet(obj1, predicate, obj2, scene_type)
        edge_embeddings.append(emb)

    # Node features: embed object names
    node_embeddings = []
    for global_idx in graph_data["node_indices"]:
        obj_name = graph_data["idx_to_obj_local"][global_idx]
        emb = llm_embedder.embed_triplet(obj_name, "", "")
        node_embeddings.append(emb)

    node_features = torch.stack(node_embeddings)
    edge_attr = torch.stack(edge_embeddings)

    feature_info = {
        "num_objects": len(data_sum.all_objects),
        "num_concepts": len(data_sum.all_concepts),
        "node_feature_dim": node_features.shape[1],
        "edge_dim": edge_attr.shape[1],
    }

    data = Data(
        x=node_features,
        edge_index=torch.tensor(graph_data["edge_index_local"], dtype=torch.long)
        .t()
        .contiguous(),
        edge_attr=edge_attr,
        y=torch.tensor([graph_label], dtype=torch.long),
        edge_y=torch.tensor(graph_data["edge_labels"], dtype=torch.float),
        num_nodes=len(node_features),
        global_node_indices=torch.tensor(graph_data["node_indices"]),
    )

    return data, feature_info


if __name__ == "__main__":
    print("Don't call function like this!")
