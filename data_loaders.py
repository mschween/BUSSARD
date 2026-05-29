import json
import random
from itertools import chain
from typing import Dict, List, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from scene_graph_dataset import SceneGraphDataset
from utils import set_seed


def create_train_loader(
    config_path: str,
    scene_name: str,
    seed: int = 42,
    use_anomal_training: bool = False,
    use_autoencoder: bool = False,
    autoencoder_path: str = None,
    autoencoder_latent_dim: int = 512,
    mutation_rate: float = None,
    word_embed=None,
    embed_type: str = "glove",
    dataset_source: str = "sard",
) -> Tuple[DataLoader, Dict]:
    """Create only training data loader."""

    # Load config
    with open(config_path, "r") as f:
        all_config = json.load(f)

    config = all_config["dataset"]
    batch_size = config.get("batch_size", 512)
    shuffle = config.get("shuffle", True)
    test_ratio = config.get("test_ratio", 0.2)

    # Handle "all"
    if dataset_source == "mit-67" and scene_name == "all":
        scene_names = ["dining_room", "office"]
    elif dataset_source == "sard" and scene_name == "all":
        scene_names = ["dining_room", "office"]
    else:
        scene_names = [scene_name]

    # Collect data from all scenes
    all_subsets = []
    feature_info = None

    print(f"Loading training data from {len(scene_names)} scenes...")

    for scene in scene_names:
        if len(scene_names) > 1:
            print(f"  Loading {scene}...")

        # Create dataset for this scene
        train_dataset = SceneGraphDataset(
            config_path,
            scene_name=scene,
            mutation_rate=mutation_rate,
            word_embed=word_embed,
            embed_type=embed_type,
            dataset_source=dataset_source,
        )

        print(
            f"{scene}: total={len(train_dataset)}, normal_idx={len(train_dataset.normal_idx)}"
        )

        # Get training indices based on dataset source
        if dataset_source == "sard":
            # Split SARD data into train/test
            train_normal, _ = train_test_split(
                train_dataset.normal_idx, test_size=test_ratio, random_state=seed
            )

            if use_anomal_training:
                train_anomal, _ = train_test_split(
                    train_dataset.anomal_idx, test_size=test_ratio, random_state=seed
                )
                train_idx = train_normal + train_anomal
            else:
                train_idx = train_normal
        else:
            # VG or Indoor
            # Use all data for training (no split for cross-dataset generalization)
            train_idx = train_dataset.normal_idx

        # Create subset
        train_subset = Subset(train_dataset, train_idx)

        # Apply autoencoder if needed
        if use_autoencoder:
            train_subset = _apply_autoencoder_to_subset(
                train_subset,
                autoencoder_path,
                autoencoder_latent_dim,
                embed_type,
                all_config,
            )

        # Append encoded subset
        all_subsets.append(train_subset)

        if feature_info is None:
            feature_info = train_dataset.feature_info

        # Cleanup
        train_dataset.cleanup_embeddings()

    # Combine all subsets (works for both single and multiple scenes)
    if len(all_subsets) == 1:
        # Single scene - use subset directly
        train_loader = DataLoader(
            all_subsets[0], batch_size=batch_size, shuffle=shuffle
        )
    else:
        # Multiple scenes - combine them
        combined_data = list(chain(*all_subsets))
        train_loader = DataLoader(combined_data, batch_size=batch_size, shuffle=shuffle)
        print(
            f"Combined training set: {len(combined_data)} samples from {len(scene_names)} scenes"
        )

    return train_loader, feature_info


def create_test_loader(
    config_path: str,
    scene_name: str,
    seed: int = 42,
    use_anomal_training: bool = False,
    use_autoencoder: bool = False,
    autoencoder_path: str = None,
    autoencoder_latent_dim: int = 512,
    mutation_rate: float = None,
    word_embed=None,
    embed_type: str = "glove",
    dataset_source: str = "sard",
) -> Tuple[DataLoader, Dict]:
    """Create only test data loader."""

    # Create dataset
    test_dataset = SceneGraphDataset(
        config_path,
        scene_name=scene_name,
        mutation_rate=mutation_rate,
        word_embed=word_embed,
        embed_type=embed_type,
        dataset_source=dataset_source,
    )

    # Load config
    with open(config_path, "r") as f:
        all_config = json.load(f)

    config = all_config["dataset"]
    batch_size = config.get("batch_size", 32)
    test_ratio = config.get("test_ratio", 0.2)

    # Get test indices based on dataset source
    if dataset_source == "sard":
        # Split SARD data - take test portion
        _, test_normal = train_test_split(
            test_dataset.normal_idx, test_size=test_ratio, random_state=seed
        )

        if use_anomal_training:
            _, test_anomal = train_test_split(
                test_dataset.anomal_idx, test_size=test_ratio, random_state=seed
            )
        else:
            # For SARD, test set includes all anomalies
            test_anomal = test_dataset.anomal_idx

        test_idx = test_normal + test_anomal
    else:
        # For cross-dataset: use all SARD data for testing
        # For same-dataset: this shouldn't be called (use efficient path instead)
        test_idx = list(range(len(test_dataset)))

    test_subset = Subset(test_dataset, test_idx)

    # Apply autoencoder if needed
    if use_autoencoder:
        test_subset = _apply_autoencoder_to_subset(
            test_subset,
            autoencoder_path,
            autoencoder_latent_dim,
            embed_type,
            all_config,
        )

    # Create loader
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # Cleanup
    test_dataset.cleanup_embeddings()

    return test_loader, test_dataset.feature_info


def create_sard_subset_loaders(
    config_path: str,
    scene_name: str,
    seed: int = 42,
    use_anomal_training: bool = False,
    normal_samples_per_anomaly: int = 10,
    use_autoencoder: bool = False,
    autoencoder_path: str = None,
    autoencoder_latent_dim: int = 512,
    mutation_rate: float = None,
    word_embed=None,
    embed_type: str = "glove",
) -> List[DataLoader]:
    """Create SARD-style anomaly subset loaders."""

    # Create dataset
    dataset = SceneGraphDataset(
        config_path,
        scene_name=scene_name,
        mutation_rate=mutation_rate,
        word_embed=word_embed,
        embed_type=embed_type,
        dataset_source="sard",
    )

    # Load config
    with open(config_path, "r") as f:
        all_config = json.load(f)

    config = all_config["dataset"]
    test_ratio = config.get("test_ratio", 0.2)

    # Get test split indices (same as create_test_loader)
    _, test_normal = train_test_split(
        dataset.normal_idx, test_size=test_ratio, random_state=seed
    )
    if use_anomal_training:
        _, test_anomal = train_test_split(
            dataset.anomal_idx, test_size=test_ratio, random_state=seed
        )
    else:
        # For SARD, test set includes all anomalies
        test_anomal = dataset.anomal_idx

    # Apply autoencoder if needed
    if use_autoencoder:
        dataset = _apply_autoencoder(
            dataset, autoencoder_path, autoencoder_latent_dim, embed_type, all_config
        )

    # Create subset loaders
    anomaly_subset_loaders = []
    for anomaly_idx_single in test_anomal:
        # Randomly sample normal samples for this anomaly
        sampled_normal = random.sample(
            test_normal, min(normal_samples_per_anomaly, len(test_normal))
        )
        subset_indices = [anomaly_idx_single] + sampled_normal

        anomaly_subset = Subset(dataset, subset_indices)
        # Each batch is one complete graph
        subset_loader = DataLoader(anomaly_subset, batch_size=1, shuffle=False)
        anomaly_subset_loaders.append(subset_loader)

    # Cleanup
    dataset.cleanup_embeddings()

    return anomaly_subset_loaders


def _create_sard_subsets_from_dataset(
    dataset, test_normal, test_anomal, normal_samples_per_anomaly
):
    """Create SARD subset loaders from existing dataset."""
    anomaly_subset_loaders = []
    for anomaly_idx_single in test_anomal:
        # Randomly sample normal samples for this anomaly
        sampled_normal = random.sample(
            test_normal, min(normal_samples_per_anomaly, len(test_normal))
        )
        subset_indices = [anomaly_idx_single] + sampled_normal

        anomaly_subset = Subset(dataset, subset_indices)
        # Each batch is one complete graph
        subset_loader = DataLoader(anomaly_subset, batch_size=1, shuffle=False)
        anomaly_subset_loaders.append(subset_loader)

    return anomaly_subset_loaders


# Helper functions
def _apply_autoencoder(dataset, autoencoder_path, latent_dim, embed_type, all_config):
    """Apply autoencoder encoding to dataset."""
    from autoencoder import Autoencoder, concatenate_features

    embed_config = all_config["embeddings"][embed_type]
    raw_dim = 3 * embed_config["dimension"]

    autoencoder = Autoencoder(raw_dim=raw_dim, latent_dim=latent_dim)
    autoencoder.load_state_dict(torch.load(autoencoder_path))
    autoencoder.freeze_weights()
    autoencoder.eval()

    print(f"Encoding {len(dataset)} graphs...")
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset.data_list[i]
            # Encode node features (x) while keeping everything else,
            # except for the edge features
            concatenated = concatenate_features(data)
            encoded_x = autoencoder.encode(concatenated)

            # Create new Data object with encoded features
            encoded_data = Data(
                x=encoded_x,  # Encoded node features
                edge_index=data.edge_index,
                y=data.y,
                edge_y=data.edge_y,
                global_node_indices=data.global_node_indices,
                num_nodes=data.num_nodes,
                batch=data.batch if hasattr(data, "batch") else None,
                ptr=data.ptr if hasattr(data, "ptr") else None,
            )

            # Replace the data in dataset
            dataset.data_list[i] = encoded_data

    print(f"Data encoded from {raw_dim}D to {latent_dim}D")
    return dataset


def _apply_autoencoder_to_subset(
    subset, autoencoder_path, latent_dim, embed_type, all_config
):
    """Apply autoencoder to a Subset."""
    from autoencoder import Autoencoder, concatenate_features

    embed_config = all_config["embeddings"][embed_type]
    raw_dim = 3 * embed_config["dimension"]

    autoencoder = Autoencoder(raw_dim=raw_dim, latent_dim=latent_dim)
    autoencoder.load_state_dict(torch.load(autoencoder_path))
    autoencoder.freeze_weights()
    autoencoder.eval()

    print(f"Encoding {len(subset)} graphs...")
    with torch.no_grad():
        for i in range(len(subset)):
            data = subset[i]  # Gets data through Subset's __getitem__
            concatenated = concatenate_features(data)
            encoded_x = autoencoder.encode(concatenated)

            # Get the actual index in the original dataset
            actual_idx = subset.indices[i]

            # Update in the underlying dataset
            subset.dataset.data_list[actual_idx] = Data(
                x=encoded_x,
                edge_index=data.edge_index,
                y=data.y,
                edge_y=data.edge_y,
                global_node_indices=data.global_node_indices,
                num_nodes=data.num_nodes,
            )

    print(f"Encoded from {raw_dim}D to {latent_dim}D")
    return subset


# Keep original function for backward compatibility
def create_data_loaders(
    config_path: str,
    scene_name: str,
    seed: int = 42,
    use_anomal_training: bool = False,
    anomaly_subset_mode: bool = False,
    normal_samples_per_anomaly: int = 10,
    use_autoencoder: bool = False,
    autoencoder_path: str = None,
    autoencoder_latent_dim: int = 512,
    mutation_rate: float = None,
    word_embed=None,
    embed_type: str = "glove",
    dataset_source: str = "sard",
    test_on_sard: bool = True,
):
    """
    Create train and test data loaders efficiently - loads dataset only once when possible.
    Args:
        config_path: Path to config
        scene_name: Name of current scene
        seed: Seed used for train_test_split()
        use_anomal_training: Flag if only normal, or both data should be used for training
        anomaly_subset_mode: If True, returns additional anomaly subset loaders
        normal_samples_per_anomaly: Number of normal samples per anomaly in subsets
        use_autoencoder: If True, encode features using trained autoencoder
        autoencoder_path: Path to saved autoencoder weights (required if use_autoencoder=True)
        autoencoder_latent_dim: Latent dimension of the autoencoder
        mutation_rate
        embed_type: Can be either "glove" or name of llm for embedding for two different embedding methods
        dataset_source: If set to visual_genome trains on vg and tests on SARD
        test_on_sard: flag for autoencoder, that only needs visual genome
    Returns:
        train_loader, test_loader, feature_info
        OR train_loader, test_loader, anomaly_subset_loaders, feature_info (if anomaly_subset_mode=True)
    """
    set_seed(seed)

    with open(config_path, "r") as f:
        all_config = json.load(f)

    config = all_config["dataset"]
    batch_size = config.get("batch_size", 512)
    shuffle = config.get("shuffle", True)
    test_ratio = config.get("test_ratio", 0.2)

    # Check if we can use same dataset for train and test
    same_dataset = (dataset_source == "sard" and test_on_sard) or (
        dataset_source != "sard" and not test_on_sard
    )

    if same_dataset and scene_name != "all":
        # Load once, split indices
        dataset = SceneGraphDataset(
            config_path,
            scene_name=scene_name,
            mutation_rate=mutation_rate,
            word_embed=word_embed,
            embed_type=embed_type,
            dataset_source=dataset_source,
        )

        # Split indices
        if dataset_source == "sard":
            train_normal, test_normal = train_test_split(
                dataset.normal_idx, test_size=test_ratio, random_state=seed
            )

            if use_anomal_training:
                train_anomal, test_anomal = train_test_split(
                    dataset.anomal_idx, test_size=test_ratio, random_state=seed
                )
                train_idx = train_normal + train_anomal
            else:
                train_idx = train_normal
                test_anomal = dataset.anomal_idx  # All anomalies in test

            test_idx = test_normal + test_anomal
        else:
            # For VG/Indoor: split all data into train/test
            all_indices = list(range(len(dataset)))
            train_idx, test_idx = train_test_split(
                all_indices, test_size=test_ratio, random_state=seed
            )

        # Create subsets
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        # Apply autoencoder to subsets
        if use_autoencoder:
            train_subset = _apply_autoencoder_to_subset(
                train_subset,
                autoencoder_path,
                autoencoder_latent_dim,
                embed_type,
                all_config,
            )
            test_subset = _apply_autoencoder_to_subset(
                test_subset,
                autoencoder_path,
                autoencoder_latent_dim,
                embed_type,
                all_config,
            )

        # Create loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        feature_info = dataset.feature_info

        # Handle anomaly subsets
        if anomaly_subset_mode:
            subset_loaders = _create_sard_subsets_from_dataset(
                dataset, test_normal, test_anomal, normal_samples_per_anomaly
            )
            dataset.cleanup_embeddings()
            return train_loader, test_loader, subset_loaders, feature_info

        dataset.cleanup_embeddings()
        return train_loader, test_loader, feature_info

    else:
        # Different datasets for train/test or multi-scene
        train_loader, feature_info = create_train_loader(
            config_path,
            scene_name,
            seed,
            use_anomal_training,
            use_autoencoder,
            autoencoder_path,
            autoencoder_latent_dim,
            mutation_rate,
            word_embed,
            embed_type,
            dataset_source,
        )

        test_loader, _ = create_test_loader(
            config_path,
            scene_name,
            seed,
            use_anomal_training,
            use_autoencoder,
            autoencoder_path,
            autoencoder_latent_dim,
            mutation_rate,
            word_embed,
            embed_type,
            dataset_source if not test_on_sard else "sard",
        )

        if anomaly_subset_mode:
            subset_loaders = create_sard_subset_loaders(
                config_path,
                scene_name,
                seed,
                use_anomal_training,
                normal_samples_per_anomaly,
                use_autoencoder,
                autoencoder_path,
                autoencoder_latent_dim,
                mutation_rate,
                word_embed,
                embed_type,
            )
            return train_loader, test_loader, subset_loaders, feature_info

        return train_loader, test_loader, feature_info


if __name__ == "__main__":
    # Example usage
    config_path = "config.json"
    scene_name = "office"

    # Create data loaders
    train_loader, test_loader, feature_info = create_data_loaders(
        config_path,
        scene_name=scene_name,
        use_anomal_training=True,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
