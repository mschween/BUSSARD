import csv
import gc
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, roc_auc_score

from data_loaders import (
    create_data_loaders,
    create_sard_subset_loaders,
    create_test_loader,
    create_train_loader,
)
from flow import UnsupervisedFlow
from trainer import train_flexible
from utils import date_folders, set_seed


def recall_auc_at_k(flow_scores, all_labels, results, k_range=[1, 100]):
    k_values = range(k_range[0], k_range[1] + 1)
    recall_at_ks = []

    # Sort decending by score
    sorted_indices = np.argsort(-flow_scores)
    labels_sorted = all_labels[sorted_indices]
    total_positives = np.sum(all_labels)

    for k in k_values:
        if k > len(labels_sorted) or total_positives == 0:
            recall_at_ks.append(0.0)
        else:
            top_k_labels = labels_sorted[:k]
            recall_k = np.sum(top_k_labels) / total_positives
            recall_at_ks.append(recall_k)

    # AUC over Recall@K curve
    auc_at_k = auc(k_values, recall_at_ks)

    results["metrics"]["auc_at_k"] = auc_at_k
    # Store as flat dictionary for easy CSV writing
    for k, r in zip(k_values, recall_at_ks):
        results["metrics"][f"recall@{k}"] = r

    return results, recall_at_ks, auc_at_k


def detect_anomalies(model: UnsupervisedFlow, test_loader, results_out):
    """
    Detect anomalies in test data

    Args:
        model: Trained UnsupervisedFlow model
        test_loader: Test data loader (can contain both normal and anomalous edges)
        results_out: Output path for results

    Returns:
        Dictionary with scores, predictions, and metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    flow_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Compute anomaly scores
            scores_dict = model.compute_anomaly_scores(batch)

            flow_scores.extend(scores_dict["flow_scores"].cpu().numpy())

            # Store labels
            all_labels.extend(batch.edge_y.cpu().numpy())

    flow_scores = np.array(flow_scores)

    results = {
        "anomaly_scores": flow_scores,
    }

    # Compute metrics
    all_labels = np.array(all_labels)
    auc_val = roc_auc_score(all_labels, flow_scores)
    ap = average_precision_score(all_labels, flow_scores)

    results["metrics"] = {
        "auc": auc_val,
        "average_precision": ap,
    }

    # Recall @ k & AUC @ k
    results, recall_k, auc_k = recall_auc_at_k(
        flow_scores=flow_scores, all_labels=all_labels, results=results
    )

    # Writing to CSV
    with open(results_out, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(results["metrics"].keys())  # Write header
        writer.writerow(results["metrics"].values())  # Write values

    return results


def sard_eval(model, anomaly_subset_loaders):
    """
    SARD-compatible evaluation using graph-level labels.
    Each subset contains N graphs (N-1 normal + 1 anomalous).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    ranks = []
    flow_scores, all_labels = [], []

    with torch.no_grad():
        # Each loader represents one "image set" (N-1 normal + 1 anomalous)
        for subset_loader in anomaly_subset_loaders:
            subset_edges = []  # Store (edge_score, edge_label, graph_label)

            for batch in subset_loader:
                batch = batch.to(device)
                scores_dict = model.compute_anomaly_scores(batch)
                edge_scores = scores_dict["flow_scores"].cpu().numpy()
                edge_labels = batch.edge_y.cpu().numpy()
                graph_label = batch.y.item()

                # Store each edge with its score and labels
                for score, edge_label in zip(edge_scores, edge_labels):
                    subset_edges.append((score, edge_label, graph_label))
                    flow_scores.append(score)
                    all_labels.append(edge_label)

            # Sort edges by anomaly score (descending)
            subset_edges.sort(key=lambda x: -x[0])

            # Find rank of first anomalous edge from the anomalous graph
            rank = -1
            for i, (score, edge_label, graph_label) in enumerate(subset_edges):
                # Edge must be anomalous (edge_label=1) AND from anomalous graph
                if edge_label == 1 and graph_label == 1:
                    rank = i + 1
                    break

            if rank == -1:
                print(
                    f"Warning: No anomalous graph found in subset - Skipping this sample for evaluation."
                )
                continue

            ranks.append(rank)

    # Calculate SARD-style metrics with locality constraint
    k_vals = range(1, 101)
    recall_k = []
    for k in k_vals:
        # Success rate: fraction of subsets where anomaly found in top-k
        success_rate = sum(1 for r in ranks if 0 < r <= k) / len(ranks)
        recall_k.append(success_rate)

    sard_auc_k = auc(k_vals, recall_k)

    # Global ROC-AUC
    global_auc = roc_auc_score(all_labels, flow_scores)

    return sard_auc_k, global_auc, recall_k


def main(
    config: str,
    scene_name: str = "office",  # Can be "all" or single scene
    flow_name: str = "RealNVP",
    seed: int = 0,
    hidden_dim: int = 512,
    epochs: int = 1000,
    out_path="output/flow",
    simulate_sard=True,
    cross_check: bool = False,
    flow_input_dim=512,
    mutation_rate=None,
    word_embed=None,
    embed_type="glove",
    dataset_source="sard",
):
    if dataset_source == "visual_genome":
        autoencoder_path = f"output/autoencoder_weights/{embed_type}/autoencoder_{flow_input_dim}_vg.pth"
    elif dataset_source == "mit-67":
        autoencoder_path = f"output/autoencoder_weights/{embed_type}/autoencoder_{flow_input_dim}_balanced_id_domain.pth"
    else:
        autoencoder_path = f"output/autoencoder_weights/{embed_type}/autoencoder_{flow_input_dim}_{scene_name}.pth"

    # Determine if we're doing cross-dataset evaluation
    cross_dataset = dataset_source == "visual_genome" or dataset_source == "mit-67"

    # Determine test scene
    if scene_name == "all" and dataset_source == "mit-67":
        test_scene_name = "office"  # Test on SARD office -> Doesn't matter which one, when using cross-eval setting
        test_dataset_source = "sard"
    elif scene_name == "all" and dataset_source == "sard":
        test_scene_name = "office"
        test_dataset_source = "sard"
    else:
        test_scene_name = scene_name
        test_dataset_source = dataset_source

    # Data loading
    print(f"Loading data...")
    # Use create_data_loaders when train/test use same dataset
    if not cross_dataset and scene_name != "all":
        # Single scene, same dataset for train/test
        if simulate_sard:
            train_loader, test_loader, anomaly_subset_loaders, feature_info = (
                create_data_loaders(
                    config_path=config,
                    scene_name=scene_name,
                    seed=seed,
                    use_anomal_training=False,
                    anomaly_subset_mode=True,
                    normal_samples_per_anomaly=10,
                    use_autoencoder=True,
                    autoencoder_path=autoencoder_path,
                    autoencoder_latent_dim=flow_input_dim,
                    mutation_rate=mutation_rate,
                    word_embed=word_embed,
                    embed_type=embed_type,
                    dataset_source=dataset_source,
                    test_on_sard=True,
                )
            )
        else:
            train_loader, test_loader, feature_info = create_data_loaders(
                config_path=config,
                scene_name=scene_name,
                seed=seed,
                use_anomal_training=False,
                anomaly_subset_mode=False,
                use_autoencoder=True,
                autoencoder_path=autoencoder_path,
                autoencoder_latent_dim=flow_input_dim,
                mutation_rate=mutation_rate,
                word_embed=word_embed,
                embed_type=embed_type,
                dataset_source=dataset_source,
                test_on_sard=True,
            )
    else:
        # Multi-scene or cross-dataset
        train_loader, feature_info = create_train_loader(
            config_path=config,
            scene_name=scene_name,
            seed=seed,
            use_anomal_training=False,
            use_autoencoder=True,
            autoencoder_path=autoencoder_path,
            autoencoder_latent_dim=flow_input_dim,
            mutation_rate=mutation_rate,
            word_embed=word_embed,
            embed_type=embed_type,
            dataset_source=dataset_source,
        )

        test_loader, _ = create_test_loader(
            config_path=config,
            scene_name=test_scene_name,
            seed=seed,
            use_anomal_training=False,
            use_autoencoder=True,
            autoencoder_path=autoencoder_path,
            autoencoder_latent_dim=flow_input_dim,
            mutation_rate=mutation_rate,
            word_embed=word_embed,
            embed_type=embed_type,
            dataset_source=test_dataset_source,
        )

        if simulate_sard:
            anomaly_subset_loaders = create_sard_subset_loaders(
                config_path=config,
                scene_name=test_scene_name,
                seed=seed,
                use_anomal_training=False,
                normal_samples_per_anomaly=10,
                use_autoencoder=True,
                autoencoder_path=autoencoder_path,
                autoencoder_latent_dim=flow_input_dim,
                mutation_rate=mutation_rate,
                word_embed=word_embed,
                embed_type=embed_type,
            )

    arch_path = Path(out_path, flow_name)
    arch_path.mkdir(parents=True, exist_ok=True)
    # Create unsupervised model
    model = UnsupervisedFlow(
        hidden_dim=hidden_dim,
        flow_name=flow_name,
        flow_input_dim=flow_input_dim,
        use_running_norm=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Flow created with {sum(p.numel() for p in model.parameters())} parameters")

    model, log_path = train_flexible(
        model=model,
        train_loader=train_loader,
        epochs=epochs,
        lr=1e-4,
        out_path=arch_path,
    )

    # Compute statistics for correct anomaly detection
    model.compute_and_store_training_stats(train_loader=train_loader)

    results_out = arch_path / f"metrics.csv"
    # Detect anomalies in test set
    results = detect_anomalies(model, test_loader, results_out=results_out)

    if simulate_sard:
        sard_auc_at_k, sard_auc, recall_at_k = sard_eval(model, anomaly_subset_loaders)

        print(f"SARD like AUC: {sard_auc:.4f}")
        print(f"SARD like AUC @ k: {sard_auc_at_k:.4f}")
        print(f"SARD Recall@1: {recall_at_k[0]:.4f}")

        # Writing to CSV
        with open(results_out, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["sard_like_auc", "sard_like_auc_at_k", "recall_at_1"])
            writer.writerow([sard_auc, sard_auc_at_k, recall_at_k[0]])

            # Write recall@k values (k from 1 to 100)
            recall_k_headers = [f"s_recall@{k}" for k in range(1, 101)]
            writer.writerow(recall_k_headers)
            writer.writerow(recall_at_k)

        if cross_check:
            # make cross check with not trained scene
            if test_scene_name == "office":
                other_name = "dining_room"
            else:
                other_name = "office"

            # Use the same autoencoder (trained on scene_name) for cross-scene evaluation
            cross_anomaly_subset_loaders = create_sard_subset_loaders(
                config_path=config,
                scene_name=other_name,
                seed=seed,
                use_autoencoder=True,
                autoencoder_path=autoencoder_path,
                autoencoder_latent_dim=flow_input_dim,
                mutation_rate=mutation_rate,
                embed_type=embed_type,
            )

            cross_sard_auc_at_k, cross_sard_auc, cross_recall_at_k = sard_eval(
                model, cross_anomaly_subset_loaders
            )

            print(f"Cross SARD like AUC: {cross_sard_auc:.4f}")
            print(f"Cross SARD like AUC @ k: {cross_sard_auc_at_k:.4f}")
            print(f"Cross SARD Recall@1: {cross_recall_at_k[0]:.4f}\n\n")

            # Writing to CSV
            with open(results_out, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "cross_sard_like_auc",
                        "cross_sard_like_auc_at_k",
                        "cross_recall_at_1",
                    ]
                )
                writer.writerow(
                    [cross_sard_auc, cross_sard_auc_at_k, cross_recall_at_k[0]]
                )
    gc.collect()  # Python garbage collection
    torch.cuda.empty_cache()  # Free GPU memory


# Example usage
if __name__ == "__main__":
    log_dir = "output/flow"
    out_path = date_folders(base_dir=log_dir)
    seed = 0
    set_seed(seed)
    main(
        config="config.json",
        seed=seed,
        out_path=out_path,
        scene_name="office",
        embed_type="glove",
    )
