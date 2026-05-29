import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data_loaders import create_train_loader
from utils import set_seed


class Autoencoder(nn.Module):
    def __init__(self, raw_dim=900, latent_dim=64):
        super().__init__()
        hidden_dim = 128
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, raw_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True


def concatenate_features(data):
    # Concatenate node features for each edge
    edge_indices = data.edge_index
    node_features_i = data.x[edge_indices[0]]  # Features of source nodes
    node_features_j = data.x[edge_indices[1]]  # Features of target nodes

    # Concatenate: [edge_features, source_node_features, target_node_features]
    concatenated = torch.cat([data.edge_attr, node_features_i, node_features_j], dim=1)

    # Normalize concatenated features
    mean = concatenated.mean(dim=0, keepdim=True)
    std = concatenated.std(dim=0, keepdim=True)

    # Add larger epsilon and check for zero std
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    concatenated = (concatenated - mean) / std

    return concatenated


def train_autoencoder(model, data_loader, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch in data_loader:
            data = concatenate_features(batch)
            optimizer.zero_grad()

            reconstructed = model(data)
            loss = criterion(reconstructed, data)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.6f}")


def evaluate_reconstruction(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()

    total_mse = 0
    total_mae = 0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            data = concatenate_features(batch)
            reconstructed = model(data)
            mse_loss = criterion(reconstructed, data)
            mae_loss = torch.mean(torch.abs(reconstructed - data))

            total_mse += mse_loss.item()
            total_mae += mae_loss.item()
            num_batches += 1

    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches

    print(f"Reconstruction MSE: {avg_mse:.6f}")
    print(f"Reconstruction MAE: {avg_mae:.6f}")
    return avg_mse, avg_mae


def main(
    config="config.json",
    scene_name="office",
    seed=42,
    epochs=100,
    embed_type="glove",
    dataset_source="sard",
):
    set_seed(seed)

    # Load config
    with open(config, "r") as f:
        embed_config = json.load(f)["embeddings"][embed_type]

    if embed_type == "glove":
        raw_dim = 3 * 300
    else:
        # Load dim from config
        raw_dim = 3 * embed_config["dimension"]

    train_loader, feature_info = create_train_loader(
        config_path=config,
        scene_name=scene_name,
        seed=seed,
        use_anomal_training=False,
        embed_type=embed_type,
        dataset_source=dataset_source,
    )

    # Split train_loader for train/validation internally
    # Get the dataset from the loader
    train_dataset = train_loader.dataset
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader_split = DataLoader(
        train_subset, batch_size=train_loader.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=train_loader.batch_size, shuffle=False
    )

    # latent_dims = [16, 32, 64, 128, 256, 512, 900]
    latent_dims = [512]

    for latent_dim in latent_dims:
        print(f"Current dim: {latent_dim}")

        model = Autoencoder(raw_dim=raw_dim, latent_dim=latent_dim)

        train_autoencoder(model=model, data_loader=train_loader_split, epochs=epochs)
        evaluate_reconstruction(model=model, test_loader=val_loader)

        # After training, freeze weights for inference
        model.freeze_weights()

        # Create directory if it doesn't exist
        ae_path = Path(f"output/autoencoder_weights/{embed_type}")
        ae_path.mkdir(parents=True, exist_ok=True)

        # Use dataset_source in filename
        if dataset_source == "visual_genome":
            filename = f"autoencoder_{latent_dim}_vg.pth"
        elif dataset_source == "mit-67":
            # filename = f"autoencoder_{latent_dim}_id.pth"
            filename = f"autoencoder_{latent_dim}_balanced_mit.pth"  # For only office & dining room
        else:
            filename = f"autoencoder_{latent_dim}_{scene_name}.pth"

        ae_path = Path(ae_path, filename)

        # Save the trained model
        torch.save(
            model.state_dict(),
            ae_path,
        )


if __name__ == "__main__":
    scenes = ["office", "dining_room"]
    for scene in scenes:
        print(f"Training autoencoder on SARD {scene}")
        main(scene_name=scene, embed_type="glove", dataset_source="sard")

    # Enable if MIT-67 dataset should be used
    # print("Training autoencoder on mit-67")
    # main(scene_name="all", epochs=100, embed_type="glove", dataset_source="mit-67")
