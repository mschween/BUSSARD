import csv
from collections import deque
from pathlib import Path

import torch

from flow import UnsupervisedFlow


class FlexibleFlowTrainer:
    """Trainer supporting Normalizing Flow models"""

    def __init__(self, model: UnsupervisedFlow, lr):
        """
        Args:
            model: Flow model
            lr: Base learning rate
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.device = device

        # Setup optimizers
        flow_params = list(model.flow.parameters())

        self.flow_optimizer = torch.optim.AdamW(
            flow_params, lr=lr, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-8
        )

        self.flow_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.flow_optimizer,
            mode="min",
            factor=0.8,
            patience=30,
            min_lr=1e-7,
            verbose=True,
        )

        # Loss tracking
        self.loss_history = deque(maxlen=100)
        # Track best losses for stability monitoring
        self.best_flow_loss = float("inf")

    def check_for_collapse(self, train_loader, epoch, check_interval=10, threshold=15):
        """Check if flow is collapsing to zeros"""
        # Check every check_interval epochs
        if epoch == 0 or epoch % check_interval != 0:
            return False

        self.model.eval()
        with torch.no_grad():
            zeros_found = 0
            samples_checked = 0

            for i, batch in enumerate(train_loader):
                if i >= 5:  # Check 5 batches
                    break
                batch = batch.to(self.device)
                outputs = self.model(batch, return_all=True)
                log_probs = outputs["flow_log_probs"]

                zeros_found += (log_probs == 0).sum().item()
                samples_checked += len(log_probs)

            print(
                f"[Epoch {epoch}] Zero check: {zeros_found}/{samples_checked} "
                f"({100*zeros_found/samples_checked:.1f}%)"
            )

            if zeros_found > threshold:
                print(f"Collapse detected! {zeros_found} > {threshold}")
                return True

        self.model.train()
        return False

    def train_epoch(self, train_loader):
        self.model.train()

        epoch_stats = {
            "flow_loss": 0,
            "batches_processed": 0,
        }

        for batch_idx, batch in enumerate(train_loader):
            # Forward pass
            outputs = self.model(batch, return_all=True)

            flow_input = outputs["flow_input"]
            flow_log_probs = outputs["flow_log_probs"]

            # Flow loss
            flow_loss = -flow_log_probs.mean()

            # Check for NaN/Inf in loss
            if torch.isnan(flow_loss) or torch.isinf(flow_loss):
                raise ValueError(f"Warning: Invalid loss at batch {batch_idx}")

            # Backward pass
            self.flow_optimizer.zero_grad()
            flow_loss.backward()

            # Gradient clipping with fixed values
            flow_clip = 0.5

            torch.nn.utils.clip_grad_norm_(
                self.model.flow.parameters(), max_norm=flow_clip
            )

            # Optimizer steps
            self.flow_optimizer.step()

            # Update statistics
            epoch_stats["flow_loss"] += flow_loss.item()
            epoch_stats["batches_processed"] += 1

            # Update loss history
            self.loss_history.append(flow_loss.item())

            self.loss_history.append(flow_loss.item())

        # Compute averages
        if epoch_stats["batches_processed"] > 0:
            epoch_stats["flow_loss"] /= epoch_stats["batches_processed"]

        return epoch_stats


def train_flexible(
    model: UnsupervisedFlow,
    train_loader,
    epochs=100,
    lr=1e-3,
    out_path="output/flow/",
    checkpoint_freq=500,
):
    """
    Flexible training function supporting both joint and sequential training

    Args:
        model: Combined Flow model
        train_loader: Training data loader
        epochs: Total epochs (for joint training)
        lr: Base learning rate
        out_path: Output directory
        checkpoint_freq: Save checkpoint every N epochs
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    trainer = FlexibleFlowTrainer(model, lr=lr)

    # Setup logging
    log_file = out_path / f"training_log_flow.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "flow_loss",
                "flow_lr",
                "batches_processed",
            ]
        )

    # Training state
    best_loss = float("inf")
    patience_counter = 0

    print(f"\nStarting training of flow model")
    print(f"Learning rate: {lr}")
    print(f"Output path: {out_path}")

    # Init flow_scale
    model.initialize_flow_scale(train_loader=train_loader)

    # Training loop
    for epoch in range(epochs):
        is_collapsing = trainer.check_for_collapse(
            train_loader=train_loader, epoch=epoch
        )
        if is_collapsing:
            print("Stopping due to collapse")
            break
        epoch_stats = trainer.train_epoch(train_loader)

        # Update scheduler
        trainer.flow_scheduler.step(epoch_stats["flow_loss"])

        # Get current learning rates
        flow_lr = trainer.flow_optimizer.param_groups[0]["lr"]

        # Log to CSV
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    epoch_stats["flow_loss"],
                    flow_lr,
                    epoch_stats["batches_processed"],
                ]
            )

        # Save best model
        if epoch_stats["flow_loss"] < best_loss:
            best_loss = epoch_stats["flow_loss"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "flow_lr": flow_lr,
                },
                out_path / f"best_model_flow.pt",
            )
        else:
            patience_counter += 1

        # Logging
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}")
            print(f"  Flow Loss: {epoch_stats['flow_loss']:.4f}")
            print(f"  LR - Flow: {flow_lr:.2e}")
            print(f"  Batches: {epoch_stats['batches_processed']} processed")

    print(f"\nTraining completed. Best loss: {best_loss:.6f}")

    # Load best model
    checkpoint = torch.load(out_path / f"best_model_flow.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, log_file
