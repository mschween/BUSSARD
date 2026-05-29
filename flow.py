import numpy as np
import torch
import torch.nn as nn

from stable_normalizing_flow import StableNormalizingFlow, get_flow_architectures


class UnsupervisedFlow(nn.Module):
    def __init__(
        self,
        hidden_dim,
        flow_name,
        flow_input_dim=64,
        use_running_norm=True,
    ):
        super().__init__()
        self.use_running_norm = use_running_norm

        self.register_buffer("running_mean", torch.zeros(flow_input_dim))
        self.register_buffer("running_var", torch.ones(flow_input_dim))
        self.register_buffer("num_batches_tracked", torch.tensor(0))

        # Normalizing Flow for density estimation
        flows = get_flow_architectures(
            dim=flow_input_dim, hidden_dim=hidden_dim, specific=flow_name
        )
        self.flow = StableNormalizingFlow(flows=flows, dim=flow_input_dim)

        # Learnable combination weight for anomaly scoring
        self.combination_weight = nn.Parameter(torch.tensor(0.5))

        # Add buffers to store training statistics
        self.register_buffer("flow_mean", torch.tensor(0.0))
        self.register_buffer("flow_std", torch.tensor(1.0))
        self.register_buffer("stats_computed", torch.tensor(False))

    # Call before training the model
    def initialize_flow_scale(self, train_loader):
        """Initialize flow scale based on actual flow inputs during training"""
        device = next(self.parameters()).device
        self.eval()

        all_flow_inputs = []

        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= 10:  # Sample from first 10 batches
                    break

                batch = batch.to(device)
                flow_input = batch.x

                all_flow_inputs.append(flow_input)

        # Combine all samples
        all_flow_inputs = torch.cat(all_flow_inputs, dim=0)

        # Set scale to normalize to unit variance
        flow_mean = all_flow_inputs.mean(dim=0)
        flow_std = all_flow_inputs.std(dim=0) + 1e-6

        self.flow.input_scale.data = 1.0 / flow_std
        self.flow.input_shift.data = -flow_mean / flow_std

        print(f"Flow initialized with:")
        print(f"  Input mean: {flow_mean.mean():.4f}")
        print(f"  Input std: {flow_std.mean():.4f}")
        print(
            f"  Scale range: [{self.flow.input_scale.min():.4f}, {self.flow.input_scale.max():.4f}]"
        )

        self.train()

    def compute_and_store_training_stats(self, train_loader):
        """
        Compute normalization statistics from training data
        Should be called after training, before evaluation
        """
        self.eval()
        device = next(self.parameters()).device

        flow_scores = []

        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                outputs = self.forward(batch, return_all=True)

                # Raw flow scores (negative log prob)
                batch_flow_scores = -outputs["flow_log_probs"]
                flow_scores.extend(batch_flow_scores.cpu().numpy())

        # Compute statistics
        flow_scores = np.array(flow_scores)
        self.flow_mean = torch.tensor(flow_scores.mean(), device=device)
        self.flow_std = torch.tensor(flow_scores.std(), device=device)

        self.stats_computed = torch.tensor(True, device=device)

        print(f"Training statistics computed:")
        print(f"  Flow:  mean={self.flow_mean:.4f}, std={self.flow_std:.4f}")

    def forward(self, data, return_all=False):
        """
        Forward pass for training (returns log prob and flow input)
        """
        flow_input = data.x.to(self.running_var.device)

        # Update running statistics during training
        if self.training and self.use_running_norm:
            with torch.no_grad():
                self.num_batches_tracked += 1
                batch_mean = flow_input.mean(dim=0)
                batch_var = flow_input.var(dim=0, unbiased=False)

                # Exponential moving average
                momentum = 0.1
                self.running_mean = (
                    1 - momentum
                ) * self.running_mean + momentum * batch_mean
                self.running_var = (
                    1 - momentum
                ) * self.running_var + momentum * batch_var

        if self.use_running_norm:
            # Normalize using running statistics
            flow_input = (flow_input - self.running_mean) / (
                torch.sqrt(self.running_var + 1e-5)
            )

        # Get flow log probabilities
        edge_log_probs = self.flow.log_prob(flow_input)

        if return_all:
            return {
                "flow_log_probs": edge_log_probs,
                "flow_input": flow_input,
            }

        return edge_log_probs

    def compute_anomaly_scores(self, data):
        """
        Compute anomaly scores using training statistics for normalization (used during testing).

        NaN/Infs are counted as anomaly.
        """
        self.eval()
        with torch.no_grad():
            # Get outputs
            outputs = self.forward(data, return_all=True)

            flow_scores = -outputs["flow_log_probs"]

            # Handle NaN/Inf as extreme anomalies
            nan_mask = torch.isnan(flow_scores) | torch.isinf(flow_scores)
            if nan_mask.any():
                print(
                    f"Detected {nan_mask.sum()}/{len(flow_scores)} extreme anomalies (NaN/Inf)"
                )
                # Set extreme anomalies to very high score
                flow_scores[nan_mask] = flow_scores[~nan_mask].max() + 10

            flow_scores_normalized = flow_scores

            return {
                "flow_scores": flow_scores_normalized,
                "raw_flow_scores": flow_scores,
                "num_extreme_anomalies": nan_mask.sum().item() if nan_mask.any() else 0,
            }
