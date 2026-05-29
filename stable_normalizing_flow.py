# Original codebase from https://github.com/anky3733/Implementing_Research_Papers/tree/main/Normalizing_Flows before modifications
import numpy as np
import torch
import torch.nn as nn


def soft_clamp(s, alpha=3.0):
    """
    Soft clamping using σα(h) = (2α / π) * arctan(h / α)
    This smoothly maps (-∞, ∞) to (-α, α)
    Using this because of differnet

    Args:
        s: input tensor
        alpha: clamping bound (output will be in range (-alpha, alpha))
    """
    return (2 * alpha / np.pi) * torch.arctan(s / alpha)


class RealNVPCoupling(nn.Module):
    """
    Real NVP coupling layer
    Reference: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, dim, hidden_dim=64, mask_type="alternate"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = dim

        # Create mask
        mask = torch.ones(dim)
        if mask_type == "alternate":
            mask[::2] = 0  # Alternate mask pattern
        elif mask_type == "half":
            mask[: dim // 2] = 0
        self.register_buffer("mask", mask)

        # Neural networks for scale and translation
        self.s_network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.t_network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

        # Initialize the last layer with small weights for stability
        nn.init.normal_(self.s_network[-1].weight, 0, 0.0001)
        nn.init.normal_(self.t_network[-1].weight, 0, 0.001)
        nn.init.zeros_(self.s_network[-1].bias)
        nn.init.zeros_(self.t_network[-1].bias)

    def forward(self, z):
        masked_z = z * self.mask
        s = self.s_network(masked_z) * (1 - self.mask)  # Scale
        t = self.t_network(masked_z) * (1 - self.mask)  # Translation

        # Clamp scale for stability
        # s = soft_clamp(s, alpha=1.0)
        s = torch.tanh(s) * 0.5  # Range [-0.5, 0.5]

        # Scale and shift
        transformed = z * self.mask + (1 - self.mask) * (z * torch.exp(s) + t)
        log_det = torch.sum(s * (1 - self.mask), dim=1)  # Log determinant

        return transformed, log_det

    def inverse(self, z):
        """Inverse transformation for sampling"""
        masked_z = z * self.mask
        s = self.s_network(masked_z) * (1 - self.mask)
        t = self.t_network(masked_z) * (1 - self.mask)

        s = torch.clamp(s, min=-2, max=2)

        z_out = z * self.mask + (1 - self.mask) * ((z - t) * torch.exp(-s))
        log_det = -torch.sum(s * (1 - self.mask), dim=1)

        return z_out, log_det


class StableNormalizingFlow(nn.Module):
    """
    A sequence of normalizing flows with proper log determinant handling
    """

    def __init__(self, flows, dim):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.dim = dim  # Store dimension

        # Use learnable scaling
        self.input_scale = nn.Parameter(torch.ones(dim) * 0.5)
        self.input_shift = nn.Parameter(torch.zeros(dim))

        # Base distribution (standard normal)
        self.register_buffer("base_mean", torch.zeros(dim))
        self.register_buffer("base_cov", torch.eye(dim))

    def forward(self, x):
        """
        Forward transformation: x -> z
        Returns transformed z and log determinant of Jacobian
        """
        # Apply input scaling
        z = x * self.input_scale + self.input_shift

        # Initialize log_det properly
        batch_size = x.size(0)
        # Compute input scale contribution
        input_log_det = torch.sum(torch.log(torch.abs(self.input_scale) + 1e-8))
        log_det_sum = input_log_det.repeat(batch_size)

        for i, flow in enumerate(self.flows):
            if isinstance(flow, RealNVPCoupling):
                z, log_det = flow(z)
            else:
                log_det = flow.log_det_jacobian(z)
                z = flow(z)

            # Ensure proper shape
            if log_det.dim() > 1:
                log_det = log_det.sum(1)

            # Check for NaN/Inf
            # Changed NaNs to numbers! 200 or -200
            if torch.isnan(log_det).any() or torch.isinf(log_det).any():
                print(f"Warning: Input contains NaN/Inf values")
                # Replace NaN/Inf with zeros
                log_det = torch.nan_to_num(
                    log_det, nan=0.0, posinf=200.0, neginf=-200.0
                )
                # raise ValueError(f"Warning: NaN/Inf detected in flow {i}")
            if torch.isnan(z).any() or torch.isinf(z).any():
                print(f"Warning: Input contains NaN/Inf values")
                # Replace NaN/Inf with zeros
                z = torch.nan_to_num(z, nan=0.0, posinf=200.0, neginf=-200.0)

            log_det_sum += log_det

        return z, log_det_sum

    def inverse(self, z):
        """
        Inverse transformation: z -> x
        Used for sampling from the distribution
        """
        batch_size = z.size(0)
        log_det_sum = torch.zeros(batch_size, device=z.device)

        # Apply flows in reverse order
        for flow in reversed(self.flows):
            if hasattr(flow, "inverse"):
                z, log_det = flow.inverse(z)
                log_det_sum -= log_det
            else:
                # For flows without explicit inverse, use numerical inverse
                # This is approximate and should be avoided if possible
                print(f"Warning: {type(flow).__name__} doesn't have inverse method")

        # Inverse input scaling
        x = (z - self.input_shift) / self.input_scale
        input_log_det = torch.sum(torch.log(torch.abs(self.input_scale)))
        log_det_sum = log_det_sum - input_log_det.repeat(batch_size)

        return x, log_det_sum

    def log_prob(self, x):
        """
        Compute log probability of x under the learned distribution.

        p(x) = p(z) * |det(dz/dx)|
        log p(x) = log p(z) + log|det(dz/dx)|

        For anomaly detection:
        - Higher log_prob = more likely (normal)
        - Lower log_prob = less likely (anomalous)
        """
        # Transform x to z
        z, log_det_jacobian = self.forward(x)

        # Base distribution log prob (standard normal)
        # log p(z) = -0.5 * ||z||^2 - 0.5 * d * log(2π)
        # Add small epsilon for numerical stability
        log_prob_z = -0.5 * (z**2).sum(dim=1) - 0.5 * self.dim * np.log(2 * np.pi)

        # # Clamp for stability
        # log_prob_z = torch.clamp(log_prob_z, min=-500, max=10)
        # log_det_jacobian = torch.clamp(log_det_jacobian, min=-100, max=100)

        # Only sanitize NaNs/Infs before summing:
        log_prob_z = torch.nan_to_num(log_prob_z, nan=-1e6, posinf=-1e6, neginf=1e6)
        log_det_jacobian = torch.nan_to_num(
            log_det_jacobian, nan=0.0, posinf=1e6, neginf=-1e6
        )

        # Apply change of variables formula
        # log p(x) = log p(z) + log|det(dz/dx)|
        log_prob_x = log_prob_z + log_det_jacobian

        # # Final clamping
        # log_prob_x = torch.clamp(log_prob_x, min=-800, max=100)

        return log_prob_x


def get_flow_architectures(dim, hidden_dim, specific=None):
    """
    Returns a dict with possible flow architectures.
    If specific is set, returns only the requested architecture.
    Possible values for specific: 'RealNVP'
    """
    flow_architectures = {
        "RealNVP": [
            RealNVPCoupling(dim=dim, hidden_dim=hidden_dim),
            RealNVPCoupling(dim=dim, hidden_dim=hidden_dim, mask_type="alternate"),
            RealNVPCoupling(dim=dim, hidden_dim=hidden_dim, mask_type="half"),
        ],
    }
    if specific is not None:
        return flow_architectures[specific]
    return flow_architectures
