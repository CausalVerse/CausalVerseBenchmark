import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_vae import ActivationAnalyzer, FrameDecoder, FrameEncoder


LATENT_DIM = 8 * 16 * 16


class TCLLatentFrameVAE(nn.Module):
    """BaseVAE7 backbone with an auxiliary TCL-style segment classifier on latents."""

    def __init__(
        self,
        channel: int = 16,
        time: int = 16,
        z_dim: int = 128,
        enable_analysis: bool = False,
        segment_size: int = 1,
    ):
        super().__init__()
        self.analyzer = ActivationAnalyzer() if enable_analysis else None
        self.encoder = FrameEncoder(in_channels=channel, z_dim=z_dim, analyzer=self.analyzer)
        self.decoder = FrameDecoder(out_channels=channel, z_dim=z_dim, analyzer=self.analyzer)
        self.channel = channel
        self.time = time
        self.z_dim = z_dim
        self.latent_dim = LATENT_DIM
        self.segment_size = max(1, segment_size)
        self.num_segments = max(1, math.ceil(time / self.segment_size))
        self.segment_classifier = nn.Linear(self.latent_dim, self.num_segments)

    def _build_segment_labels(self, batch_size: int, time_steps: int, device: torch.device) -> torch.Tensor:
        frame_ids = torch.arange(time_steps, device=device)
        labels = torch.div(frame_ids, self.segment_size, rounding_mode="floor")
        labels = torch.clamp(labels, max=self.num_segments - 1)
        return labels.unsqueeze(0).expand(batch_size, -1)

    def forward(self, x: torch.Tensor):
        batch_size, channels, time_steps, height, width = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * time_steps, channels, height, width)

        z_flat, mu, logvar = self.encoder(x_flat)
        recon_flat = self.decoder(z_flat)
        recon = recon_flat.view(batch_size, time_steps, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        z = z_flat.view(batch_size, time_steps, -1)

        logits = self.segment_classifier(z.reshape(batch_size * time_steps, -1))
        labels = self._build_segment_labels(batch_size, time_steps, z.device).reshape(-1)
        return recon, mu, logvar, z, logits, labels

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        beta: float = 0.0,
        tcl_weight: float = 0.1,
    ):
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (x.size(0) * x.size(2) * x.size(3) * x.size(4))
        tcl_loss = F.cross_entropy(logits, labels)
        total_loss = recon_loss + beta * kl_loss + tcl_weight * tcl_loss
        return total_loss, recon_loss, kl_loss, tcl_loss

    def optimize_activations(self):
        if not self.analyzer:
            return

        self.analyzer.print_statistics()
        for _, module in self.named_modules():
            if hasattr(module, "layer_name") and hasattr(module, "update_slope") and module.layer_name:
                optimal_slope = self.analyzer.get_optimal_leaky_relu_slope(module.layer_name)
                module.update_slope(optimal_slope)
