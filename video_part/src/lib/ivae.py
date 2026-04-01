import torch
import torch.nn as nn
import torch.nn.functional as F

from base_vae import ActivationAnalyzer, FrameDecoder, FrameEncoder


LATENT_DIM = 8 * 16 * 16


def build_auxiliary_variables(
    batch_size: int,
    time_steps: int,
    noise_dim: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Build time-only auxiliary variables, optionally concatenated with noise."""
    time_index = torch.linspace(0.0, 1.0, time_steps, device=device, dtype=dtype)
    time_index = time_index.view(1, time_steps, 1).expand(batch_size, -1, -1)

    if noise_dim <= 0:
        return time_index

    noise = torch.randn(batch_size, time_steps, noise_dim, device=device, dtype=dtype)
    noise = noise * noise_scale
    return torch.cat([time_index, noise], dim=-1)


class ConditionalPrior(nn.Module):
    def __init__(self, aux_dim: int, latent_dim: int = LATENT_DIM, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(u)
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), min=-8.0, max=8.0)
        return mu, logvar


class LatentFrameIVAE(nn.Module):
    """iVAE variant built on the same shared encoder/decoder backbone."""

    def __init__(
        self,
        channel: int = 4,
        time: int = 16,
        aux_dim: int = 1,
        z_dim: int = 128,
        hidden_dim: int = 256,
        enable_analysis: bool = False,
    ):
        super().__init__()
        self.analyzer = ActivationAnalyzer() if enable_analysis else None
        self.channel = channel
        self.time = time
        self.aux_dim = aux_dim
        self.z_dim = z_dim
        self.latent_dim = LATENT_DIM

        self.encoder = FrameEncoder(in_channels=channel, z_dim=z_dim, analyzer=self.analyzer)
        self.decoder = FrameDecoder(out_channels=channel, z_dim=z_dim, analyzer=self.analyzer)

        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.posterior_mu_delta = nn.Linear(hidden_dim, self.latent_dim)
        self.posterior_logvar_delta = nn.Linear(hidden_dim, self.latent_dim)
        nn.init.zeros_(self.posterior_mu_delta.weight)
        nn.init.zeros_(self.posterior_mu_delta.bias)
        nn.init.zeros_(self.posterior_logvar_delta.weight)
        nn.init.zeros_(self.posterior_logvar_delta.bias)
        self.prior = ConditionalPrior(aux_dim=aux_dim, latent_dim=self.latent_dim, hidden_dim=hidden_dim)
        self.posterior_delta_scale = nn.Parameter(torch.tensor(0.1))
        self.prior_logvar_bias = nn.Parameter(torch.tensor(-4.0))
        self.posterior_logvar_bias = nn.Parameter(torch.tensor(-6.0))
        self.sample_posterior = False

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _encode_visual(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visual_mu = self.encoder.mu(x)
        visual_logvar = self.encoder.logvar(x)
        visual_mu = self.encoder.to_1d(visual_mu)
        visual_logvar = self.encoder.to_1d(visual_logvar)
        return visual_mu, visual_logvar

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        B, C, T, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous().reshape(B * T, C, H, W)
        u_flat = u.reshape(B * T, -1)

        visual_mu, visual_logvar = self._encode_visual(x_flat)
        aux_feat = self.aux_encoder(u_flat)

        encoder_mu = visual_mu + self.posterior_delta_scale * self.posterior_mu_delta(aux_feat)
        encoder_logvar = torch.clamp(
            visual_logvar + self.posterior_logvar_delta(aux_feat) + self.posterior_logvar_bias,
            min=-8.0,
            max=8.0,
        )
        if self.sample_posterior:
            z = self.reparameterize(encoder_mu, encoder_logvar)
        else:
            z = encoder_mu

        recon_flat = self.decoder(z)
        recon_x = recon_flat.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        prior_mu, prior_logvar = self.prior(u_flat)
        prior_logvar = torch.clamp(prior_logvar + self.prior_logvar_bias, min=-8.0, max=8.0)

        encoder_params = (encoder_mu, torch.exp(encoder_logvar))
        prior_params = (prior_mu, torch.exp(prior_logvar))
        decoder_params = (recon_x, None)
        z = z.reshape(B, T, -1)
        return decoder_params, encoder_params, prior_params, z

    def elbo(self, x: torch.Tensor, u: torch.Tensor, beta: float = 1.0):
        decoder_params, encoder_params, prior_params, z = self.forward(x, u)
        recon_x, _ = decoder_params
        encoder_mu, encoder_var = encoder_params
        prior_mu, prior_var = prior_params

        recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        encoder_logvar = torch.log(torch.clamp(encoder_var, min=1e-8))
        prior_logvar = torch.log(torch.clamp(prior_var, min=1e-8))
        kl_div = 0.5 * torch.sum(
            prior_logvar
            - encoder_logvar
            + (encoder_var + (encoder_mu - prior_mu) ** 2) / torch.clamp(prior_var, min=1e-8)
            - 1.0,
            dim=1,
        ).mean()

        total_loss = recon_loss + beta * kl_div
        return total_loss, recon_loss, kl_div, z, recon_x

    def optimize_activations(self):
        if not self.analyzer:
            return

        self.analyzer.print_statistics()
        for _, module in self.named_modules():
            if hasattr(module, "layer_name") and hasattr(module, "update_slope") and module.layer_name:
                optimal_slope = self.analyzer.get_optimal_leaky_relu_slope(module.layer_name)
                module.update_slope(optimal_slope)
