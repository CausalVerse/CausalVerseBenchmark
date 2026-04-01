import sys
import types

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))
from lib.components.transition import NPTransitionPrior
from base_vae import LatentFrameVAE


class TemporalPrior(nn.Module):
    def __init__(self, z_dim, lag, hidden_dim=128, beta=0.0025, gamma=0.0075):
        super().__init__()
        self.z_dim = z_dim
        self.lag = lag
        self.beta = beta
        self.gamma = gamma

        self.transition_prior = NPTransitionPrior(
            lags=lag,
            latent_size=z_dim,
            num_layers=3,
            hidden_dim=hidden_dim,
        )

        self.register_buffer("base_dist_mean", torch.zeros(self.z_dim))
        self.register_buffer("base_dist_var", torch.eye(self.z_dim))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def loss_function(self, mus, logvars, zs, init_steps):
        batch_size, length, _ = zs.shape
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)

        init_kl = -0.5 * (
            1 + logvars[:, :init_steps] - mus[:, :init_steps].pow(2) - logvars[:, :init_steps].exp()
        )
        kld_normal = init_kl.sum(dim=-1).sum(dim=-1).mean()

        if length <= self.lag:
            zero = zs.new_zeros(())
            return kld_normal, zero

        residuals, logabsdet, _ = self.transition_prior(zs)
        log_pz_future = self.base_dist.log_prob(residuals).sum(dim=1) + logabsdet
        log_qz_future = log_qz[:, self.lag :].sum(dim=-1).sum(dim=-1)
        kld_future = ((log_qz_future - log_pz_future) / (length - self.lag)).mean()
        return kld_normal, kld_future


class CaRiNGModel(nn.Module):
    def __init__(
        self,
        channel=4,
        time=16,
        latent_dim=2048,
        z_dim=8,
        lag=2,
        context_frames=3,
        hidden_dim=128,
        beta=0.00025,
        gamma=0.0075,
        recon_weight=1.0,
        enable_analysis=False,
    ):
        super().__init__()
        self.channel = channel
        self.time = time
        self.latent_dim = latent_dim
        self.z_dim = z_dim
        self.lag = lag
        self.context_frames = context_frames
        self.recon_weight = recon_weight

        self.backbone = LatentFrameVAE(
            channel=channel,
            time=time,
            z_dim=latent_dim,
            enable_analysis=enable_analysis,
        )

        context_dim = latent_dim * context_frames
        self.causal_mu_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, z_dim),
        )
        self.causal_logvar_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, z_dim),
        )
        self.decode_head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.temporal_prior = TemporalPrior(
            z_dim=z_dim,
            lag=lag,
            hidden_dim=hidden_dim,
            beta=beta,
            gamma=gamma,
        )

    def _build_context_windows(self, frame_latents):
        batch_size, length, dim = frame_latents.shape
        pad_len = self.context_frames - 1
        if pad_len > 0:
            pad = frame_latents.new_zeros(batch_size, pad_len, dim)
            padded = torch.cat([pad, frame_latents], dim=1)
        else:
            padded = frame_latents

        windows = padded.unfold(dimension=1, size=self.context_frames, step=1)
        windows = windows.reshape(batch_size, length, self.context_frames * dim)
        return windows

    def encode_causal_latent(self, frame_latents, random_sampling=True):
        context = self._build_context_windows(frame_latents)
        flat_context = context.reshape(-1, context.shape[-1])

        causal_mu = self.causal_mu_head(flat_context)
        causal_logvar = self.causal_logvar_head(flat_context).clamp(min=-8.0, max=8.0)

        if random_sampling:
            eps = torch.randn_like(causal_logvar)
            causal_z = causal_mu + eps * torch.exp(0.5 * causal_logvar)
        else:
            causal_z = causal_mu

        batch_size, length, _ = frame_latents.shape
        causal_mu = causal_mu.view(batch_size, length, self.z_dim)
        causal_logvar = causal_logvar.view(batch_size, length, self.z_dim)
        causal_z = causal_z.view(batch_size, length, self.z_dim)
        return causal_mu, causal_logvar, causal_z

    def decode_observation(self, causal_z, batch_size, length, height, width):
        decoded_latents = self.decode_head(causal_z.reshape(-1, self.z_dim))
        recon = self.backbone.decoder(decoded_latents)
        recon = recon.view(batch_size, length, self.channel, height, width).permute(0, 2, 1, 3, 4).contiguous()
        return recon, decoded_latents.view(batch_size, length, self.latent_dim)

    def forward(self, x, random_sampling=True):
        batch_size, _, length, height, width = x.shape
        frame_first = x.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * length, self.channel, height, width)
        frame_latents, _, _ = self.backbone.encoder(frame_first)
        frame_latents = frame_latents.view(batch_size, length, self.latent_dim)

        causal_mu, causal_logvar, causal_z = self.encode_causal_latent(
            frame_latents,
            random_sampling=random_sampling,
        )
        x_recon, decoded_latents = self.decode_observation(causal_z, batch_size, length, height, width)

        init_steps = max(self.lag, self.context_frames - 1)
        kld_normal, kld_future = self.temporal_prior.loss_function(
            causal_mu,
            causal_logvar,
            causal_z,
            init_steps=init_steps,
        )
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = self.temporal_prior.beta * kld_normal + self.temporal_prior.gamma * kld_future
        total_loss = self.recon_weight * recon_loss + kl_loss

        losses = {
            "total": total_loss,
            "recon": recon_loss,
            "kl": kl_loss,
            "independence": kld_normal,
            "temporal": kld_future,
        }
        outputs = {
            "x_recon": x_recon,
            "causal_z": causal_z,
            "causal_mu": causal_mu,
            "causal_logvar": causal_logvar,
            "frame_latents": frame_latents,
            "decoded_latents": decoded_latents,
        }
        return outputs, losses
