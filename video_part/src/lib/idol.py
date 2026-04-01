import sys
import types

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))
from lib.components.transition import NPInstantaneousTransitionPrior
from base_vae import LatentFrameVAE


class InstantaneousProcess(nn.Module):
    def __init__(
        self,
        z_dim,
        lag,
        hidden_dim=128,
        beta=0.0025,
        gamma=0.0075,
        theta=0.02,
        w_hist=None,
        w_inst=None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.lag = lag
        self.beta = beta
        self.gamma = gamma
        self.theta = theta

        self.transition_prior = NPInstantaneousTransitionPrior(
            lags=lag,
            latent_size=z_dim,
            num_layers=3,
            hidden_dim=hidden_dim,
        )

        self.w_hist = w_hist if w_hist is not None else [1.0] * z_dim
        self.w_inst = w_inst if w_inst is not None else [1.2] * z_dim

        self.register_buffer("base_dist_mean", torch.zeros(self.z_dim))
        self.register_buffer("base_dist_var", torch.eye(self.z_dim))

        self.trans_show = None
        self.inst_show = None

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def loss_function(self, mus, logvars, zs):
        batch_size, length, _ = zs.shape
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)

        init_kl = -0.5 * (
            1 + logvars[:, : self.lag] - mus[:, : self.lag].pow(2) - logvars[:, : self.lag].exp()
        )
        kld_normal = init_kl.sum(dim=-1).sum(dim=-1).mean()

        if length <= self.lag:
            zero = zs.new_zeros(())
            self.trans_show = torch.zeros(self.z_dim, self.lag * self.z_dim)
            self.inst_show = torch.zeros(self.z_dim, self.z_dim)
            return zero, kld_normal, zero

        residuals, logabsdet, hist_jac = self.transition_prior(zs)
        log_pz_future = self.base_dist.log_prob(residuals).sum(dim=1) + logabsdet
        log_qz_future = log_qz[:, self.lag :].sum(dim=-1).sum(dim=-1)
        kld_future = ((log_qz_future - log_pz_future) / (length - self.lag)).mean()

        sparsity_loss = zs.new_zeros(())
        numm = 0
        trans_show = []
        inst_show = []

        for idx, jac in enumerate(hist_jac):
            hist_part = jac[:, 0, : self.lag * self.z_dim]
            inst_part = jac[:, 0, self.lag * self.z_dim :]

            sparsity_loss = sparsity_loss + self.w_hist[idx] * hist_part.abs().sum()
            sparsity_loss = sparsity_loss + self.w_inst[idx] * inst_part.abs().sum()
            numm += jac.numel()

            trans_show.append(hist_part.detach().cpu())
            inst_show.append(
                F.pad(
                    inst_part.detach().cpu(),
                    (0, self.z_dim - inst_part.shape[1], 0, 0),
                    mode="constant",
                    value=0,
                )
            )

        sparsity_loss = sparsity_loss / max(numm, 1)
        self.trans_show = torch.stack(trans_show, dim=1).abs().mean(dim=0)
        self.inst_show = torch.stack(inst_show, dim=1).abs().mean(dim=0)

        return sparsity_loss, kld_normal, kld_future


class IDOLLatentVAE(nn.Module):
    def __init__(
        self,
        channel=4,
        time=16,
        latent_dim=2048,
        z_dim=8,
        lag=2,
        hidden_dim=128,
        beta=0.00025,
        gamma=0.0075,
        theta=0.02,
        recon_weight=1.0,
        enable_analysis=False,
    ):
        super().__init__()
        self.channel = channel
        self.time = time
        self.latent_dim = latent_dim
        self.z_dim = z_dim
        self.recon_weight = recon_weight

        self.latent_vae = LatentFrameVAE(
            channel=channel,
            time=time,
            z_dim=latent_dim,
            enable_analysis=enable_analysis,
        )

        self.causal_mu_head = nn.LazyLinear(z_dim)
        self.causal_logvar_head = nn.LazyLinear(z_dim)

        self.idol_process = InstantaneousProcess(
            z_dim=z_dim,
            lag=lag,
            hidden_dim=hidden_dim,
            beta=beta,
            gamma=gamma,
            theta=theta,
        )

    def encode_causal_latent(self, z_full, random_sampling=True):
        flat_z = z_full.reshape(-1, z_full.shape[-1])
        causal_mu = self.causal_mu_head(flat_z)
        causal_logvar = self.causal_logvar_head(flat_z).clamp(min=-8.0, max=8.0)

        if random_sampling:
            eps = torch.randn_like(causal_logvar)
            causal_z = causal_mu + eps * torch.exp(0.5 * causal_logvar)
        else:
            causal_z = causal_mu

        batch_size, length, _ = z_full.shape
        causal_mu = causal_mu.view(batch_size, length, self.z_dim)
        causal_logvar = causal_logvar.view(batch_size, length, self.z_dim)
        causal_z = causal_z.view(batch_size, length, self.z_dim)
        return causal_mu, causal_logvar, causal_z

    def forward(self, x, random_sampling=True):
        x_recon, _, _, z_full = self.latent_vae(x)
        causal_mu, causal_logvar, causal_z = self.encode_causal_latent(
            z_full,
            random_sampling=random_sampling,
        )
        return x, x_recon, causal_mu, causal_logvar, causal_z, z_full, None

    def compute_loss(self, x, random_sampling=True):
        _, x_recon, causal_mu, causal_logvar, causal_z, z_full, _ = self.forward(
            x,
            random_sampling=random_sampling,
        )

        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        sparsity_loss, kld_normal, kld_future = self.idol_process.loss_function(
            causal_mu,
            causal_logvar,
            causal_z,
        )

        loss = (
            self.recon_weight * recon_loss
            + self.idol_process.beta * kld_normal
            + self.idol_process.gamma * kld_future
            + self.idol_process.theta * sparsity_loss
        )

        outputs = {
            "x_recon": x_recon,
            "z_full": z_full,
            "causal_mu": causal_mu,
            "causal_logvar": causal_logvar,
            "causal_z": causal_z,
        }
        losses = {
            "loss": loss,
            "recon_loss": recon_loss,
            "sparsity_loss": sparsity_loss,
            "kld_normal": kld_normal,
            "kld_future": kld_future,
        }
        return outputs, losses
