import itertools
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from rich import pretty
pretty.install()
class SSLModel(nn.Module):
    def __init__(self, z_dim, dim):
        super().__init__()
        self.z_dim = z_dim

        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        pretty.pprint(f"feature_dim: {feature_dim}")
        pretty.pprint(f"z_dim: {z_dim}, dim: {dim}")

        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        pretty.pprint(f"dim: {dim}")
        # 4. Encoder: feature_dim -> dim -> (mu/logvar)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim//4),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(dim//4, self.z_dim)
        self.fc_logvar = nn.Linear(dim//4, self.z_dim)

        # 5. Decoder: z_dim -> dim -> feature_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, dim//4),
            nn.ReLU(),
            nn.Linear(dim//4, dim),
            nn.ReLU(),
            nn.Linear(dim, feature_dim)
        )
        pretty.pprint(f"self.z_dim: {self.z_dim}")

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        feats = self.backbone(x)
        if feats.ndim == 4:
            feats = self.pool_layer(feats)

        h   = self.encoder(feats)
        mu  = self.fc_mu(h)
        lv  = self.fc_logvar(h)

        if self.training:
            z = self.reparameterize(mu, lv)
        else:
            z = mu

        recon = self.decoder(z)
        return feats, mu, lv, z, recon

    
