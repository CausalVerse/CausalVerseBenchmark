import itertools
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import models
class SSLModel(nn.Module):
    def __init__(self, z_dim, dim):
        super().__init__()
        self.z_dim = z_dim
        print(f"z_dim: {z_dim}, dim: {dim}")

        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features

        backbone.fc = nn.Identity()
        self.backbone = backbone

 
        self.pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

        # 4. Encoder: feature_dim -> dim -> (mu/logvar)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, dim),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc_mu     = nn.Linear(dim, self.z_dim)
        self.fc_logvar = nn.Linear(dim, self.z_dim)

        # 5. Decoder: z_dim -> dim -> feature_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, feature_dim)
        )

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

    def get_parameters(self, base_lr=1.0):
        encoder_params = itertools.chain(
            self.encoder.parameters(),
            self.fc_mu.parameters(),
            self.fc_logvar.parameters(),
            self.decoder.parameters()
        )
        return [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr},
            {"params": encoder_params,         "lr": 1.0 * base_lr},
        ]




class ResNetContrastiveModel(nn.Module):
    """
    A simple contrastive‐learning model using ResNet‐18 as backbone.
    Given an input batch of images x of shape (B, 3, H, W),
    it outputs feature vectors of shape (B, content_n + 1).
    """
    def __init__(self, content_n: int = 8, pretrained: bool = True):
        super().__init__()
        # 1) Load a ResNet‐18 and drop its final fully‐connected layer
        print(f"content_n: {content_n}, pretrained: {pretrained}")
        backbone = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1],  # all layers except the last FC
                                     nn.Flatten())                     # flatten (B, 512, 1, 1) → (B, 512)

        # 2) Projection head: maps 512 → (content_n + 1)
        feat_dim = backbone.fc.in_features  # typically 512 for ResNet‐18
        dim = 64
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim//4),
            nn.ReLU(),
            nn.Linear(dim//4, content_n + 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) input images
        returns: (B, content_n + 1) projected features for contrastive loss
        """
        h = self.encoder(x)       # → (B, 512)
        z = self.projection(h)    # → (B, content_n + 1)
        return z