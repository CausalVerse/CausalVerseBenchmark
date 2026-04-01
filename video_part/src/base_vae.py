import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class ActivationAnalyzer:
    def __init__(self):
        self.activation_stats = defaultdict(list)
        self.negative_ratios = defaultdict(list)
        
    def analyze_activation(self, x, layer_name):
        with torch.no_grad():
            negative_count = (x < 0).sum().float()
            total_count = x.numel()
            negative_ratio = negative_count / total_count
            
            stats = {
                'negative_ratio': negative_ratio.item(),
                'mean': x.mean().item(),
                'std': x.std().item(),
                'min': x.min().item(),
                'max': x.max().item(),
                'median': x.median().item(),
                'q25': x.quantile(0.25).item(),
                'q75': x.quantile(0.75).item()
            }
            
            self.activation_stats[layer_name].append(stats)
            self.negative_ratios[layer_name].append(negative_ratio.item())
            
            return stats
    
    def get_optimal_leaky_relu_slope(self, layer_name, target_negative_ratio=0.1):

        if layer_name not in self.negative_ratios:
            return 0.01  
        
        avg_negative_ratio = np.mean(self.negative_ratios[layer_name])
        
        if avg_negative_ratio > 0.5:
            slope = 0.2  
        elif avg_negative_ratio > 0.3:
            slope = 0.1  
        elif avg_negative_ratio > 0.1:
            slope = 0.05  
        else:
            slope = 0.01 
            
        return slope
    
    def print_statistics(self):
        for layer_name, stats_list in self.activation_stats.items():
            if stats_list:
                avg_stats = {k: np.mean([s[k] for s in stats_list]) for k in stats_list[0].keys()}
                print(f"\n{layer_name}:")
                print(f"  Negative ratio: {avg_stats['negative_ratio']:.4f}")
                print(f"  Mean: {avg_stats['mean']:.4f}")
                print(f"  Std: {avg_stats['std']:.4f}")
                print(f"  Min: {avg_stats['min']:.4f}")
                print(f"  Max: {avg_stats['max']:.4f}")
                print(f"  Suggested LeakyReLU slope: {self.get_optimal_leaky_relu_slope(layer_name):.4f}")

class AdaptiveLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01, layer_name=None, analyzer=None):
        super().__init__()
        self.negative_slope = negative_slope
        self.layer_name = layer_name
        self.analyzer = analyzer
        
    def forward(self, x):
        if self.analyzer and self.layer_name:
            self.analyzer.analyze_activation(x, self.layer_name)
        
        return F.leaky_relu(x, negative_slope=self.negative_slope, inplace=False)
    
    def update_slope(self, new_slope):
        self.negative_slope = new_slope

class BiasCompensatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 layer_name=None, analyzer=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.layer_name = layer_name
        self.analyzer = analyzer
        
        self.bias_compensation = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        x = self.conv(x)
        
        if self.analyzer and self.layer_name:
            stats = self.analyzer.analyze_activation(x, f"{self.layer_name}_pre_bias")
            
            if stats['negative_ratio'] > 0.5:
                compensation = torch.abs(torch.tensor(stats['min'])) * 0.1
                self.bias_compensation.data.fill_(compensation)
        
        x = x + self.bias_compensation.view(1, -1, 1, 1)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, analyzer=None, block_name="residual"):
        super().__init__()
        self.conv1 = BiasCompensatedConv(channels, channels, 3, 1, 1, 
                                       f"{block_name}_conv1", analyzer)
        self.conv2 = BiasCompensatedConv(channels, channels, 3, 1, 1, 
                                       f"{block_name}_conv2", analyzer)
        self.relu1 = AdaptiveLeakyReLU(0.01, f"{block_name}_relu1", analyzer)
        self.relu2 = AdaptiveLeakyReLU(0.01, f"{block_name}_relu2", analyzer)
        
    def forward(self, x):
        residual = x
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu2(out)

class FrameEncoder(nn.Module):
    def __init__(self, in_channels=16, z_dim=128, analyzer=None):
        super().__init__()
        self.analyzer = analyzer
        
       
        self.mu = nn.Sequential(
            BiasCompensatedConv(in_channels, 4, 4, 2, 1, "encoder_conv1", analyzer),
            BiasCompensatedConv(4, 8, 4, 2, 1, "encoder_conv2", analyzer),
            AdaptiveLeakyReLU(0.01, "encoder_relu2", analyzer)
        )
        
        self.logvar = nn.Sequential(
            BiasCompensatedConv(in_channels, 4, 4, 2, 1, "encoder_conv12", analyzer),
            BiasCompensatedConv(4, 8, 4, 2, 1, "encoder_conv22", analyzer),
            AdaptiveLeakyReLU(0.01, "encoder_relu22", analyzer)
        )
        
        self.to_1d = nn.Sequential(
            nn.Flatten(),
        )
    
    
    def reparameterize(self, mean, logvar, random_sampling=False):
        if random_sampling:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            return mean + eps * std
        return mean
    
    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        z_1d = self.to_1d(z)
        
        return z_1d, mu, logvar  
        

class FrameDecoder(nn.Module):
    def __init__(self, out_channels=16, z_dim=128, analyzer=None):
        super().__init__()
        self.analyzer = analyzer
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 4, 2, 1),
            AdaptiveLeakyReLU(0.01, "decoder_relu2", analyzer)
        )
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(4, out_channels, 4, 2, 1),
        )
        
        
        self.from_1d = nn.Sequential(
            nn.Unflatten(1, (8, 16, 16))
        )
    
    def forward(self, z, skip_connections=None):
        x = self.from_1d(z)
        
        x = self.deconv2(x)
        
        x = self.deconv1(x)
        
        return x

class LatentFrameVAE(nn.Module):
    def __init__(self, channel=16, time=4, z_dim=128, enable_analysis=False):
        super().__init__()
        
        self.analyzer = ActivationAnalyzer() if enable_analysis else None
        
        self.encoder = FrameEncoder(in_channels=channel, z_dim=z_dim, analyzer=self.analyzer)
        self.decoder = FrameDecoder(out_channels=channel, z_dim=z_dim, analyzer=self.analyzer)
        self.channel = channel
        self.time = time
        self.z_dim = z_dim
        
        self.use_variational = False
    
    def forward(self, x, use_mean=False):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)
        
        z, mu, logvar = self.encoder(x)

       
        recon = self.decoder(z)

        recon = recon.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        z = z.view(B, T, -1)
        
        return recon, mu, logvar, z
    
    
    def optimize_activations(self):
        if not self.analyzer:
            return
            
        self.analyzer.print_statistics()
        
        for name, module in self.named_modules():
            if isinstance(module, AdaptiveLeakyReLU) and module.layer_name:
                optimal_slope = self.analyzer.get_optimal_leaky_relu_slope(module.layer_name)
                module.update_slope(optimal_slope)
                print(f"Updated slope for {module.layer_name}: {optimal_slope:.4f}")

def analyze_and_optimize_network():
    model = LatentFrameVAE(channel=16, time=4, z_dim=128, enable_analysis=True)
    
    x = torch.randn(2, 16, 4, 64, 64)
    
    print("Starting activation analysis...")
    
    for i in range(5):
        with torch.no_grad():
            recon, z = model(x + torch.randn_like(x) * 0.1)  
    
    model.optimize_activations()
    
    print("\nPost-optimization check:")
    with torch.no_grad():
        recon, z = model(x)
        mse_loss = F.mse_loss(recon, x)
        print(f"Reconstruction loss: {mse_loss.item():.6f}")
    
    return model

def vae_loss(recon_x, x, mu, logvar, beta=0.1, loss_type='mse'):
    if loss_type == 'mse':
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    elif loss_type == 'l1':
        recon_loss = F.l1_loss(recon_x, x, reduction='mean')
    elif loss_type == 'smooth_l1':
        recon_loss = F.smooth_l1_loss(recon_x, x, reduction='mean')
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    kl_loss = 0
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, total_loss


    
