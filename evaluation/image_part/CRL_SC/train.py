#!/usr/bin/env python3
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
from image_dataset import UnsupervisedImageDataset, BalancedBatchSampler
from basic_model import SSLModel,iVAE
from tqdm import tqdm
from metrics.correlation import compute_mcc
from metrics.block import compute_r2
from torch.func import jacfwd, vmap
from sklearn import metrics
from rich import pretty
import time
pretty.install()
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and test SSLModel predicting Z via VAE+sparsity with MCC/R² eval"
    )
    parser.add_argument('--root', type=str, default='../da_datasets/domainnet',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='DomainNet')
    parser.add_argument('-s', '--source', help='source domain(s)', default='i,p,q,r,s')
    parser.add_argument('-t', '--target', help='target domain(s)', default='c')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',)
    parser.add_argument('--bottleneck-dim', default=2048, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    # parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
    #                     metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    # parser.add_argument('--epochs', default=40, type=int, metavar='N',
    #                     help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=2500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('-e', '--eval-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='logs',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    # parser.add_argument('--z_dim', type=int, default=10, metavar='N')
    parser.add_argument('--train_batch_size', default=16, type=int)
    
    # parser.add_argument('--hidden_dim', type=int, default=512, metavar='N')
    parser.add_argument('--beta', type=float, default=1., metavar='N')
    parser.add_argument('--name', type=str, default='', metavar='N')
    parser.add_argument('--flow', type=str, default='ddsf', metavar='N')
    parser.add_argument('--flow_dim', type=int, default=16, metavar='N')
    parser.add_argument('--flow_nlayer', type=int, default=2, metavar='N')
    parser.add_argument('--init_value', type=float, default=0.0, metavar='N')
    parser.add_argument('--flow_bound', type=int, default=5, metavar='N')
    parser.add_argument('--flow_bins', type=int, default=8, metavar='N')
    parser.add_argument('--flow_order', type=str, default='linear', metavar='N')
    parser.add_argument('--net', type=str, default='dirt', metavar='N')
    parser.add_argument('--n_flow', type=int, default=2, metavar='N')
    # parser.add_argument('--lambda_vae', type=float, default=1e-3, metavar='N')
    parser.add_argument('--lambda_cls', type=float, default=1., metavar='N')
    parser.add_argument('--lambda_ent', type=float, default=0.1, metavar='N')
    parser.add_argument('--entropy_thr', type=float, default=0.5, metavar='N')
    parser.add_argument('--C_max', type=float, default=20., metavar='N')
    parser.add_argument('--C_stop_iter', type=int, default=10000, metavar='N')
    parser.add_argument('--num_classes', type=float, default=8, metavar='N')
    
    
    parser.add_argument("--meta_csv", type=str,
                        default="/Spring.csv",
                        help="Path to metadata CSV file")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for MLP")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/Test split ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model")
    parser.add_argument('--s_dim', type=int, default=10, metavar='N')
    parser.add_argument("--z_dim", type=int, default=15, metavar="N", help="Latent dimension")
    parser.add_argument("--vae_lambda", type=float, default=1e-1, metavar="N", help="Weight for VAE loss")
    parser.add_argument("--sparsity_lambda", type=float, default=1e-1, metavar="N", help="Weight for sparsity loss")
    parser.add_argument("--lambda_gauss", type=float, default=1e-1, metavar="N", help="Weight for Gaussian loss")
    parser.add_argument("--lambda_vae", type=float, default=1e-1, metavar="N", help="Weight for VAE loss")
    return parser.parse_args()

def train_epoch(model, loader, optimizer, device, args):
    model.train()
    running_loss = 0.0
    all_z_pred = []
    all_z_true = []
    model:iVAE
    model.train()
    model.to(device)
    normal_distribution = torch.distributions.MultivariateNormal(torch.zeros(args.z_dim).cuda(), torch.eye(args.z_dim).cuda())
    prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(args.s_dim, device=device),
            covariance_matrix=torch.eye(args.s_dim, device=device)
        )
    train_bar = tqdm(loader, desc="  Train  ", leave=True)
    total_step = 0
    for _id, images, views, z_true in train_bar:
        images, z_true = images.to(device), z_true.to(device)
        views = views.to(device)
        # print(f"images.shape: {images.shape}, z_target.shape: {z_true.shape}, _view: {views}")
        # forward pass
        losses_gauss = []
        losses_kl = []
        z_all = []
        x_all = []
        for i in range(4):
            domain_id = i
            index = views == domain_id
            domain_x = images[index]
            view_all = views[index]
            domain_z = z_true[index]
            # print(f"domain x,min: {domain_x.min()}, max: {domain_x.max()}, mean: {domain_x.mean()}")
            x_dom = model.backbone(domain_x, False)
            # print(f"min: {x_dom.min()}, max: {x_dom.max()}, mean: {x_dom.mean()}")
            z, tilde_z, mu, log_var, logdet_u, logit = model.encode(x_dom, u=view_all, track_bn=False)
            tilde_zs = tilde_z[:,args.c_dim:]
            # pretty.pprint(f"max: {tilde_z.max()}, min: {tilde_z.min()}, mean: {tilde_z.mean()}")

            loss_gauss =- prior.log_prob(tilde_zs).mean()
            losses_gauss.append(loss_gauss)
            
            # print(f"tilde_zs.shape: {tilde_zs.shape}")
            # print(f"loss_gauss: {loss_gauss}")
            q_dist = torch.distributions.Normal(mu, torch.exp(torch.clamp(log_var, min=-10) / 2))
            log_qz = q_dist.log_prob(z)
            log_pz = normal_distribution.log_prob(tilde_z) + logdet_u
            kl = (log_qz.sum(dim=1) - log_pz).mean()
            C = torch.clamp(torch.tensor(args.C_max) / args.C_stop_iter * 10000, 0, args.C_max)
            loss_kl = args.beta * (kl - C).abs()
            losses_kl.append(loss_kl)
            x_all.append(x_dom)
            z_all.append(z)
        x_all = torch.cat(x_all, 0)
        z_all = torch.cat(z_all, 0)
        x_all_hat = model.decode(z_all)
        mean_loss_recon = F.mse_loss(x_all, x_all_hat, reduction='sum') / len(x_all)
        mean_loss_kl = torch.stack(losses_kl, dim=0).mean()
        mean_loss_vae = mean_loss_recon + mean_loss_kl
        mean_loss_gauss = torch.stack(losses_gauss, dim=0).mean()
        total_loss = args.lambda_vae * mean_loss_vae + args.lambda_gauss * mean_loss_gauss
        total_loss = total_loss.clamp(0, 1000)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
        total_step += 1

        # collect for global metrics
        all_z_pred.append(z_all.detach().cpu())
        all_z_true.append(z_true.cpu())

        train_bar.set_postfix({
            "recon":    f"{mean_loss_recon.item():.4f}",
            "KL":       f"{mean_loss_kl.item():.4f}",
            "mean_loss_gauss": f"{mean_loss_gauss.item():.4f}",
            "total":    f"{total_loss.item():.4f}"
        })
        # break
    
    avg_loss = running_loss / len(loader.dataset)

    # compute global MCC & R² on all training samples
    all_z_pred = torch.cat(all_z_pred, dim=0)  # (N_total, z_dim)
    all_z_true = torch.cat(all_z_true, dim=0)  # (N_total, z_dim)
    #only used first args_c_dim
    all_z_pred = all_z_pred[:, :args.c_dim]
    all_z_true = all_z_true[:, :args.c_dim]
    
    
    z_pred_np  = all_z_pred.permute(1, 0).numpy()  # (z_dim, N_total)
    z_true_np  = all_z_true.permute(1, 0).numpy()  # (z_dim, N_total)
 
    train_mcc = compute_mcc(z_pred_np, z_true_np, "Pearson")
    print(f"train_mcc: {train_mcc}, z_pred_np.shape: {z_pred_np.shape}, z_true_np.shape: {z_true_np.shape}")
    
    z_pred_r2 = z_pred_np.T   # = all_z_pred.numpy()
    z_true_r2 = z_true_np.T   # = all_z_true.numpy()
    print(f"z_pred_r2.shape: {z_pred_r2.shape}, z_true_r2.shape: {z_true_r2.shape}")
    # 计算 R²
    # train_r2 = metrics.r2_score(z_true_r2, z_pred_r2, multioutput="uniform_average")
    z_pred_r2 = torch.tensor(z_pred_r2)
    z_true_r2 = torch.tensor(z_true_r2)
    train_r2 = compute_r2(z_true_r2, z_pred_r2)
    

    pretty.pprint(f"Train loss: {avg_loss:.4f} ┃ MCC: {train_mcc:.4f} ┃ R²: {train_r2:.4f} | average R²: {train_r2:.4f} ┃ average MCC: {train_mcc:.4f}")
    return avg_loss, train_mcc, train_r2

def test_epoch(model, loader, device,args):
    model.eval()
    all_z_pred = []
    all_z_true = []
    total_r2 = 0
    total_step = 0
    total_mcc = 0
    test_bar = tqdm(loader, desc="  Test   ", leave=False)
    with torch.no_grad():
        for _id, images, views, z_true in test_bar:
            images, z_true = images.to(device), z_true.to(device)
            views = views.to(device)
            # print(f"images.shape: {images.shape}, z_target.shape: {z_true.shape}, _view: {views}")
            # forward pass
            losses_gauss = []
            losses_kl = []
            z_all = []
            z_true_all = []
            for i in range(4):
                domain_id = i
                index = views == domain_id
                domain_x = images[index]
                view_all = views[index]
                domain_z = z_true[index]
                # print(f"domain x,min: {domain_x.min()}, max: {domain_x.max()}, mean: {domain_x.mean()}")
                x_dom = model.backbone(domain_x, False)
                # print(f"min: {x_dom.min()}, max: {x_dom.max()}, mean: {x_dom.mean()}")
                z, tilde_z, mu, log_var, logdet_u, logit = model.encode(x_dom, u=view_all, track_bn=False)
                z_all.append(tilde_z)
                z_true_all.append(domain_z)

            tilde_z = torch.cat(z_all, 0)
            z_true = torch.cat(z_true_all, 0)
            all_z_pred.append(tilde_z[:,:args.c_dim].cpu())
            all_z_true.append(z_true[:,:args.c_dim].cpu())
            # pretty.pprint(f"max: {tilde_z.max()}, min: {tilde_z.min()}, mean: {tilde_z.mean()}")
            z_pred = tilde_z[:,:args.c_dim]
            z_true = z_true[:,:args.c_dim]
            # mcc = compute_mcc(z_pred.cpu().numpy().T, z_true.cpu().numpy().T, "Pearson")
            # total_mcc += mcc
            total_step += 1
            # total_r2 += compute_r2()
    all_z_pred = torch.cat(all_z_pred, dim=0)
    all_z_true = torch.cat(all_z_true, dim=0)

    z_pred_np = all_z_pred.permute(1, 0).numpy()
    z_true_np = all_z_true.permute(1, 0).numpy()
    print(f"z_pred_np.shape: {z_pred_np.shape}, z_true_np.shape: {z_true_np.shape}")
    mcc = compute_mcc(z_pred_np, z_true_np, "Pearson")
    z_pred_np = z_pred_np.T   # = all_z_pred.numpy()
    z_true_np = z_true_np.T   # = all_z_true.numpy()
    print(f"z_pred_np.shape: {z_pred_np.shape}, z_true_np.shape: {z_true_np.shape}")
    z_pred_r2 = torch.tensor(z_pred_np)
    z_true_r2 = torch.tensor(z_true_np)
    r2 = compute_r2(z_true_r2, z_pred_r2)

    pretty.pprint(f"Test MCC: {mcc:.4f} ┃ Test R²: {r2:.4f} | average R²: {total_r2/total_step:.4f} ┃ average MCC: {total_mcc/total_step:.4f}")
    return mcc, r2, total_mcc/total_step, total_r2/total_step




def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.c_dim = args.z_dim - args.s_dim
    
    pretty.pprint(f" z_c dim is {args.c_dim}")
    # prepare log file
    data_name = os.path.splitext(os.path.basename(args.meta_csv))[0]
    pretty.pprint(f"data_name is {data_name}")
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_dir = f"./splog/{time_str}/"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{data_name}_{args.vae_lambda}_{args.sparsity_lambda}.txt"
    log_path = os.path.join(log_dir, log_filename)
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_mcc,train_r2,test_mcc,test_r2,test_avg_mcc,test_avg_r2\n")

    # transforms & dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    full_ds = UnsupervisedImageDataset(args.meta_csv, transform=transform)
    print(f"Full dataset size: {len(full_ds)}")


    train_size = int(len(full_ds) * args.train_ratio)
    test_size  = len(full_ds) - train_size
    train_ds, test_ds = random_split(full_ds, [train_size, test_size])

    train_sampler = BalancedBatchSampler(train_ds, batch_size=args.batch_size)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,      
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_sampler = BalancedBatchSampler(test_ds, batch_size=args.batch_size)

    test_loader = DataLoader(
        test_ds,
        batch_sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )


    for ids, imgs, views, zs in train_loader:
        print(views.bincount())  # 应当输出 tensor([bs/4, bs/4, bs/4, bs/4])
        break

    # model & optimizer
    # model     = SSLModel(args.z_dim, args.hidden_dim).to(device)
    model =iVAE(args)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mcc, train_r2 = train_epoch(
            model, train_loader, optimizer, device, args
        )
        test_mcc, test_r2,test_avg_mcc,test_avg_r2 = test_epoch(model, test_loader, device,args)

        # epoch summary
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train MCC: {train_mcc:.4f} | Train R²: {train_r2:.4f} | "
            f"Test MCC: {test_mcc:.4f} | Test R²: {test_r2:.4f}"
            f"average MCC: {test_avg_mcc:.4f} | average R²: {test_avg_r2:.4f}"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.4f},{train_mcc:.4f},{train_r2:.4f},"
                f"{test_mcc:.4f},{test_r2:.4f},{test_avg_mcc:.4f},{test_avg_r2:.4f}\n"
            )
    #report max values of mcc and r2, test_avg_mcc and test_avg_r2
    print(f"Max MCC: {test_mcc:.4f} at epoch {epoch} | ")
    print(f"Max R²: {test_r2:.4f} at epoch {epoch} | ")
    print(f"Max average MCC: {test_avg_mcc:.4f} at epoch {epoch} | ")
    print(f"Max average R²: {test_avg_r2:.4f} at epoch {epoch} | ")

    # save model
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "ssl_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

if __name__ == "__main__":
    main()






