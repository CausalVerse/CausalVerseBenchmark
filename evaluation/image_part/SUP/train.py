#!/usr/bin/env python3
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
from image_dataset import UnsupervisedImageDataset
from basic_model import SupervisedModel
from tqdm import tqdm
from metrics.correlation import compute_mcc
from torch.func import jacfwd, vmap
from sklearn import metrics
from rich import pretty
from metrics.block import compute_r2
pretty.install()
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and test SSLModel predicting Z via VAE+sparsity with MCC/R² eval"
    )
    parser.add_argument("--meta_csv", type=str,
                        default="",
                        help="Path to metadata CSV file")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for MLP")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/Test split ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model")
    parser.add_argument("--z_dim", type=int, default=3, metavar="N", help="Latent dimension")
    parser.add_argument("--vae_lambda", type=float, default=1e-1, metavar="N", help="Weight for VAE loss")
    parser.add_argument("--sparsity_lambda", type=float, default=1e-1, metavar="N", help="Weight for sparsity loss")
    return parser.parse_args()

def train_epoch(model, loader, optimizer, device, args):
    model.train()
    running_loss = 0.0
    all_z_pred = []
    all_z_true = []
    train_bar = tqdm(loader, desc="  Train  ", leave=True)
    total_r2 = 0
    total_step = 0
    total_mcc = 0
    for _id, images, _view, z_target in train_bar:
        images, z_target = images.to(device), z_target.to(device)
        z= model(images)
        
        loss = F.mse_loss(z, z_target, reduction="mean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 
        
        total_step += 1
        all_z_pred.append(z.detach().cpu())
        all_z_true.append(z_target.cpu())

        train_bar.set_postfix({
            "total":    f"{loss.item()}"
        })

        

    # average training loss
    avg_loss = running_loss /total_step

    # compute global MCC & R² on all training samples
    all_z_pred = torch.cat(all_z_pred, dim=0)  # (N_total, z_dim)
    all_z_true = torch.cat(all_z_true, dim=0)  # (N_total, z_dim)
    z_pred_np  = all_z_pred.permute(1, 0).numpy()  # (z_dim, N_total)
    z_true_np  = all_z_true.permute(1, 0).numpy()  # (z_dim, N_total)

    train_mcc = compute_mcc(z_pred_np, z_true_np, "Pearson")

    z_pred_r2 = z_pred_np.T   # = all_z_pred.numpy()
    z_true_r2 = z_true_np.T   # = all_z_true.numpy()

    z_pred_r2 = torch.tensor(z_pred_r2)
    z_true_r2 = torch.tensor(z_true_r2)
    train_r2 = compute_r2(z_true_r2, z_pred_r2)
    
    pretty.pprint(f"Train loss: {avg_loss:.4f} ┃ MCC: {train_mcc:.4f} ┃ R²: {train_r2:.4f}| average R²: {total_r2/total_step:.4f} ┃ average MCC: {total_mcc/total_step:.4f}")
    return avg_loss, train_mcc, train_r2

def test_epoch(model, loader, device):
    model.eval()
    all_z_pred = []
    all_z_true = []
    total_r2 = 0
    total_step = 0
    total_mcc = 0
    test_bar = tqdm(loader, desc="  Test   ", leave=False)
    with torch.no_grad():
        for _id, images, _view, z in test_bar:
            images, z = images.to(device), z.to(device)
            z_pred= model(images)
            mcc = compute_mcc(z_pred.cpu().numpy().T, z.cpu().numpy().T, "Pearson")
            total_mcc += mcc
            total_step += 1
            total_r2 += metrics.r2_score(z.cpu().numpy(), z_pred.cpu().numpy(), multioutput="uniform_average")
            
            all_z_pred.append(z_pred.cpu())
            all_z_true.append(z.cpu())

    all_z_pred = torch.cat(all_z_pred, dim=0)  # (N_total, z_dim)
    all_z_true = torch.cat(all_z_true, dim=0)  # (N_total, z_dim)
    z_pred_np  = all_z_pred.permute(1, 0).numpy()  # (z_dim, N_total)
    z_true_np  = all_z_true.permute(1, 0).numpy()  # (z_dim, N_total)

    mcc = compute_mcc(z_pred_np, z_true_np, "Pearson")
    z_pred_r2 = z_pred_np.T   # = all_z_pred.numpy()
    z_true_r2 = z_true_np.T   # = all_z_true.numpy()
    z_pred_r2 = torch.tensor(z_pred_r2)
    z_true_r2 = torch.tensor(z_true_r2)
    r2 = compute_r2(z_true_r2, z_pred_r2)
    pretty.pprint(f"Test MCC: {mcc:.4f} ┃ Test R²: {r2:.4f} | average R²: {total_r2/total_step:.4f} ┃ average MCC: {total_mcc/total_step:.4f}")
    return mcc, r2

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare log file
    data_name = os.path.splitext(os.path.basename(args.meta_csv))[0]
    log_dir = "./splog"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{data_name}_{args.vae_lambda}_{args.sparsity_lambda}.txt"
    log_path = os.path.join(log_dir, log_filename)
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_mcc,train_r2,test_mcc,test_r2\n")

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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # model & optimizer
    model     = SupervisedModel(args.z_dim, args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # add
    # training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mcc, train_r2 = train_epoch(
            model, train_loader, optimizer, device, args
        )
        test_mcc, test_r2 = test_epoch(model, test_loader, device)

        # epoch summary
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train MCC: {train_mcc:.4f} | Train R²: {train_r2:.4f} | "
            f"Test MCC: {test_mcc:.4f} | Test R²: {test_r2:.4f}"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.4f},{train_mcc:.4f},{train_r2:.4f},"
                f"{test_mcc:.4f},{test_r2:.4f}\n"
            )

    # save model
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "ssl_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

if __name__ == "__main__":
    main()






