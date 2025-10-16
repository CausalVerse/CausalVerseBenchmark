#!/usr/bin/env python3
import argparse
import os
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
from image_dataset import UnsupervisedImageDataset
from basic_model import SSLModel,ResNetContrastiveModel
from tqdm import tqdm
from metrics.correlation import compute_mcc
from torch.func import jacfwd, vmap
from c_dataset import PairedUnsupervisedImageDataset
from rich import pretty 
from metrics.block import compute_r2
pretty.install()
import time
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and test SSLModel predicting Z via MSE loss"
    )
    parser.add_argument("--meta_csv", type=str, default="", help="Path to metadata CSV file")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=int(128/4), help="Batch size for training and testing")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for MLP")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/Test split ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model")
    parser.add_argument('--z_dim', type=int, default=3, metavar='N')
    #add vae_lambda and sparsity_lambda
    parser.add_argument('--vae_lambda', type=float, default=1e-1, metavar='N')
    parser.add_argument('--sparsity_lambda', type=float, default=1e-1, metavar='N')
    return parser.parse_args()


def train_epoch(model, loader, optimizer, device, args):
    """
    One epoch of contrastive training, now also computing global MCC & R²
    between the learned representations and ground-truth z.
    Displays per-batch loss in the tqdm bar.
    """
    model.train()
    running_loss = 0.0

    # for global metric computation
    all_z_pred = []
    all_z_true = []

    sim = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()
    total_r2 = 0
    total_step = 0
    total_mcc = 0
    train_bar = tqdm(loader, desc=" Train", leave=True)
    for sample_ids, pair_imgs, _, pair_zs in train_bar:
        (x1, x2), (x3, x4) = pair_imgs
        (z1_true, z2_true), (z3_true, z4_true) = pair_zs

        # move to device
        x1, x2, x3, x4 = [x.to(device) for x in (x1, x2, x3, x4)]
        z_batch_true = torch.cat([z1_true, z2_true, z3_true, z4_true], dim=0).to(device)

        # forward through encoder + projection head
        z1 = model(x1)   # → (N, D_proj)
        z2 = model(x2)
        z3 = model(x3)
        z4 = model(x4)

        # accumulate for global metrics
        z_pred_batch = torch.cat([z1, z2, z3, z4], dim=0)
        # pretty.pprint(f"z_pred_batch: {z_pred_batch.shape}")
        z_pred_batch = z_pred_batch[:,:5]
        # pretty.pprint(f"z_pred_batch: {z_pred_batch.shape}")
        all_z_pred.append(z_pred_batch.detach().cpu())
        all_z_true.append(z_batch_true.cpu())
        total_step +=1
        
        def contrastive_pair_loss(a, b):
            sim11 = sim(a.unsqueeze(-2), a.unsqueeze(-3)) / args.tau
            sim22 = sim(b.unsqueeze(-2), b.unsqueeze(-3)) / args.tau
            sim12 = sim(a.unsqueeze(-2), b.unsqueeze(-3)) / args.tau

            N = sim12.size(-1)
            diag = torch.arange(N, device=device)
            sim11[..., diag, diag] = float('-inf')
            sim22[..., diag, diag] = float('-inf')

            raw1 = torch.cat([sim12,                sim11], dim=-1)      # (N,2N)
            raw2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)  # (N,2N)
            logits = torch.cat([raw1, raw2], dim=0)                     # (2N,2N)
            targets = torch.arange(2 * N, device=device, dtype=torch.long)
            return criterion(logits, targets)

        # compute and backprop loss
        loss1 = contrastive_pair_loss(z1, z2)
        loss2 = contrastive_pair_loss(z3, z4)
        loss  = 0.5 * (loss1 + loss2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_postfix({"loss": f"{loss.item()}"})

    avg_loss = running_loss /total_step

    # compute global MCC & R² on all training samples
    all_z_pred = torch.cat(all_z_pred, dim=0)  # (N_total, z_dim)
    all_z_true = torch.cat(all_z_true, dim=0)  # (N_total, z_dim)
    z_pred_np  = all_z_pred.permute(1, 0).numpy()  # (z_dim, N_total)
    z_true_np  = all_z_true.permute(1, 0).numpy()  # (z_dim, N_total)

    train_mcc = compute_mcc(z_pred_np, z_true_np, "Pearson")
    # train_mcc2 = compute_mcc(z_true_np,z_pred_np, "Pearson")
    # print(f"train_mcc: {train_mcc:.4f}, train_mcc2: {train_mcc2:.4f}")
    z_pred_r2 = z_pred_np.T   # = all_z_pred.numpy()
    z_true_r2 = z_true_np.T   # = all_z_true.numpy()
    # pretty.pprint(z_pred_r2)
    # pretty.pprint(z_true_r2)
    # 计算 R²
    z_pred_r2 = torch.tensor(z_pred_r2)
    z_true_r2 = torch.tensor(z_true_r2)
    train_r2 = compute_r2(z_true_r2, z_pred_r2)
    
    pretty.pprint(f"Train loss: {avg_loss} ┃ MCC: {train_mcc:.4f} ┃ R²: {train_r2:.4f}| average R²: {total_r2/total_step:.4f} ┃ average MCC: {total_mcc/total_step:.4f}")
    return avg_loss, train_mcc, train_r2


def test_epoch(model, loader, device):
    """
    Evaluate Pearson MCC and R² between model representations and true content z
    by concatenating all predictions and targets, then computing metrics once.
    """
    model.eval()
    all_z_pred = []
    all_z_true = []
    total_r2 = 0
    total_step = 0
    total_mcc = 0
    test_bar = tqdm(loader, desc="  Test", leave=False)
    with torch.no_grad():
        for sample_ids, pair_imgs, _, pair_zs in test_bar:
            (x1, x2), (x3, x4) = pair_imgs
            (z1, z2), (z3, z4) = pair_zs


            x_batch = torch.cat([x1, x2, x3, x4], dim=0).to(device)  # (4N, C, H, W)
            z_batch = torch.cat([z1, z2, z3, z4], dim=0).to(device)  # (4N, z_dim)

            z_pred = model(x_batch)  # (4N, D_proj)
            z_pred = z_pred[:,:5]

            all_z_pred.append(z_pred.detach().cpu())
            all_z_true.append(z_batch.cpu())
            total_step += 1
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

    print(f"Global MCC: {mcc:.4f}, Global R²: {r2:.4f}")
    return mcc, r2, total_mcc/total_step, total_r2/total_step



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretty.pprint(f"Using device: {device}")
    # prepare log file
    data_name = os.path.splitext(os.path.basename(args.meta_csv))[0]
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_dir = f"./splog/{time_str}"
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
    full_ds = PairedUnsupervisedImageDataset(args.meta_csv, transform=transform)
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
    model     = ResNetContrastiveModel(content_n=args.z_dim, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # add
    # training loop
    best_test_mcc = 0
    best_test_r2 = -10000000
    best_test_avg_mcc = 0
    best_test_avg_r2 = -10000000
    best_test_mcc_epoch = 0
    best_test_r2_epoch = 0
    best_test_avg_mcc_epoch = 0
    best_test_avg_r2_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mcc, train_r2 = train_epoch(
            model, train_loader, optimizer, device, args
        )
        test_mcc, test_r2,test_avg_mcc,test_avg_r2 = test_epoch(model, test_loader, device)

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
        if test_mcc > best_test_mcc:
            best_test_mcc = test_mcc
            best_test_mcc_epoch = epoch
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_test_r2_epoch = epoch
        if test_avg_mcc > best_test_avg_mcc:
            best_test_avg_mcc = test_avg_mcc
            best_test_avg_mcc_epoch = epoch
        if test_avg_r2 > best_test_avg_r2:
            best_test_avg_r2 = test_avg_r2
            best_test_avg_r2_epoch = epoch
    #report max values of mcc and r2, test_avg_mcc and test_avg_r2
    print(f"Best Test MCC: {best_test_mcc:.4f} at epoch {best_test_mcc_epoch}")
    print(f"Best Test R²: {best_test_r2:.4f} at epoch {best_test_r2_epoch}")
    print(f"Best Test avg MCC: {best_test_avg_mcc:.4f} at epoch {best_test_avg_mcc_epoch}")
    print(f"Best Test avg R²: {best_test_avg_r2:.4f} at epoch {best_test_avg_r2_epoch}")
    
    
    
    # save model
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "ssl_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

if __name__ == "__main__":
    main()






