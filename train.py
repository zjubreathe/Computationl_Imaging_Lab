import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from piqa import PSNR, SSIM
from tqdm import tqdm
import time
import os
import yaml
from typing import Dict, Any

import UNet.model as UNet
from dataset import ISBI_Loader
from config import load_config, print_config


def train_net(config: Dict[str, Any]):

    print_config(config)

    device = torch.device('cuda' if config['device']['use_cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model_config = config['model']
    net = UNet.UNet(
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        features=model_config['features']
    )
    net = net.to(device)

    data_config = config['data']
    train_dataset = ISBI_Loader(data_config['train_path'])
    val_dataset = ISBI_Loader(data_config['val_path'])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        pin_memory=data_config['pin_memory'],
        drop_last=True,
        num_workers=data_config['num_workers']
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=data_config['pin_memory'],
        drop_last=False,
        num_workers=2
    )

    optimizer_config = config['optimizer']
    optimizer = optim.Adam(
        net.parameters(),
        lr=config['training']['learning_rate'],
        betas=(float(optimizer_config['betas'][0]),
               float(optimizer_config['betas'][1])),
        eps=float(optimizer_config['eps']),
        weight_decay=config['training']['weight_decay'],
        amsgrad=optimizer_config['amsgrad']
    )

    mse_criterion = nn.MSELoss(reduction='mean')
    psnr_fn = PSNR().to(device)
    ssim_fn = SSIM(n_channels=1).to(device)

    use_amp = config['training']['use_amp']
    scaler = GradScaler(enabled=use_amp)

    model_save_config = config['model_save']
    model_dir = os.path.dirname(model_save_config['path'])
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    best_psnr = -float('inf')
    early_stopping_counter = 0
    early_stopping_patience = config['training']['early_stopping_patience']

    print(f"开始训练，共 {config['training']['epochs']} 个epoch")
    print("-" * 80)

    for epoch in range(config['training']['epochs']):
        net.train()
        losses = []
        train_pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}/{config['training']['epochs']}", leave=False)

        for image, label in train_pbar:
            image = image.to(device=device, dtype=torch.float32, non_blocking=True)
            label = label.to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                pred = net(image)
                mse_loss = mse_criterion(pred, label)
                loss = mse_loss

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=config['training']['gradient_clip'])

            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = sum(losses) / len(losses)

        net.eval()
        psnr_sum = 0
        ssim_sum = 0
        total = 0

        val_pbar = tqdm(val_loader, desc=f"验证 Epoch {epoch + 1}/{config['training']['epochs']}", leave=False)
        with torch.no_grad():
            for image, label in val_pbar:
                image = image.to(device=device, dtype=torch.float32, non_blocking=True)
                label = label.to(device=device, dtype=torch.float32, non_blocking=True)

                pred = net(image)

                pred = torch.clamp(pred, 0., 1.)
                label = torch.clamp(label, 0., 1.)

                psnr_sum += psnr_fn(pred, label).item()
                ssim_sum += ssim_fn(pred, label).item()
                total += 1

        psnr_ave = psnr_sum / total
        ssim_ave = ssim_sum / total
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1:3d}/{config['training']['epochs']}, "
              f"LR {current_lr:.2e}, "
              f"Loss {avg_loss:.4f}, "
              f"PSNR {psnr_ave:.4f}, "
              f"SSIM {ssim_ave:.4f}")

        if psnr_ave > best_psnr:
            best_psnr = psnr_ave
            torch.save(net.state_dict(), model_save_config['path'])
            print(f"🎉 保存最佳模型，PSNR: {best_psnr:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f"⏹️ 在epoch {epoch + 1}提前停止训练")
            break

        time.sleep(1)

    print(f"\n训练完成！最佳PSNR: {best_psnr:.4f}")


if __name__ == "__main__":
    config = load_config("config.yaml")
    train_net(config)