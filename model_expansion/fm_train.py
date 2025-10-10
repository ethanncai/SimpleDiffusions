# train.py
from email import parser
import torch
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from colored_minst import ColoredMNIST
from model_extendable_dit import ExtendableDiT
from train_utils import generate_comprehensive_visualization  # 导入可视化函数
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train a diffusion model on MNIST dataset')
    parser.add_argument('--ckpt', type=str, default='', help='CKPT')
    parser.add_argument('--epochs', type=int, default=128, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=32, help='Image size (default: 32)')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size for ViT (default: 4)')
    parser.add_argument('--save_dir', type=str, default='output', help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_freq', type=int, default=8, help='Save model every N epochs')
    parser.add_argument('--lr_decay_step', type=int, default=64, help='LR decay step size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='LR decay rate')
    parser.add_argument('--model_variation', type=str, choices=['base','sequential','parallel'], default='base', help='model variation: base or extend')
    parser.add_argument('--label_condition_dim', type=int, default=10, help='Label condition dimension (default: 10)')
    parser.add_argument('--color_condition_dim', type=int, default=3, help='Color condition dimension (default: 3), only valid when model variation is not base')
    parser.add_argument('--channels', type=int, default=3, help='Number of image channels (default: 3)')
    parser.add_argument('--visualize_freq', type=int, default=500, help='Visualize and save image every N steps (0 to disable)')

    return parser.parse_args()

def setup_data_loader(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) 
    ])
    
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    colored_dataset = ColoredMNIST(dataset)
    train_loader = DataLoader(colored_dataset, batch_size=args.batch_size, shuffle=True)
    
    return train_loader

def train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader = setup_data_loader(args)
    print(f"Training samples: {len(train_loader.dataset)}")
    
    # === Model ===
    model = ExtendableDiT(
        img_size=(args.img_size, args.img_size),
        image_channels=args.channels,
        patch_size=args.patch_size,
        label_condition_dim=args.label_condition_dim,
        color_label_condition_dim=args.color_condition_dim,
        model_variant=args.model_variation # base, sequential, parallel
    ).to(device)

    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}")
        model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"{args.model_variation}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    date_str = datetime.now().strftime("%Y-%m-%d")
    save_path = os.path.join(args.save_dir, date_str)
    os.makedirs(save_path, exist_ok=True)
    
    # === 用于可视化的固定种子 ===
    fixed_seed = 42

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                          desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (images, labels, labels_color) in progress_bar:
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
            labels_color = labels_color.to(device)
            images = images.to(device)
            optimizer.zero_grad()
            
            noise = torch.randn_like(images, device=device)
            t = torch.rand(size=(images.shape[0],), device=device)
            t_broadcast = t.view(-1, 1, 1, 1)
            x_t = images * t_broadcast + noise * (1 - t_broadcast)
            target = images - noise

            if args.model_variation == 'base':
                pred_vector = model(x_t, t, (labels, None))
            else:
                pred_vector = model(x_t, t, (labels, labels_color))

            loss = F.l1_loss(pred_vector, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            writer.add_scalar('Loss/batch', loss.item(), global_step)
            
            # ========== 调用外部可视化函数 ==========
            if args.visualize_freq > 0 and global_step % args.visualize_freq == 0:
                # 更新随机种子以获得不同的噪声
                fixed_seed += 1
                
                # 调用分离的可视化函数
                generate_comprehensive_visualization(
                    model, args, device, global_step, save_path, timestamp, writer, fixed_seed
                )
                
                # 恢复训练时的随机性
                torch.manual_seed(args.seed + global_step)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed + global_step)
            # ==========================================

            global_step += 1
            progress_bar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        
        avg_train_loss = total_loss / len(train_loader)
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs} | Average Loss: {avg_train_loss:.6f}")
        
        writer.add_scalar('Loss/epoch', avg_train_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            model_filename = f"base_{epoch+1}.pt" if args.model_variation == 'base' else f"ext_sequential_{epoch+1}.pt"
            checkpoint_path = os.path.join(save_path, model_filename)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Checkpoint saved to {checkpoint_path}")
    
    writer.close()
    print(f"Training finished. TensorBoard logs saved to: {log_dir}")
    return model

if __name__ == "__main__":
    args = parse_args()
    trained_model = train(args)