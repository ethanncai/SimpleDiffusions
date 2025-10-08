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
from model_dit_sequential import SimpleDiTConditional

def parse_args():
    parser = argparse.ArgumentParser(description='Train a diffusion model on MNIST dataset')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=32, help='Image size (default: 32)')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size for ViT (default: 4)')
    parser.add_argument('--save_dir', type=str, default='output', help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_freq', type=int, default=5, help='Save model every N epochs')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='LR decay step size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='LR decay rate')
    return parser.parse_args()

def setup_data_loader(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalization for single-channel images
    ])
    
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    return train_loader

def train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader = setup_data_loader(args)
    print(f"Training samples: {len(train_loader.dataset)}")
    
    model = SimpleDiTConditional(
        img_size=(args.img_size, args.img_size),
        image_channels=1,  # MNIST is single channel
        patch_size=args.patch_size,
        label_condition_dim=10,  # 10 classes for MNIST digits
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    save_path = os.path.join(args.save_dir, date_str)
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                          desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (images, labels) in progress_bar:
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
            images = images.to(device)
            # B C H W
            optimizer.zero_grad()
            
            noise = torch.randn_like(images, device=device)
            t = torch.rand(size=(images.shape[0],), device=device)
            t_broadcast = t.view(-1, 1, 1, 1)
            x_t = images * t_broadcast + noise * (1 - t_broadcast)

            target = images - noise  # d(x_t)/dt
            pred_vector = model(x_t, t, labels)
            loss = F.l1_loss(pred_vector, target)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        
        avg_train_loss = total_loss / len(train_loader)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Average Loss: {avg_train_loss:.6f}")
        
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            model_filename = f"fm_dit_xatt_epoch_{epoch+1}.pt"
            checkpoint_path = os.path.join(save_path, model_filename)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Checkpoint saved to {checkpoint_path}")
    
    return model

if __name__ == "__main__":
    args = parse_args()
    trained_model = train(args)