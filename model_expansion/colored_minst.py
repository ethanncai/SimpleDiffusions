import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import random
from torchvision.utils import save_image
import os

class ColoredMNIST(Dataset):
    def __init__(self, mnist_dataset):
        """
        Wrap MNIST to add random RGB coloring to the DIGIT only (background stays black).
        
        Input mnist_dataset should output images in [0, 1] (before [-1,1] normalization).
        But to keep your existing pipeline, we assume it's already in [-1, 1].
        So we convert back to [0,1], color, then back to [-1,1].
        """
        self.mnist = mnist_dataset

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        gray_img, label = self.mnist[idx]  # shape [1, H, W], values in [-1, 1]
        
        # Step 1: Convert back to [0, 1] (original ToTensor range)
        gray_01 = (gray_img + 1) / 2  # [-1,1] -> [0,1]
        # Now: background=0, digit=1
        
        # Step 2: Generate random color in [0, 1] for RGB
        color_rgb = torch.empty(3).uniform_(0, 1)  # [3], values in [0, 1]
        color_label = color_rgb * 2 - 1  # also save in [-1, 1] as requested
        
        # Step 3: Apply color ONLY to foreground (background stays 0)
        # gray_01: [1, H, W] -> [H, W]
        # color_rgb: [3] -> [3, 1, 1]
        colored_01 = gray_01.squeeze(0).unsqueeze(0) * color_rgb.view(3, 1, 1)  # [3, H, W], [0,1]
        
        # Step 4: Convert back to [-1, 1] to match your normalization
        colored_img = colored_01 * 2 - 1  # [0,1] -> [-1,1]
        
        return colored_img, label, color_label  # color_label in [-1, 1] as requested
def setup_data_loader(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5], [0.5])  # -> [-1, 1] for single channel
    ])
    
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    colored_dataset = ColoredMNIST(mnist_dataset)
    
    train_loader = DataLoader(
        colored_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=getattr(args, 'num_workers', 0)
    )
    
    return train_loader

# --- 生成并保存示例 ---
def save_colored_mnist_examples(num_examples=4, img_size=32, save_dir="colored_mnist_examples"):
    os.makedirs(save_dir, exist_ok=True)
    
    # 原始 MNIST transform（归一化到 [-1, 1]）
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [0,1] -> [-1,1]
    ])
    
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    colored_dataset = ColoredMNIST(mnist_dataset)
    
    print(f"Saving {num_examples} colored MNIST examples to '{save_dir}/'...")
    
    for i in range(num_examples):
        colored_img, label, color_label = colored_dataset[i]
        
        # 转回 [0, 1] 用于保存图像（因为 save_image 期望 [0,1]）
        img_to_save = (colored_img + 1) / 2  # [-1,1] -> [0,1]
        
        # 保存图像
        save_path = os.path.join(save_dir, f"example_{i}_digit_{label}.png")
        save_image(img_to_save, save_path)
        
        # 打印信息
        print(f"Saved: {save_path}")
        print(f"  Label: {label}")
        print(f"  Color (R,G,B) in [-1,1]: {color_label.tolist()}")
        print("-" * 50)

if __name__ == "__main__":
    save_colored_mnist_examples(num_examples=6, img_size=32)