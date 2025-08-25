import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
from model_dit import SimpleDiTConditional
from PIL import Image
import torchvision.utils as vutils
import math
# 条件向量（不变）
conditions_mappings = {
    str(i): torch.nn.functional.one_hot(torch.tensor(i), num_classes=10).float().unsqueeze(0)
    for i in range(10)
}

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using a trained diffusion model with Classifier-Free Guidance')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output', type=str, default='fm_output_sde.jpg', help='Path to save the image')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='CFG guidance scale (default: 3.0)')
    parser.add_argument('--label', type=str, default='0', help='Label to generate (0-9, default: 0)')
    parser.add_argument('--img_size', type=int, default=32, help='Image size (default: 32)')
    parser.add_argument('--channels', type=int, default=1, help='Number of image channels (default: 1)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--fm_steps', type=int, default=50, help='Steps for flow matching')
    parser.add_argument('--sde_noise_scale', type=float, default=1, help='SDE noise scale parameter a (default: 0.1)')
    parser.add_argument('--num_generated', type=int, default=1, help='Number of images to generate (default: 1)')
    return parser.parse_args()

def sde_step(x,t,dt,v,sde_noise_scale,device):
    t_int = t[0].item()
    sigma_t = sde_noise_scale * torch.sqrt((1-t) / (t + 1e-5))
    drift_correction = (sigma_t**2 / (2 * t_int + 1e-5)) * (x + (1 - t_int) * v) * dt
    noise = torch.randn_like(x) * sigma_t.view(-1, 1, 1, 1) * torch.sqrt(torch.tensor(dt).to(device))
    x = x + v * dt + drift_correction + noise
    return x

def infer_flow_matching_cfg(model, x_0, condition, guidance_scale=3.0, fm_steps=50, sde_noise_scale=0.01):
    model.eval()
    device = x_0.device
    x = x_0.clone()

    B = x.shape[0]
    dt = 1.0 / fm_steps
    uncond_condition = torch.zeros_like(condition).to(device)

    with torch.no_grad():
        for i in range(fm_steps):
            t_int = i / fm_steps
            t = torch.full((B,), t_int, dtype=torch.float, device=device)
            v_uncond = model(x, t, uncond_condition, force_drop_condition=True)
            v_cond = model(x, t, condition)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            
            # sde stepping
            x = sde_step(x,t,dt,v,sde_noise_scale,device)

    return x

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.label in conditions_mappings:
        condition = conditions_mappings[args.label].to(device)
    else:
        print(f"Warning: Label '{args.label}' not found, using default label '0'")
        condition = conditions_mappings['0'].to(device)

    model = SimpleDiTConditional(
        img_size=(args.img_size, args.img_size),
        image_channels=args.channels,
        patch_size=4,
        condition_dim=10
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    generated_images = []

    for i in range(args.num_generated):
        x_t = torch.randn((1, args.channels, args.img_size, args.img_size), device=device)
        with torch.no_grad():
            output = infer_flow_matching_cfg(
                model,
                x_t,
                condition,
                args.guidance_scale,
                fm_steps=args.fm_steps,
                sde_noise_scale=args.sde_noise_scale
            )
        output = output.clamp(-1, 1)  # 防止数值过大
        output = (output + 1) / 2  # scale to [0,1]
        generated_images.append(output)

    all_images = torch.cat(generated_images, dim=0)

    # 拼图并保存
    grid = vutils.make_grid(all_images, nrow=min(args.num_generated, 8), normalize=True, pad_value=1)
    np_img = grid.cpu().numpy().transpose((1, 2, 0))

    plt.figure(figsize=(args.num_generated, 1.5))
    plt.imshow(np_img.squeeze(), cmap='gray' if args.channels == 1 else None)
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Generated {args.num_generated} image(s) for label {args.label} saved to {args.output}")

if __name__ == "__main__":
    main()