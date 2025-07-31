import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
from model_dit import SimpleDiTConditional
# from noise_scheduler import NoiseScheduler
from PIL import Image

# Define condition vectors for digits 0-9
conditions_mappings = {
    str(i): torch.nn.functional.one_hot(torch.tensor(i), num_classes=10).float().unsqueeze(0)
    for i in range(10)
}

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using a trained diffusion model with Classifier-Free Guidance')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output', type=str, default='fm_output_ode.jpg', help='Path to save the image')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='CFG guidance scale (default: 3.0)')
    parser.add_argument('--label', type=str, default='0', help='Label to generate (0-9, default: 0)')
    parser.add_argument('--img_size', type=int, default=32, help='Image size (default: 32)')
    parser.add_argument('--channels', type=int, default=1, help='Number of image channels (default: 1)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--fm_steps', type=int, default=300, help='steps for flow matching')
    return parser.parse_args()

def infer_flow_matching_cfg(model, x_0, condition, guidance_scale=3.0, fm_steps=50):

    model.eval()
    device = x_0.device
    x = x_0.clone()

    B = x.shape[0]
    dt = 1.0 / fm_steps  # 每一步的时间间隔
    uncond_condition = torch.zeros_like(condition).to(device)

    with torch.no_grad():
        for i in range(fm_steps):
            t = torch.full((B,), i / fm_steps, dtype=torch.float, device=device)  # 当前时间 t ∈ [0,1)

            # 预测有条件与无条件速度场
            v_uncond = model(x, t, uncond_condition, force_drop_condition=True)
            v_cond = model(x, t, condition)

            # v = v_uncond + guidance_scale * (v_cond - v_uncond)
            v = v_uncond
            x = x + v * dt

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
        print(f"Warning: Label '{args.label}' not found in mappings, using default label '0'")
        condition = conditions_mappings['0'].to(device)

    model = SimpleDiTConditional(
        img_size=(args.img_size, args.img_size),
        image_channels=args.channels,
        patch_size=4,
        condition_dim=10
    ).to(device)
    
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    x_t = torch.randn((1, args.channels, args.img_size, args.img_size), device=device)

    with torch.no_grad():
        output = infer_flow_matching_cfg(model, x_t, condition, args.guidance_scale, fm_steps=args.fm_steps)

    img_array = output.cpu().detach().numpy()
    img = img_array[0, 0] 
    
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Save the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Generated image for label {args.label} saved to {args.output}")


if __name__ == "__main__":
    main()