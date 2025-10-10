import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
from model_extendable_dit import ExtendableDiT
# from noise_scheduler import NoiseScheduler
from PIL import Image

# Define condition vectors for digits 0-9
conditions_mappings = {
    str(i): torch.nn.functional.one_hot(torch.tensor(i), num_classes=10).float().unsqueeze(0)
    for i in range(10)
}

def get_corresponding_color_tensor(color_char):
    color_char = color_char.lower()
    if color_char == 'r':
        return torch.tensor([1.0, -1.0, -1.0]).view(1, 3)  # Red
    elif color_char == 'g':
        return torch.tensor([-1.0, 1.0, -1.0]).view(1, 3)  # Green
    elif color_char == 'b':
        return torch.tensor([-1.0, -1.0, 1.0]).view(1, 3)  # Blue
    else:
        raise ValueError(f"Invalid color character '{color_char}'. Use 'r', 'g', or 'b'.")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using a trained diffusion model with Classifier-Free Guidance')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output', type=str, default='fm_output_ode.jpg', help='Path to save the image')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='CFG guidance scale (default: 3.0)')
    parser.add_argument('--label', type=str, default='0',choices=[str(i) for i in range(10)], help='Label to generate (0-9, default: 0)')
    parser.add_argument('--label_color', type=str, default='r', help='color of the digit (r/g/b), only valid when model variation is not base')
    parser.add_argument('--model_variation', type=str, choices=['base','sequential','parallel'], default='base', help='model variation: base or extend')
    parser.add_argument('--img_size', type=int, default=32, help='Image size (default: 32)')
    parser.add_argument('--channels', type=int, default=3, help='Number of image channels (default: 1)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--fm_steps', type=int, default=300, help='steps for flow matching')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size for ViT (default: 4)')
    parser.add_argument('--label_condition_dim', type=int, default=10, help='Label condition dimension (default: 10)')
    parser.add_argument('--color_condition_dim', type=int, default=3, help='Color condition dimension (default: 3), only valid when model variation is not base')
    return parser.parse_args()

def infer_flow_matching_cfg(model, x_0, conditions, guidance_scale=3.0, fm_steps=50):

    label_condition, color_label_condition = conditions
    model.eval()
    device = x_0.device
    x = x_0.clone()

    B = x.shape[0]
    dt = 1.0 / fm_steps  # 每一步的时间间隔
    uncond_condition_label = torch.zeros_like(label_condition).to(device)
    if color_label_condition is None:
        uncond_color_condition_label = None
    else:
        uncond_color_condition_label = torch.zeros_like(color_label_condition).to(device)

    with torch.no_grad():
        for i in range(fm_steps):
            t = torch.full((B,), i / fm_steps, dtype=torch.float, device=device)  # 当前时间 t ∈ [0,1)

            # 预测有条件与无条件速度场
            v_uncond = model(x, t, (uncond_condition_label, uncond_color_condition_label), force_drop_condition=True)
            v_cond = model(x, t, (label_condition, color_label_condition))

            v = v_uncond + guidance_scale * (v_cond - v_uncond)
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

    # get condition acctording to args
    label_condition = conditions_mappings[args.label].to(device)
    if args.model_variation != 'base':
        label_color_condition = get_corresponding_color_tensor(args.label_color).to(device)
    else:
        label_color_condition = None

    # get model according to args
    model = ExtendableDiT(
        img_size=(args.img_size, args.img_size),
        image_channels=args.channels,  # now 3 for RGB
        patch_size=args.patch_size,
        label_condition_dim=args.label_condition_dim,
        color_label_condition_dim=args.color_condition_dim,
        model_variant=args.model_variation
    ).to(device)
    
    model.load_state_dict(torch.load(args.ckpt, map_location=device),strict=False)
    model.eval()

    # Generate random noise with 3 channels
    x_t = torch.randn((1, args.channels, args.img_size, args.img_size), device=device)

    conditions = (label_condition, label_color_condition)
    with torch.no_grad():
        output = infer_flow_matching_cfg(model, x_t, conditions, args.guidance_scale, fm_steps=args.fm_steps)

    # Move to CPU and detach
    img_array = output.cpu().detach().numpy()
    img = img_array[0]  # shape: (3, H, W)

    # Normalize each channel independently to [0, 1]
    img = np.transpose(img, (1, 2, 0))  # (H, W, C)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Clip to [0, 1] just in case
    img = np.clip(img, 0, 1)

    # Save the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)  # No cmap for RGB
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    
    print(f"Generated RGB image for label {args.label} saved to {args.output}")

if __name__ == "__main__":
    main()