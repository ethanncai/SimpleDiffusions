# visualization_utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from fm_ode_infer import get_corresponding_color_tensor, conditions_mappings,infer_flow_matching_cfg

def generate_comprehensive_visualization(model, args, device, global_step, save_path, timestamp, writer, fixed_seed=None):
    """
    生成所有标签(0-9)和颜色('r','g','b')组合的可视化结果
    布局为3x10矩阵：3行(r,g,b) x 10列(0-9)
    使用批量推理提高效率
    """
    print(f"\n[Generating comprehensive visualization at step {global_step}]")
    
    # Set fixed seed for reproducibility during visualization
    if fixed_seed is not None:
        torch.manual_seed(fixed_seed)
        np.random.seed(fixed_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fixed_seed)
    
    model.eval()
    
    # 准备条件：3种颜色 x 10个数字 = 30个组合
    colors = ['r', 'g', 'b']
    labels = [str(i) for i in range(10)]  # ['0', '1', ..., '9']
    
    # 按行优先顺序生成条件组合 (r0,r1,...,r9, g0,g1,...,g9, b0,b1,...,b9)
    all_conditions = []
    for color in colors:
        for label in labels:
            all_conditions.append((label, color))
    
    total_samples = len(all_conditions)  # 30
    
    print(f"Generating {total_samples} images in batch...")
    
    # 创建批量噪声 (30, channels, img_size, img_size)
    x_t_batch = torch.randn((total_samples, args.channels, args.img_size, args.img_size), device=device)
    
    # 准备批量条件 - 先获取单个样本的条件来了解维度
    sample_label_condition = conditions_mappings[labels[0]]  # 获取样本条件
    sample_color_condition = get_corresponding_color_tensor(colors[0]) if args.model_variation != 'base' else None
    
    print(f"Sample label condition shape: {sample_label_condition.shape}")
    if sample_color_condition is not None:
        print(f"Sample color condition shape: {sample_color_condition.shape}")
    
    # 准备批量条件
    batch_label_conditions = []
    batch_color_conditions = []
    
    for label, color in all_conditions:
        # 获取标签条件
        label_condition = conditions_mappings[label]  # 不要在这里添加 .to(device)
        batch_label_conditions.append(label_condition)
        
        # 获取颜色条件
        if args.model_variation != 'base':
            color_condition = get_corresponding_color_tensor(color)  # 不要在这里添加 .to(device)
            batch_color_conditions.append(color_condition)
        else:
            batch_color_conditions.append(None)
    
    # 堆叠成批量张量并确保维度正确
    batch_label_conditions = torch.stack(batch_label_conditions).to(device)  # (30, ...)
    
    if args.model_variation != 'base' and batch_color_conditions[0] is not None:
        batch_color_conditions = torch.stack(batch_color_conditions).to(device)  # (30, ...)
    else:
        batch_color_conditions = None
    
    print(f"Batch label conditions shape: {batch_label_conditions.shape}")
    if batch_color_conditions is not None:
        print(f"Batch color conditions shape: {batch_color_conditions.shape}")
    
    # 检查维度并确保正确的格式
    # 如果标签条件是1D的，需要确保它有正确的批量维度
    batch_label_conditions = batch_label_conditions.squeeze(1)
    
    if batch_color_conditions is not None:
        if batch_color_conditions.dim() == 2:
            # 应该是 (batch_size, condition_dim)，这是正确的
            pass
        elif batch_color_conditions.dim() == 1:
            # 如果是1D的，需要添加条件维度
            batch_color_conditions = batch_color_conditions.unsqueeze(-1)
    
    # 批量推理
    print("Running batch inference...")
    with torch.no_grad():
        # print(batch_label_conditions.shape)
        # print(x_t_batch.shape)
        # print(batch_color_conditions.shape)
        batch_outputs = infer_flow_matching_cfg(
            model, x_t_batch, (batch_label_conditions, batch_color_conditions),
            guidance_scale=3.0,
            fm_steps=50
        )
    
    print("Batch inference completed, creating visualization...")
    
    # 创建3x10的网格图像
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    fig.suptitle(f'Comprehensive Visualization - Step {global_step}\nRows: Colors (r,g,b) | Columns: Digits (0-9)', fontsize=14)
    
    # 处理批量输出并组织成网格
    grid_images = []
    
    for idx, (output, (label, color)) in enumerate(zip(batch_outputs, all_conditions)):
        # 处理图像
        img_array = output.cpu().detach().numpy()  # (C, H, W)
        img_array = np.transpose(img_array, (1, 2, 0))  # (H, W, C)
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        img_array = np.clip(img_array, 0, 1)
        
        # 计算在3x10网格中的位置
        row = idx // 10  # 颜色行 (0=r, 1=g, 2=b)
        col = idx % 10   # 数字列 (0-9)
        
        axes[row, col].imshow(img_array)
        axes[row, col].set_title(f'{color}{label}', fontsize=10)
        axes[row, col].axis('off')
        
        # 保存用于TensorBoard的图像
        grid_images.append(img_array)
    
    # 添加行和列标签
    color_names = ['Red', 'Green', 'Blue']
    for i, color_name in enumerate(color_names):
        axes[i, 0].set_ylabel(color_name, rotation=0, labelpad=50, fontsize=12, ha='right')
    
    for j in range(10):
        axes[0, j].set_xlabel(f'Digit {j}', fontsize=10)
        axes[0, j].xaxis.set_label_position('top')
    
    # 保存网格图像
    plt.tight_layout()
    vis_img_path = os.path.join(save_path, f"comprehensive_vis_{timestamp}_step_{global_step}.jpg")
    plt.savefig(vis_img_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()
    print(f"Saved comprehensive visualization to {vis_img_path}")
    
    # 为TensorBoard创建单个网格图像
    grid_h, grid_w = 3, 10
    img_h, img_w = args.img_size, args.img_size
    
    # 创建大的网格图像
    grid_image = np.zeros((grid_h * img_h, grid_w * img_w, 3))
    
    for idx, img_array in enumerate(grid_images):
        row = idx // grid_w
        col = idx % grid_w
        start_h, end_h = row * img_h, (row + 1) * img_h
        start_w, end_w = col * img_w, (col + 1) * img_w
        grid_image[start_h:end_h, start_w:end_w] = img_array
    
    # 添加到TensorBoard
    grid_tensor = torch.from_numpy(grid_image).permute(2, 0, 1).float()  # (C, H, W)
    writer.add_image('Comprehensive_Visualization', grid_tensor, global_step)
    
    model.train()  # 切换回训练模式
    print(f"Comprehensive visualization completed for step {global_step}")

