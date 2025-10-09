import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import torch.nn.init as init

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class PatchEmbedding(nn.Module):
    def __init__(self, image_channels, embed_dim, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(image_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size
        
    def forward(self, x):
        # Convert [B, C, H, W] -> [B, embed_dim, H/patch_size, W/patch_size]
        x = self.proj(x)
        # Reshape to [B, embed_dim, N] where N = H*W/(patch_size^2)
        B, C, H, W = x.shape
        x = x.flatten(2)  # [B, embed_dim, H*W]
        # Transpose to [B, N, embed_dim]
        x = x.transpose(1, 2)  # [B, H*W, embed_dim]
        return x, (H, W)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_ratio=4, dropout=0., short_circuit_init=False):
        """
        Transformer Block with optional identity (short-circuit) initialization.
        
        Args:
            dim (int): Embedding dimension.
            heads (int): Number of attention heads.
            dim_head (int): Dimension per head (unused here, kept for compatibility).
            mlp_ratio (float): Ratio for hidden dim in FFN.
            dropout (float): Dropout rate.
            short_circuit_init (bool): If True, initialize as identity block.
        """
        super().__init__()
        # Self-attention
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.norm_cross = nn.LayerNorm(dim)

        # FFN
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

        # Optional identity (short-circuit) initialization
        if short_circuit_init:
            self._init_as_identity()

    def _init_as_identity(self):
        """Initialize key layers to zero to make the block act as identity."""
        # Zero-init self-attention output projection
        init.zeros_(self.attn.out_proj.weight)
        init.zeros_(self.attn.out_proj.bias)

        # Zero-init cross-attention output projection
        init.zeros_(self.cross_attn.out_proj.weight)
        init.zeros_(self.cross_attn.out_proj.bias)

        # Zero-init the last Linear layer in FFN (critical for identity)
        init.zeros_(self.ffn[-2].weight)
        init.zeros_(self.ffn[-2].bias)

    def forward(self, x, t=None, c=None):
        # x shape: [B, N, embed_dim]
        # Self-attention
        res_x = x
        x_norm = self.norm1(x)
        x_attn = x_norm.transpose(0, 1)  # [N, B, embed_dim]
        x_attn, _ = self.attn(x_attn, x_attn, x_attn)
        x_attn = x_attn.transpose(0, 1)  # [B, N, embed_dim]
        x = res_x + x_attn

        # Cross-attention (if context provided)
        if c is not None:
            c = c.unsqueeze(1)  # [B, 1, embed_dim]
            x_norm = self.norm_cross(x).transpose(0, 1)      # [N, B, embed_dim]
            c_norm = self.norm_cross(c).transpose(0, 1)      # [1, B, embed_dim]
            cross_out, _ = self.cross_attn(x_norm, c_norm, c_norm)
            cross_out = cross_out.transpose(0, 1)            # [B, N, embed_dim]
            x = x + cross_out

        # FFN
        res_x = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = res_x + x

        return x
    

class SequentiallyExtendedDiT(nn.Module):
    """
    A simplified Diffusion Transformer (DiT) architecture.
    """
    def __init__(self, 
                 image_channels=1,
                 out_dim=None,
                 img_size=(32, 32),
                 patch_size=4,
                 embed_dim=768,
                 label_condition_dim=10,
                 color_label_condition_dim=3,
                 depth=6,
                 heads=12,
                 mlp_ratio=4,
                 time_emb_dim=32,
                 cfg_dropout=0.1,
                 ):
        super().__init__()
        
        # Store parameters
        self.image_channels = image_channels
        self.out_dim = out_dim if out_dim is not None else image_channels
        self.time_emb_dim = time_emb_dim
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.cfg_dropout = cfg_dropout  

        # for condition embedding
        self.label_condition_mlp = nn.Linear(label_condition_dim, embed_dim)
        self.color_label_condition_mlp = nn.Linear(color_label_condition_dim, embed_dim)
        
        # Check if image size is valid for the patch size
        assert self.img_height % patch_size == 0, f"Image height must be divisible by patch_size {patch_size}"
        assert self.img_width % patch_size == 0, f"Image width must be divisible by patch_size {patch_size}"
        
        # Calculate the number of patches
        self.num_patches = (self.img_height // patch_size) * (self.img_width // patch_size)
        self.h_patches = self.img_height // patch_size
        self.w_patches = self.img_width // patch_size
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, embed_dim)
        )
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(image_channels, embed_dim, patch_size)
        
        # Just a simple parameter position embedding inspired by ViT
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, embed_dim // heads, mlp_ratio, 0.1)
            for _ in range(depth)
        ])
        # Note that we used short_circuit_init = True when initalizing extended blocks
        self.extended_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, embed_dim // heads, mlp_ratio, 0.1,short_circuit_init=True)
            for _ in range(depth)
        ])
        
        # Output projection to image space
        self.norm = nn.LayerNorm(embed_dim)
        self.to_pixels = nn.Linear(embed_dim, patch_size * patch_size * self.out_dim)
        
    def forward(self, x,  timestep, conditions, force_drop_condition=False):

        label_condition, color_label_condition = conditions
        # Check input shape
        _, c, h, w = x.shape
        if c != self.image_channels:
            raise ValueError(f"Input should have {self.image_channels} channels, but got {c}")
        if h != self.img_height or w != self.img_width:
            print(f"Warning: Input size ({h},{w}) doesn't match configured size ({self.img_height},{self.img_width})")
        
        t = self.time_embedding(timestep)  # t: from [B] to [B, embed_dim]
        x, (H, W) = self.patch_embed(x)  # x: from [B, C, H, W] to [B, num_patches, embed_dim]
        
        # shape of pos_embed: [1, embed_dim] addictive condition
        addictive_condition = self.pos_embed + t.unsqueeze(1) 
        x += addictive_condition
        
        # Dropping condition
        if self.training or force_drop_condition:
            assert label_condition is not None and color_label_condition is not None
            assert label_condition.shape[0] == color_label_condition.shape[0]
            b = label_condition.shape[0]
            label_condition = label_condition.float()
            # c -> [B, condition_dim]
            drop_mask = torch.rand(b, device=label_condition.device) < self.cfg_dropout

            label_condition = label_condition.clone()
            label_condition[drop_mask] = 0  # Set dropped conditions to zero, looks like assign a single 0, but it actually assigns the whole vector as 0
            color_label_condition = color_label_condition.clone()
            color_label_condition[drop_mask] = 0

        label_condition = self.label_condition_mlp(label_condition)
        color_label_condition = self.color_label_condition_mlp(color_label_condition)

        # Apply transformer blocks
        # for simplicity, we assume extended_blocks and blocks have the same length
        assert len(self.extended_blocks) == len(self.blocks), \
            "extended_blocks and blocks must have the same length for alternating execution"

        for ext_block, block in zip(self.extended_blocks, self.blocks):
            x = ext_block(x, t, color_label_condition)  # 用 color 条件
            x = block(x, t, label_condition)            # 用 label 条件
         
        # Normalize and convert tokens back to image
        x = self.norm(x)
        x = self.to_pixels(x)  # [B, num_patches, patch_size*patch_size*out_dim]
        
        # Reshape to image
        B = x.shape[0]
        x = x.reshape(B, self.h_patches, self.w_patches, self.patch_size, self.patch_size, self.out_dim)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.out_dim, self.img_height, self.img_width)
    
        return x


if __name__ == "__main__":
    # Example 1: Default 1-channel model
    channels = 1
    model = SequentiallyExtendedDiT(
        image_channels=channels,
        img_size=(64, 64),
        patch_size=4,
        embed_dim=768,
        depth=24,
        heads=12,
        label_condition_dim=10,
        
    ).to(device)
    print("\nExample 1: Single-channel model")
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    print(f"Configured image size: {model.img_height}x{model.img_width}")
    print(f"Input channels: {model.image_channels}, Output channels: {model.out_dim}")
    
    # Create random input tensors matching the configured size
    batch_size = 16
    
    # Random image tensor with shape [batch_size, channels, height, width]
    x = torch.randn(batch_size, channels, 64, 64).to(device)
    c = torch.randn(batch_size, 10).to(device) # float
    c2 = torch.randn(batch_size, 3).to(device) # float
    # Random timestep tensor with shape [batch_size]
    timestep = torch.randint(0, 300, (batch_size,)).to(device)
    
    print(f"Input shape: {x.shape}")
    print(f"Timestep shape: {timestep.shape}")
    
    # Forward pass
    output = model(x, timestep, (c, c2))
    
    print(f"Output shape: {output.shape}")
    