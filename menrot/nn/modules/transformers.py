import torch
from torch import nn
import einops.layers.torch as einops_nn
from einops import rearrange, repeat

__all__ = [
    "VisionTransformer3d",
    "AutoregTransformer"
]

# source: https://github.com/lucidrains/vit-pytorch

#--- sub-modules ---#
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, heads=8, head_dim=64, dropout = 0.):
        super().__init__()
        inner_dim = head_dim *  heads
        project_out = not (heads == 1 and head_dim == embed_dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(embed_dim)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)    
        )

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # b, h, n
        if mask is not None:
            n = dots.shape[-1]  # number of tokens

            if mask.dim() != 2 or mask.shape[0] != mask.shape[1] or mask.shape[0] != n:
                raise ValueError(
                    f"[Invalid Attenion Mask] Expected a square 2D mask of shape [{n}, {n}], "
                    f"but got {mask.shape}."
                )
            
            mask = mask.view(1,1,n,n)
            dots = dots + mask  # broadcastable to [B, h, N, N]
            
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth, heads, head_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiheadAttention(embed_dim, heads = heads, head_dim = head_dim, dropout = dropout),
                FeedForward(embed_dim, hidden_dim, dropout = dropout)
            ]))
            
        self.cls_tokens = None

    def forward(self, x, mask=None):
        self.cls_tokens = []
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
            self.cls_tokens.append(self.norm(x))#[:, 0])
            
        return self.norm(x)


class PatchEmbedding3d(torch.nn.Module):
    # source: https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html#patch-embedding
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        patch_frame: int,
        embed_dim: int=512,
        groups: int=1,
        conv_version=True
    ) -> None:
        super().__init__()        
        if conv_version:
            self.patch_embed = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, groups=groups),
                einops_nn.Rearrange('b e f h w -> b (f h w) e', e=embed_dim)
            )
        else:
            patch_dim = in_channels * patch_size * patch_size * patch_frame
            self.patch_embed = torch.nn.Sequential(
                einops_nn.Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (c pf p1 p2)', pf= patch_frame, p1= patch_size, p2= patch_size),
                torch.nn.LayerNorm(patch_dim),
                torch.nn.Linear(patch_dim, embed_dim), # b (h w) (c p1 p2) -> b (h w) e
                torch.nn.LayerNorm(patch_dim)
            )
    
    def forward(self, x):
        return self.patch_embed(x)

#--- transofrmers class ---#
class VisionTransformer3d(nn.Module):
    def __init__(self, 
                 image_size, 
                 patch_size, 
                 frame_size, 
                 frame_patch_size,
                 num_classes, 
                 embed_dim, 
                 depth, 
                 hidden_dim,         # should be: 4 * embed_dim 
                 heads,              # should be: embed_dim / 64
                 head_dim = 64, 
                 channels = 3, 
                 pool = 'cls', 
                 dropout = 0.2, 
                 embed_dropout = 0.1,
                 groups=1,
                 pos_embed_type = 'normal'
    ):
        super().__init__()
        image_height, image_width = self._pair(image_size)
        patch_height, patch_width = self._pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frame_size % frame_patch_size == 0, 'Frame size must be divisible by frame patch size'
         
        assert pool in {'cls', 'mean', 'none', 'spatial'}, 'pool type must be either cls (only cls token), mean (mean pooling), spatial (only spatial patches) or none (all token)'
        assert embed_dim//head_dim == heads, f'Heads should be = embed_dim // head_dims={embed_dim // head_dim}, given embed_dim={embed_dim} and head_dim={head_dim}, but got heads={heads}.'
        
        self.embed_dim = embed_dim
        
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frame_size // frame_patch_size)
        self.to_patch_embed = PatchEmbedding3d(channels, patch_frame=frame_patch_size,  patch_size=patch_size, embed_dim=embed_dim, groups=groups)
        
        self.pos_embed_type = pos_embed_type
        if pos_embed_type == 'normal':
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        elif pos_embed_type == 'none':
            self.pos_embed = None

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(embed_dropout)
            
        self.transformer_encoder = TransformerEncoder(embed_dim, depth, heads, head_dim, hidden_dim, dropout)

        self.pool = pool
        
        self.num_classes = num_classes
        self.reshape_out = False
        if isinstance(num_classes, list):
            self.reshape_out = True
            self.string_lengt =  num_classes[0]
            self.num_char = num_classes[1]
            self.num_classes = num_classes[0]*num_classes[1]

        self.mlp_head = nn.Linear(embed_dim, self.num_classes) if self.num_classes is not None else nn.Identity()

    def _pair(self, t):
        return t if isinstance(t, tuple) else (t, t)
    
    def forward(self, img):
        x = self.to_patch_embed(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)    

        if self.pos_embed_type == 'normal':
            x += self.pos_embed #[:, :(n + 1)]
        elif self.pos_embed_type == 'none':
            pass
            
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        if self.pool == 'mean':
            x = x.mean(dim = 1)  # avg all token 
        elif self.pool == 'cls':
            x = x[:, 0]   # remove spatial patches, keep only cls
        elif self.pool == 'spatial':
            x = x[:, 1:]  # remove CLS, keep only spatial patches
        elif self.pool == 'none':
            pass # keep CLS + spatial
        
        if self.reshape_out:
            return self.mlp_head(x).view(-1, self.string_lengt, self.num_char)
        else:
            return self.mlp_head(x)


class AutoregTransformer(nn.Module):
    def __init__(self, embed_dim=512, vocab_size=6, seq_len=9, num_layers=3, heads=8):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.sos_token_id = vocab_size  # e.g. token ID 6 for <SOS>, vocab = 0..5

        # Token embedding for vocabulary tokens + 1 for <SOS>
        self.token_embed = nn.Embedding(vocab_size+1, embed_dim)  # [vocab+1, D]
        self.pos_embed = nn.Parameter(torch.randn(seq_len, embed_dim)) # [seq_len, D]

        # Transformer decoder: memory for cross-attention comes from ViT output tokens
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=embed_dim, nhead=heads, batch_first=True),
            num_layers=num_layers
        )

        self.fc_out = nn.Linear(embed_dim, vocab_size)  # Projection to vocab: [D → vocab]

    def forward(self, input_tokens, memory, tgt_mask=None):
        """
        input_tokens: [B, T] — token indices including <SOS> + GT tokens (for training)
        memory:       [B, T, D] — Output token from ViT encoder
        tgt_mask:     [T, T] — causal mask for decoding

        Returns:
            logits: [B, T, vocab_size]
        """
        _, T = input_tokens.size()

        # Embed tokens + add positional encoding
        token_embed = self.token_embed(input_tokens)      # [B, T, D]
        pos_embed = self.pos_embed[:T, :]                 # [T, D]

        # broadcast → [B, T, D]
        token_embed += pos_embed.unsqueeze(0)    # [B, T, D] + [1, T, D] → [B, T, D]

        # memory 
        if memory.dim() == 2:
            memory = memory.unsqueeze(1) # [B, 1, D]


        decoded = self.transformer_decoder(
            tgt=token_embed,        # [B, T, D]
            memory=memory,          # [B, 1, D]
            tgt_mask=tgt_mask       # [T, T]
        ) # → [B, T, D]

        logits = self.fc_out(decoded)
        return logits # [B, T, vocab_size]
        
    