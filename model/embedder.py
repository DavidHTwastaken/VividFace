
import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=64, temperature=100):
        super().__init__()

        self.num_freqs = num_freqs
        self.temperature = temperature

        freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)
        freq_bands = freq_bands[None, None, None]
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def __call__(self, x: torch.Tensor):
        x = self.freq_bands * x.unsqueeze(-1)
        return torch.stack((x.sin(), x.cos()), dim=-1).permute(0, 1, 3, 4, 2).reshape(*x.shape[:2], -1)

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x: torch.Tensor, latents: torch.Tensor):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q: torch.Tensor = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = torch.chunk(self.to_kv(kv_input), chunks=2, dim=-1)

        q = q.view(b, q.size(1), self.heads, -1).transpose(1, 2)
        k = k.view(b, k.size(1), self.heads, -1).transpose(1, 2)
        v = v.view(b, v.size(1), self.heads, -1).transpose(1, 2)

        out: torch.Tensor = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(b, l, -1)
        # out = out.to(q.dtype)
        out = self.to_out(out)
        return out


class FacePerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()
        
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, int(dim * ff_mult), bias=False),
                    nn.GELU(),
                    nn.Linear(int(dim * ff_mult), dim, bias=False),
                )
            ]) for _ in range(depth)
        ])


    def forward(self, latents: torch.Tensor, extra_latents: torch.Tensor):
        extra_latents = self.proj_in(extra_latents)
        for attn, ff in self.layers:
            latents = attn(extra_latents, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        latents = self.norm_out(latents)
        return latents


class ProjPlusModel(nn.Module):
    def __init__(self, 
        cross_attention_dim: int = 768, 
        id_embeddings_dim: int = 512, 
        attr_embeddings_dim: int = 512,
        dino_embeddings_dim: int = 768, 
        num_tokens: int = 8,
        shortcut: bool = False,
        depth: int = 4,
        ff_mult: int = 4
    ):
        super().__init__()

        self.shortcut = shortcut
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

        self.dino_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=depth,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=dino_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=ff_mult,
        )

        self.attr_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=depth,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=attr_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=ff_mult,
        )

    def forward(self, id_embeds: torch.Tensor, attr_embeds: torch.Tensor, dino_embeds: torch.Tensor, attr_scale: float = 1.0, dino_scale: float = 1.0):
        x: torch.Tensor = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        attr_out = self.attr_resampler(x, attr_embeds)
        dino_out = self.dino_resampler(x, dino_embeds)
        out = x + attr_scale * attr_out + dino_scale * dino_out
        return out


