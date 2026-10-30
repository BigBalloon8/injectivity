import torch
import torch.nn as nn
import torch.nn.functional as F
from projection import project_kernel_into_safe_region
from proj_v3 import nearest_with_kernel_angle_bound_fast

class FFN(nn.Module):
    def __init__(self, dim, h, realized=False):
        super().__init__()
        self.l1 = nn.Linear(dim, h)
        self.l2 = nn.Linear(h, dim)
        self.realized = realized
        self.realized_sample = None
        
    
    def forward(self, x):
        acts = torch.relu(self.l1(x))
        if self.realized:
            self.realized_sample = acts.reshape(-1, acts.shape[-1])
        return self.l2(acts)
    
    @torch.no_grad()
    def proj(self, gamma=0.4):
        up_proj = self.l1.weight
        down = self.l2.weight
        #new_down = nearest_with_kernel_angle_bound_fast(up_proj, down, gamma)
        new_down, _, _ = project_kernel_into_safe_region(down, up_proj, gamma_lower=gamma)
        self.l2.weight.set_(new_down)
    
    
    def mean_sq_cos(self):
        # W: (d_ff, d_model); rows are neuron directions
        W = F.normalize(self.l1.weight, dim=1)
        n, d = W.shape
        fp = (W.T @ W).pow(2).sum()        # sum of cos² over all pairs incl. diagonal (which sums to n)
        return (fp - n) / (n * (n - 1))    # mean cos² over distinct pairs; floor ≈ 1/d when n ≫ d
            
        

class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        dim_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = dim_q == dim_k and dim_q == dim_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(dim_q, dim_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(dim_q, dim_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(dim_k, dim_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(dim_v, dim_total, bias=bias, **factory_kwargs)
        E_out = dim_q
        self.out_proj = nn.Linear(dim_total, E_out, bias=bias, **factory_kwargs)
        assert dim_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.dim_head = dim_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.dim_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.dim_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.dim_head]).transpose(1, 2)

        # (N, nheads, L_t, E_head)
        if self.training:
            dropout = self.dropout
        else:
            dropout = 0.0
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout, is_causal=is_causal, attn_mask=attn_mask
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output
    
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, dim, h_dim, n_heads=2, realized=False, ffn_norm=True):
        super().__init__()
        self.atten = MultiHeadAttention(dim, dim, dim, dim, n_heads, dropout=0.05)
        self.ffn = FFN(dim, h_dim, realized)
        self.atten_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim) if ffn_norm else nn.Identity()
    
    def forward(self, x):
        normed_in = self.atten_norm(x)
        h = x + self.atten(normed_in, normed_in, normed_in)
        return h + self.ffn(self.ffn_norm(h))
    
    def proj(self, gamma=0.4):
        self.ffn.proj(gamma)
    
    def mean_sq_cos(self):
        return self.ffn.mean_sq_cos()
    
class Transformer(nn.Module):
    def __init__(self, dim, h_dim, n_heads, n_layers, vocab_size, realized=False, ffn_norm=True, freeze_embedding=False):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, dim, _freeze=freeze_embedding)
        self.layers = nn.ModuleList([TransformerBlock(dim, h_dim, n_heads, realized=realized, ffn_norm=ffn_norm) for _ in range(n_layers)])
        self.out_norm = RMSNorm(dim)
        self.out = nn.Linear(dim, vocab_size)
        self.realized = realized
    
    def forward(self, token_ids):
        embeddings = self.embeddings(token_ids)
        h = embeddings
        for l in self.layers:
            h = l(h)
        if self.realized:
            self.out_states = h
        logits = self.out_norm(h)
        return self.out(logits)

    def proj(self, idxs=None, gamma=0.4):
        if idxs is None:
            idxs = list(range(len(self.layers)))
            
        if isinstance(idxs, int):
            idxs = [idxs]
        
        for i, b in enumerate(self.layers):
            if i in idxs:
                b.proj(gamma)
                
    def get_realized(self):
        if self.realized:
            samples = []
            for b in self.layers:
                samples.append(
                    b.ffn.realized_sample
                )
            return samples
            
    def mean_sq_cos(self):
        return sum([l.mean_sq_cos() for l in self.layers]) / len(self.layers)
        