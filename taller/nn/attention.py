import torch
import torch.nn as nn
import einops


# TODO: add QKVAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, dim: int = 64, n_heads: int = 4):
        """
        Multi-Head Attention module as described in Attention Is All
        You Need" (https://arxiv.org/abs/1706.03762).
        Args:
            input_dim: Input dimension.
            dim: Dimension of keys, queries, values.
            n_heads: Number of heads for multi-head attention.
        """
        super().__init__()

        self.dim = dim

        self.to_qkv = nn.Linear(input_dim, dim * n_heads * 3, bias=False)
        self.scale = dim ** -0.5

        self.unify_heads = nn.Linear(dim * n_heads, input_dim)

    def forward(self, x: torch.Tensor):
        qkv = self.to_qkv(x)
        q, k, v = einops.rearrange(qkv, 'b t (qkv h d) -> qkv b h t d', qkv=3, d=self.dim)     # [bs, h, t, d]

        dot = torch.einsum('bhtd, bhkd -> bhtk', q, k) * self.scale     # [bs, h, t, t]

        att = torch.softmax(dot, dim=-1)

        res = torch.einsum('bhdt, bhtv -> bhdv', att, v)
        res = einops.rearrange(res, 'b h t d -> b t (h d)')
        res = self.unify_heads(res)

        return res


class ConvAttention(nn.Module):
    def __init__(self, n_channels: int, dim_keys: int = 32, n_heads: int = 2):
        """
        Applies self-attention like in "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
        to an image by reshaping it into a sequence. Only for small field sizes.

        Args:
            n_channels (int): Number of channels of the input feature maps
            dim_keys (int): Dimension of queries, keys, and values
            n_heads (int): Number of heads for attention
        """
        super().__init__()
        self.scale = dim_keys ** -0.5
        self.heads = n_heads
        hidden_dim = dim_keys * n_heads
        self.to_qkv = nn.Conv2d(n_channels, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, n_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attention = sim.softmax(dim=-1)

        res = torch.einsum("b h i j, b h d j -> b h i d", attention, v)
        res = einops.rearrange(res, "b h (x y) d -> b (h d) x y", x=h, y=w)

        return self.to_out(res)


class LinearConvAttention(nn.Module):
    def __init__(self, n_channels: int, dim_keys: int = 32, n_heads: int = 2):
        """
        Efficient Attention (https://arxiv.org/abs/1812.01243), which instead of
        computing V (Q K.T) like in dot-product attention, computes Q (K.T V).
        This results in less complexity, O(d_k * d_v) instead of O(n²).

        Args:
            n_channels (int): Number of channels of the input feature maps
            dim_keys (int): Dimension of queries, keys, and values
            n_heads (int): Number of heads for attention
        """
        super().__init__()
        self.scale = dim_keys ** -0.5
        self.n_heads = n_heads
        hidden_dim = dim_keys * n_heads
        self.to_qkv = nn.Conv2d(n_channels, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, n_channels, 1),
                                    nn.GroupNorm(1, n_channels))

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self.n_heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        res = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        res = einops.rearrange(res, "b h c (x y) -> b (h c) x y", h=self.n_heads, x=h, y=w)

        return self.to_out(res)


if __name__ == "__main__":
    ipt = torch.randn((4, 1024, 8, 8))

    attn = MultiHeadAttention(1024, dim=32, n_heads=4)
    out = attn(einops.rearrange(ipt, 'b c h w -> b (h w) c'))
    print("Multi-Head Attention")
    print("\tInput:", ipt.shape)
    print("\tOutput:", out.shape)

    attn = ConvAttention(1024, dim_keys=32, n_heads=2)
    out = attn(ipt)
    print("Conv Attention")
    print("\tInput:", ipt.shape)
    print("\tOutput:", out.shape)

    lin_attn = LinearConvAttention(1024, dim_keys=32, n_heads=2)
    out = lin_attn(ipt)
    print("Linear Conv Attention")
    print("\tInput:", ipt.shape)
    print("\tOutput:", out.shape)
