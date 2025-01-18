# type: ignore

"""

Simple implementation of Tensor Product Attention from the T6 paper
https://arxiv.org/abs/2501.06425

Done without reference to the source code, so may not be faithful to the
original work. Basic idea is to use tensor decompositions to represent
Q/K/Vs by factorizing them into low-rank components. Has a 10x smaller
kv_cache and (apparently) no degradation to perplexity tested up to
~700m param models.

TODO:
- kernel to compute attn directly from factors
- support higher-order tpa
- use max_seq_len and pre-allocate kvcache

Execution results:

> nvidia-smi
    Sat Jan 18 19:56:41 2025
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA GeForce RTX 4090        On  |   00000000:02:00.0 Off |                  Off |
    |  0%   37C    P8             22W /  450W |       2MiB /  24564MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

> uv run tpa.py

    Testing forward pass...
    Output shape: torch.Size([1, 2048, 8192])

    Benchmarking forward pass...
    Output shape: torch.Size([1, 2048, 8192])
    Avg time per step: 6.15ms
    Tokens per second (prefill): 332825.2

    Testing chunked/autoregressive generation...
    Maximum difference between full and chunked: 0.00000716
    Mean difference between full and chunked: 0.00000010
    Outputs match: True

    Testing generation...
    With cache: 3696.34ms (554.1 tokens/sec)
    No cache: 22526.97ms (90.9 tokens/sec)

    Speedup from KV cache: 6.09x
"""

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)

Tensor = torch.Tensor

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, False)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight.to(input.dtype))

class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.register_buffer("inv_freq", (1 / base) ** (torch.arange(0, dim, 2) / dim))

    def forward(self, x: Tensor, cache_len: int):
        T = x.shape[1]
        t = torch.arange(T, device=x.device) + cache_len
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]
        x1, x2 = x.chunk(2, dim=3)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)

def create_causal_mod(cache_len: int):
    def causal_mod(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx + cache_len
    return causal_mod

class TPAttention(nn.Module):
    """Tensor Product Attention with factorizes Q/K/V values"""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        r_Q: int = 1,
        r_K: int = 1,
        r_V: int = 1,
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0, f"d_model ({d_model}) must be evenly divisible by n_head ({n_head})"
        d_head = d_model // n_head

        self.nh = n_head
        self.dh = d_head
        self.ch = n_head + d_head

        self.r_Q = r_Q
        self.r_K = r_K
        self.r_V = r_V

        self.ab_q = CastedLinear(d_model, (r_Q * self.ch))
        self.ab_kv = CastedLinear(d_model, (r_K * self.ch) + (r_V * self.ch))
        self.W_o = CastedLinear(d_model, d_model)

        self.rotary = Rotary(dim=d_head)

        ab_q_tensor = self.ab_q.weight.view(d_model, r_Q, n_head + d_head)
        nn.init.xavier_uniform_(ab_q_tensor)
        self.ab_q.weight.data = ab_q_tensor.view_as(self.ab_q.weight)
        ab_kv_tensor = self.ab_kv.weight.view(d_model, r_K + r_V, n_head + d_head)
        nn.init.xavier_uniform_(ab_kv_tensor)
        self.ab_kv.weight.data = ab_kv_tensor.view_as(self.ab_kv.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

    @torch.compile
    def factorized_qkv(self, x: Tensor, kv_cache: Tensor):
        B, T, D = x.shape
        nh, dh = self.nh, self.dh

        abq = self.ab_q(x).reshape(B, T, self.r_Q, nh + dh)
        abkv = self.ab_kv(x).reshape(B, T, self.r_K + self.r_V, nh + dh)
        abkv[:, :, :, nh:] = self.rotary(abkv[:, :, :, nh:], kv_cache.size(1)) # TODO: add rms norm
        abkv = torch.cat((kv_cache, abkv), dim=1) # TODO: pre-allocate max_seq_len

        a_q = abq[..., :nh]
        b_q = abq[..., nh:]
        a_k = abkv[..., :self.r_K, :nh]
        b_k = abkv[..., :self.r_K, nh:]
        a_v = abkv[..., self.r_K:, :nh]
        b_v = abkv[..., self.r_K:, nh:]

        q = torch.einsum("btrh, btrd -> bthd", a_q, b_q) * (1/self.r_Q)
        k = torch.einsum("btrh, btrd -> bthd", a_k, b_k) * (1/self.r_K)
        v = torch.einsum("btrh, btrd -> bthd", a_v, b_v) * (1/self.r_V)

        return (q, k, v), abkv

    def forward(self, x: Tensor, kv_cache: Tensor | None = None) -> tuple[Tensor, Tensor]:
        B, T, D = x.shape
        nh, dh = self.nh, self.dh
        if kv_cache is None:
            kv_cache = torch.zeros((B, 0, self.r_K + self.r_V, nh + dh), device=x.device, dtype=self.ab_kv.weight.dtype)

        # Factorization w/ cache
        (q, k, v), kv_cache = self.factorized_qkv(x, kv_cache)

        # Attn operation
        causal_mod = create_causal_mod(kv_cache.size(1) - T)
        block_mask = create_block_mask(
            causal_mod, B=None, H=None, Q_LEN=T, KV_LEN=k.size(1)
        )

        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            kernel_options={
                # kernel options for a 40GB card (gpu poor)
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "BLOCK_M1": 16,
                "BLOCK_N1": 32,
                "BLOCK_M2": 32,
                "BLOCK_N2": 16,
            },
        )
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.W_o(y)
        return y, kv_cache



if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    D_MODEL = 8192
    N_HEAD = D_MODEL // 128
    T = 4096

    # Create attention block
    device = "cuda"
    model = TPAttention(
        d_model=D_MODEL,
        n_head=N_HEAD,
        r_Q=1,
        r_K=1,
        r_V=1,
    ).to(device).bfloat16()
    for param in model.named_parameters():
        if isinstance(param, CastedLinear):
            param.float()
    flex_attention = torch.compile(flex_attention)

    # Prefill prompt
    prompt_len = 2048
    gen_len = 2048
    prompt = torch.randn((1, prompt_len, D_MODEL), device=device, dtype=torch.float32)

    # Forward pass (warmup)
    print("\nTesting forward pass...")
    with torch.no_grad():
        for _ in range(25):
            out, _ = model(prompt)
    torch.cuda.synchronize()
    print(f"Output shape: {out.shape}")

    print("\nBenchmarking forward pass...")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    with torch.no_grad():
        for _ in range(1000):
            out, _ = model(prompt)

    end_time.record()
    torch.cuda.synchronize()
    elapsed_ms = start_time.elapsed_time(end_time)
    tokens_per_sec = (prompt.size(1) * 1000 * 1000) / elapsed_ms
    print(f"Output shape: {out.shape}")
    print(f"Avg time per step: {elapsed_ms/1000:.2f}ms")
    print(f"Tokens per second (prefill): {tokens_per_sec:.1f}")

    print("\nTesting chunked/autoregressive generation...")
    def test_chunked_generation():
        # Test with full prompt
        with torch.no_grad():
            out_full, _ = model(prompt)

        # Test with chunked prompt
        split_idx = prompt_len // 2
        prompt_chunk1 = prompt[:, :split_idx]
        prompt_chunk2 = prompt[:, split_idx:]

        with torch.no_grad():
            out_chunk1, kv_cache = model(prompt_chunk1)
            out_chunk2, _ = model(prompt_chunk2, kv_cache)
            out_chunked = torch.cat([out_chunk1, out_chunk2], dim=1)

        # Compare results
        max_diff = torch.max(torch.abs(out_full - out_chunked))
        mean_diff = torch.mean(torch.abs(out_full - out_chunked))
        print(f"Maximum difference between full and chunked: {max_diff:.8f}")
        print(f"Mean difference between full and chunked: {mean_diff:.8f}")
        print(f"Outputs match: {mean_diff < 1e-3}")

    test_chunked_generation()

    print("\nTesting generation...")
    def test_generation(use_cache: bool):
        tokens = prompt.clone()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        # Forward passes
        with torch.no_grad():
            tokens, kv_cache = model(tokens, None)
            pred = tokens

            for i in range(gen_len):
                if use_cache:
                    pred, kv_cache = model(pred[:, -1:], kv_cache)
                else:
                    pred, _ = model(tokens, None)
                    tokens = torch.cat([tokens, pred[:, -1:]], dim=1)

        end_time.record()
        torch.cuda.synchronize()
        elapsed_ms = start_time.elapsed_time(end_time)
        tokens_per_sec = (gen_len * 1000) / elapsed_ms
        return elapsed_ms, tokens_per_sec

    # Test with cache
    time_with_cache, tps_with_cache = test_generation(use_cache=True)
    print(f"With cache: {time_with_cache:.2f}ms ({tps_with_cache:.1f} tokens/sec)")

    # Test without cache
    time_no_cache, tps_no_cache = test_generation(use_cache=False)
    print(f"No cache: {time_no_cache:.2f}ms ({tps_no_cache:.1f} tokens/sec)")

    speedup = time_no_cache / time_with_cache
    print(f"\nSpeedup from KV cache: {speedup:.2f}x")
