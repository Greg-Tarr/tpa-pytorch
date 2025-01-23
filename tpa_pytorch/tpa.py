# type: ignore

"""

Simple implementation of Tensor Product Attention from the T6 paper
https://arxiv.org/abs/2501.06425

Done without reference to the source code, so may not be faithful to the
original work. Basic idea is to use tensor decompositions to represent
Q/K/Vs by factorizing them into low-rank components. Has a 10x smaller
kv_cache and no degradation to perplexity tested up to ~700m param models.

TODO:
    - kernel to compute attn directly from factors
    - support higher-order tpa

Changelog:
    - added max_seq_len to pre-allocate kvcache
    - made cache a separate class
    - made cache an arguement to the forward pass to allow torch.compile

Execution results:

> nvidia-smi
    Thu Jan 23 17:09:29 2025
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 565.57.01              Driver Version: 565.57.01      CUDA Version: 12.7     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA A40                     On  |   00000000:05:00.0 Off |                    0 |
    |  0%   38C    P8             31W /  300W |       1MiB /  46068MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

> uv run tpa.py

    Warming up model...
    Output shape: torch.Size([1, 2048, 8192])

    Benchmarking With Cache prefill...
    Avg time per step: 9.35ms
    Tokens per second (prefill): 219072.6

    Testing chunked/autoregressive generation with With Cache...
    Maximum difference between full and chunked: 0.00000072
    Mean difference between full and chunked: 0.00000001
    Outputs match: True

    Testing generation with With Cache...

    Total Time: 6257.58ms (327.3 tok/sec)
    Time per token: 3.06ms
    Max throughput: 327.3 tok/sec
    With Cache: 6257.58ms (327.3 tokens/sec)

    Warming up model...
    Output shape: torch.Size([1, 2048, 8192])

    Benchmarking No Cache prefill...
    Avg time per step: 9.86ms
    Tokens per second (prefill): 207768.2

    Testing generation with No Cache...

    Total Time: 49092.46ms (41.7 tok/sec)
    Time per token: 23.97ms
    Max throughput: 41.7 tok/sec
    No Cache: 49092.46ms (41.7 tokens/sec)

    Comparing with and without cache speeds.

    Testing generation with No Cache...

    Total Time: 51251.46ms (40.0 tok/sec)
    Time per token: 25.03ms
    Max throughput: 40.0 tok/sec

    Testing generation with With Cache...

    Total Time: 4204.21ms (487.1 tok/sec)
    Time per token: 2.05ms
    Max throughput: 487.1 tok/sec

    Speedup from KV cache: 12.19x
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

class NoKVCache(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = 0

    def zero(self):
        self.offset = 0

    def forward(self, abkv: Tensor) -> Tensor:
        self.offset += abkv.size(1)
        return abkv

class KVCache(nn.Module):
    def __init__(self, kv_cache_shape: tuple = None) -> None:
        super().__init__()
        self.kv_cache_shape = kv_cache_shape
        self.register_buffer('data', torch.zeros((1, *self.kv_cache_shape)))
        self.zero()

    def zero(self):
        self.offset = 0
        self.data.zero_()

    def forward(self, abkv: Tensor) -> Tensor:
        assert self.offset + abkv.size(1) <= self.data.size(1), "KV Cache Exceeded"

        self.data = self.data.to(abkv.dtype)
        self.data[
            :, self.offset : self.offset + abkv.size(1), :, :
        ] = abkv
        self.offset += abkv.size(1)

        return self.data[:, :self.offset]


class TPAttention(nn.Module):
    """Tensor Product Attention with factorizes Q/K/V values"""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_seq_len: int = 8192,
        r_Q: int = 1,
        r_K: int = 1,
        r_V: int = 1,
    ) -> None:
        super().__init__()
        assert (
            d_model % n_head == 0
        ), f"d_model ({d_model}) must be evenly divisible by n_head ({n_head})"
        d_head = d_model // n_head

        self.nh = n_head
        self.dh = d_head
        self.ch = n_head + d_head
        self.max_seq_len = max_seq_len

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

    def forward(self, x: Tensor, cache: KVCache | NoKVCache) -> tuple[Tensor, Tensor]:
        B, T, D = x.shape
        nh, dh = self.nh, self.dh

        # Projections Q for each of R_x, a(nh) & b(dh)
        abq = self.ab_q(x).reshape(B, T, self.r_Q, nh + dh)
        a_q = abq[..., :nh]
        b_q = abq[..., nh:]

        # Projection for KV for each of R_x, a(nh) & b(dh)
        abkv = self.ab_kv(x).reshape(B, T, self.r_K + self.r_V, nh + dh)

        # Apply rotary to the dh section of K&V offset by cache
        abkv[:, :, :, nh:] = self.rotary(abkv[:, :, :, nh:], cache.offset)

        # Insert and retrieve cache (noop if no cache)
        kv_cache = cache(abkv)

        # Extract current kv cache
        a_k = kv_cache[..., : self.r_K, :nh]
        b_k = kv_cache[..., : self.r_K, nh:]
        a_v = kv_cache[..., self.r_K :, :nh]
        b_v = kv_cache[..., self.r_K :, nh:]

        # Do the a @ p operation and sum over r_X
        q = torch.einsum("btrh, btrd -> bthd", a_q, b_q) * (1 / self.r_Q)
        k = torch.einsum("btrh, btrd -> bthd", a_k, b_k) * (1 / self.r_K)
        v = torch.einsum("btrh, btrd -> bthd", a_v, b_v) * (1 / self.r_V)

        # Attn operation
        # TODO: support batches with masking
        causal_mod = create_causal_mod(cache.offset - T) # Pre-offset size
        block_mask = create_block_mask(
            causal_mod, B=None, H=None, Q_LEN=q.size(1), KV_LEN=k.size(1), _compile=True
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
        return y

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # Model configuration
    D_MODEL = 8192
    N_HEAD = D_MODEL // 128
    T = 4096
    device = "cuda"

    # Create both non-cached and cached models
    model_nocache = (
        TPAttention(
            d_model=D_MODEL,
            n_head=N_HEAD,
            max_seq_len=T,
            r_Q=1,
            r_K=1,
            r_V=1,
        )
        .to(device)
        .bfloat16()
        .eval()
    )
    no_cache = NoKVCache()
    model_nocache = torch.compile(model_nocache)

    model_cache = (
        TPAttention(
            d_model=D_MODEL,
            n_head=N_HEAD,
            max_seq_len=T,
            r_Q=1,
            r_K=1,
            r_V=1,
        )
        .to(device)
        .bfloat16()
        .eval()
    )
    kv_cache = KVCache(kv_cache_shape=(model_cache.max_seq_len, model_cache.r_K + model_cache.r_V, model_cache.nh + model_cache.dh)).to(device)
    model_cache = torch.compile(model_cache)

    # Test parameters
    prompt_len = 2048
    gen_len = 2048
    prompt = torch.randn((1, prompt_len, D_MODEL), device=device, dtype=torch.float32)

    def warmup(model, cache):
        print("\nWarming up model...")
        with torch.no_grad():
            for _ in range(25):
                cache.zero()
                out = model(prompt, cache)
        torch.cuda.synchronize()
        print(f"Output shape: {out.shape}")
        return out

    def benchmark_prefill(model, name, cache):
        print(f"\nBenchmarking {name} prefill...")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        with torch.no_grad():
            for _ in range(1000):
                cache.zero()
                out = model(prompt, cache)
        end_time.record()

        torch.cuda.synchronize()
        elapsed_ms = start_time.elapsed_time(end_time)
        tokens_per_sec = (prompt.size(1) * 1000 * 1000) / elapsed_ms
        print(f"Avg time per step: {elapsed_ms/1000:.2f}ms")
        print(f"Tokens per second (prefill): {tokens_per_sec:.1f}")
        return out

    def test_chunked_generation(model, name, cache):
        print(f"\nTesting chunked/autoregressive generation with {name}...")

        # Test with full prompt
        cache.zero()
        with torch.no_grad():
            out_full = model(prompt, cache)

        # Test with chunked prompt
        split_idx = prompt_len // 2
        prompt_chunk1 = prompt[:, :split_idx]
        prompt_chunk2 = prompt[:, split_idx:]

        cache.zero()
        with torch.no_grad():
            out_chunk1 = model(prompt_chunk1, cache)
            out_chunk2 = model(prompt_chunk2, cache)
            out_chunked = torch.cat([out_chunk1, out_chunk2], dim=1)

        # Compare results
        max_diff = torch.max(torch.abs(out_full - out_chunked))
        mean_diff = torch.mean(torch.abs(out_full - out_chunked))
        print(f"Maximum difference between full and chunked: {max_diff:.8f}")
        print(f"Mean difference between full and chunked: {mean_diff:.8f}")
        print(f"Outputs match: {mean_diff < 1e-5}")
        return out_full, out_chunked

    def test_generation(model, name, cache):
        print(f"\nTesting generation with {name}...")
        tokens = prompt.clone()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        cache.zero()

        start_time.record()
        with torch.no_grad():
            tokens = model(tokens, cache)
            pred = tokens

            for i in range(gen_len):
                if isinstance(cache, KVCache):  # Using KVCache
                    pred = model(pred[:, -1:], cache)
                else:  # Using NoKVCache
                    pred = model(tokens, cache)
                    tokens = torch.cat([tokens, pred[:, -1:]], dim=1)

        end_time.record()
        torch.cuda.synchronize()
        elapsed_ms = start_time.elapsed_time(end_time)
        tokens_per_sec = (gen_len * 1000) / elapsed_ms
        print(f"\nTotal Time: {elapsed_ms:.2f}ms ({gen_len/elapsed_ms*1000:.1f} tok/sec)")
        print(f"Time per token: {elapsed_ms/gen_len:.2f}ms")
        print(f"Max throughput: {min(gen_len/elapsed_ms*1000,prompt_len*1000/elapsed_ms):.1f} tok/sec")
        return elapsed_ms, tokens_per_sec

    # Test both models
    for model, name, cache in [(model_cache, "With Cache", kv_cache), (model_nocache, "No Cache", no_cache)]:
        # Warmup
        warmup(model, cache)

        # Benchmark prefill
        benchmark_prefill(model, name, cache)

        # Test chunked generation
        if name == "With Cache": # Only test autoregression with cache addition
            test_chunked_generation(model, name, cache)

        # Test generation
        time_ms, tps = test_generation(model, name, cache)
        print(f"{name}: {time_ms:.2f}ms ({tps:.1f} tokens/sec)")

    # Compare speedup
    print("\nComparing with and without cache speeds.")
    nocache_time, _ = test_generation(model_nocache, "No Cache", no_cache)
    cache_time, _ = test_generation(model_cache, "With Cache", kv_cache)
    speedup = nocache_time / cache_time
    print(f"\nSpeedup from KV cache: {speedup:.2f}x")
