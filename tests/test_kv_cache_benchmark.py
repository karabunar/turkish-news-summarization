import pytest

torch = pytest.importorskip("torch")

from scripts.kv_cache_benchmark import cache_nbytes, kv_bytes_per_token


def test_qwen25_1p5b_kv_bytes_per_token():
    # 28 layers, 2 KV heads, head_dim 128, bf16 -> 2*28*2*128*2 = 28672 bytes
    assert kv_bytes_per_token(28, 2, 128, 2) == 28672


def test_cache_nbytes_counts_k_and_v_of_all_layers():
    layer = (torch.zeros(1, 2, 10, 128, dtype=torch.bfloat16), torch.zeros(1, 2, 10, 128, dtype=torch.bfloat16))
    assert cache_nbytes((layer, layer)) == 2 * 2 * (1 * 2 * 10 * 128 * 2)
