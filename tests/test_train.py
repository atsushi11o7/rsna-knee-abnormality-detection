import torch

from rsna_knee.train import cosine_with_warmup, jitter_intensity, weighted_bce


def test_jitter_keeps_dtype_and_range():
    x = torch.randint(0, 256, (2, 4, 3, 8, 8), dtype=torch.uint8)
    out = jitter_intensity(x, 0.2)
    assert out.dtype == torch.uint8 and out.shape == x.shape
    assert torch.equal(jitter_intensity(x, 0.0), x)


def test_weighted_bce_ignores_zero_weight_cells():
    logits = torch.tensor([[10.0, -10.0]])
    target = torch.tensor([[1.0, 1.0]])
    weight = torch.tensor([[1.0, 0.0]])
    assert weighted_bce(logits, target, weight) < 1e-3


def test_cosine_with_warmup_shape():
    f = cosine_with_warmup(total=100, warmup=10)
    assert f(0) == 0.1 and f(9) == 1.0 and abs(f(99)) < 1e-2
