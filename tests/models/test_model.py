import pytest
import torch

from rsna_knee.models.model import KneeModel

# 事前学習の重みを落とさず、小さい入力で形とマスクの挙動だけを確かめる
ENCODER = {"pretrained": False, "img_size": 56, "out_dim": 64}
AGGREGATOR = {"n_layers": 1, "n_heads": 4, "dropout": 0.0}


def _batch(k_valid=(5, 3), size=64):
    b, k = len(k_valid), max(k_valid)
    mask = torch.zeros(b, k, dtype=torch.bool)
    for i, n in enumerate(k_valid):
        mask[i, :n] = True
    return {
        "images": torch.randint(0, 256, (b, k, 3, size, size), dtype=torch.uint8),
        "plane": torch.randint(0, 3, (b, k)),
        "fat_sup": torch.randint(0, 2, (b, k)),
        "rel_pos": torch.rand(b, k),
        "mask": mask,
    }


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return KneeModel(ENCODER, AGGREGATOR).eval()


def test_output_shape(model):
    assert model(_batch()).shape == (2, 12)


def test_padding_does_not_change_output(model):
    batch = _batch(k_valid=(3,))
    with torch.no_grad():
        ref = model(batch)
        padded = {
            key: torch.cat([v, torch.zeros_like(v[:, :4]) if key == "mask" else v[:, :4]], 1)
            for key, v in batch.items()
        }
        padded["images"][:, 3:] = 255  # パディング部分の中身が何でも結果は同じはず
        out = model(padded)
    assert torch.allclose(ref, out, atol=1e-5)


def test_param_groups_cover_every_parameter_once(model):
    groups = model.param_groups(lr=1e-3, encoder_lr=1e-5, weight_decay=0.05)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    enc = {id(p) for p in model.encoder.backbone.parameters()}
    assert all(g["lr"] == 1e-5 for g in groups if any(id(p) in enc for p in g["params"]))


@pytest.mark.parametrize("agg", ["transformer", "mil"])
def test_both_aggregators_ignore_padding(agg):
    torch.manual_seed(0)
    cfg = {"type": agg, "dropout": 0.0} | ({} if agg == "mil" else {"n_layers": 1, "n_heads": 4})
    m = KneeModel(ENCODER, cfg).eval()
    batch = _batch(k_valid=(3,))
    with torch.no_grad():
        ref = m(batch)
        padded = {
            key: torch.cat([v, torch.zeros_like(v[:, :2]) if key == "mask" else v[:, :2]], 1)
            for key, v in batch.items()
        }
        out = m(padded)
    assert out.shape == (1, 12) and torch.allclose(ref, out, atol=1e-5)
