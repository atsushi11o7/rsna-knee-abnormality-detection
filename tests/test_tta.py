import torch

from rsna_knee.tta import mirror_batch


def test_mirror_flips_slices_for_sagittal_and_width_otherwise():
    images = torch.arange(2 * 3 * 2 * 2, dtype=torch.uint8).view(1, 2, 3, 2, 2)
    batch = {
        "images": images,
        "plane": torch.tensor([[0, 1]]),  # Sagittal, Coronal
        "rel_pos": torch.tensor([[0.2, 0.2]]),
    }
    out = mirror_batch(batch)
    assert torch.equal(out["images"][0, 0], images[0, 0].flip(0))
    assert torch.equal(out["images"][0, 1], images[0, 1].flip(-1))
    assert torch.allclose(out["rel_pos"], torch.tensor([[0.8, 0.2]]))
    assert torch.equal(mirror_batch(out)["images"], images)
