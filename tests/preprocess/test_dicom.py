import numpy as np

from rsna_knee.preprocess.dicom import SliceHeader, crop_resize, pick_indices, sort_slices, to_uint8


def test_sort_slices_uses_position_not_name():
    hs = [SliceHeader("c", 2.0, 1), SliceHeader("a", -1.0, 3), SliceHeader("b", 0.5, 2)]
    assert [h.name for h in sort_slices(hs)] == ["a", "b", "c"]


def test_crop_resize_keeps_physical_scale():
    img = np.ones((400, 400), np.float32)
    # 0.5 mm/px over 400 px = 200 mm FOV; a 100 mm crop is the central 200 px.
    out = crop_resize(img, (0.5, 0.5), crop_mm=100, size=64)
    assert out.shape == (64, 64)
    assert np.allclose(out, 1.0)


def test_crop_resize_pads_when_fov_is_small():
    img = np.ones((100, 100), np.float32)
    # 1 mm/px over 100 px = 100 mm FOV; a 200 mm crop leaves a zero border.
    out = crop_resize(img, (1.0, 1.0), crop_mm=200, size=200)
    assert out[0, 0] == 0 and out[100, 100] == 1


def test_pick_indices_caps_and_spans():
    idx = pick_indices(320, 64)
    assert len(idx) == 64 and idx[0] == 0 and idx[-1] == 319
    assert list(pick_indices(10, 64)) == list(range(10))


def test_to_uint8_range():
    out = to_uint8(np.random.rand(4, 32, 32).astype(np.float32) * 1000)
    assert out.dtype == np.uint8 and out.max() == 255


def test_pick_indices_zero_keeps_all():
    assert list(pick_indices(320, 0)) == list(range(320))
