"""DICOM series loading: physical slice ordering, fixed-mm crop and intensity normalization."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field

import cv2
import numpy as np
import pydicom


@dataclass
class SliceHeader:
    name: str
    position: float
    instance_number: int


@dataclass
class SeriesVolume:
    pixels: np.ndarray  # (S, H, W) uint8
    positions: list[float]
    meta: dict = field(default_factory=dict)


def slice_normal(orientation) -> np.ndarray:
    row = np.asarray(orientation[:3], dtype=np.float64)
    col = np.asarray(orientation[3:], dtype=np.float64)
    return np.cross(row, col)


def sort_slices(headers: list[SliceHeader]) -> list[SliceHeader]:
    # Filenames are SOP Instance UIDs and carry no anatomical order, so sort by
    # position along the slice normal, falling back to InstanceNumber on ties.
    return sorted(headers, key=lambda h: (h.position, h.instance_number))


def crop_resize(
    img: np.ndarray, pixel_spacing: tuple[float, float], crop_mm: float, size: int
) -> np.ndarray:
    """Center-crop a square of `crop_mm` (zero-padded if the FOV is smaller) and resize."""
    ps_r, ps_c = pixel_spacing
    half_r = int(round(crop_mm / 2 / ps_r))
    half_c = int(round(crop_mm / 2 / ps_c))
    cr, cc = img.shape[0] // 2, img.shape[1] // 2
    out = np.zeros((2 * half_r, 2 * half_c), dtype=np.float32)
    r0, r1 = max(cr - half_r, 0), min(cr + half_r, img.shape[0])
    c0, c1 = max(cc - half_c, 0), min(cc + half_c, img.shape[1])
    out[r0 - (cr - half_r) : r1 - (cr - half_r), c0 - (cc - half_c) : c1 - (cc - half_c)] = img[
        r0:r1, c0:c1
    ]
    interp = cv2.INTER_AREA if out.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(out, (size, size), interpolation=interp)


def to_uint8(stack: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    # Percentiles on a strided subsample; a full sort of the stack dominates runtime otherwise.
    sample = stack[:, ::4, ::4]
    nonzero = sample[sample > 0]
    if nonzero.size == 0:
        return np.zeros(stack.shape, dtype=np.uint8)
    lo, hi = np.percentile(nonzero, [lo_pct, hi_pct])
    scaled = (stack - lo) / max(hi - lo, 1e-6)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def pick_indices(n: int, max_slices: int) -> np.ndarray:
    if n <= max_slices:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_slices).round().astype(int))


def load_series(
    zf: zipfile.ZipFile,
    names: list[str],
    crop_mm: float,
    size: int,
    max_slices: int,
) -> SeriesVolume:
    first = None
    headers = []
    for name in names:
        # stop_before_pixels on a streamed member only inflates the header bytes.
        with zf.open(name) as f:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        if first is None:
            first = ds
            normal = slice_normal(ds.ImageOrientationPatient)
        pos = float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=np.float64), normal))
        headers.append(SliceHeader(name, pos, int(getattr(ds, "InstanceNumber", 0) or 0)))

    ordered = sort_slices(headers)
    keep = [ordered[i] for i in pick_indices(len(ordered), max_slices)]

    frames, positions, n_bad = [], [], 0
    for h in keep:
        ds = pydicom.dcmread(zf.open(h.name))
        try:
            img = ds.pixel_array.astype(np.float32)
        except ValueError:
            # Two training files have truncated PixelData (discussion #737163).
            n_bad += 1
            continue
        ps = tuple(float(v) for v in ds.PixelSpacing)
        frames.append(crop_resize(img, ps, crop_mm, size))
        positions.append(h.position)

    pixels = to_uint8(np.stack(frames)) if frames else np.zeros((0, size, size), dtype=np.uint8)
    center = np.asarray(first.ImagePositionPatient, dtype=np.float64)
    meta = {
        "n_slices_orig": len(names),
        "n_slices": len(frames),
        "n_bad": n_bad,
        "pixel_spacing": float(first.PixelSpacing[0]),
        "rows": int(first.Rows),
        "cols": int(first.Columns),
        "slice_thickness": float(getattr(first, "SliceThickness", 0) or 0),
        "slice_gap": float(np.median(np.diff(positions))) if len(positions) > 1 else 0.0,
        "series_description": str(getattr(first, "SeriesDescription", "")),
        "manufacturer": str(getattr(first, "Manufacturer", "")),
        "field_strength": float(getattr(first, "MagneticFieldStrength", 0) or 0),
        "patient_sex": str(getattr(first, "PatientSex", "")),
        "laterality": str(
            getattr(first, "Laterality", "") or getattr(first, "ImageLaterality", "")
        ),
        "orientation": [float(v) for v in first.ImageOrientationPatient],
        "first_position": center.tolist(),
    }
    return SeriesVolume(pixels, positions, meta)
