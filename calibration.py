"""Calibration in original-image pixels; never derive physical scale from TIFF DPI.

Use manual_calibration for a known vertical distance. bar_calibration only proposes
an upstream detection; its mm_per_pixel remains None until the caller explicitly
confirms both ruler association and physical tick spacing. Both methods assume
isotropic pixels when their scale is applied to nonvertical lengths.
"""
from functools import lru_cache
import importlib.util
from numbers import Real
from pathlib import Path
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REVISION = 'b88df46320d822734fed2f1b711ccc2e8300efa1'


def positive_distance(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError('Known distance must be a finite positive number in millimetres')
    return float(value)


def manual_calibration(points, distance_mm, image_shape):
    """Two original-image (x,y) points on a vertical ruler plus known distance.

    Uses abs(y2-y1), not Euclidean distance; small horizontal clicking offsets
    must not alter a vertical ruler's scale. image_shape is (height, width, ...).
    This records user-provided calibration, not independent anatomical validation.
    """
    distance_mm = positive_distance(distance_mm)
    points = np.asarray(points)
    if points.shape != (2, 2) or points.dtype.kind not in 'iuf' or not np.isfinite(points).all():
        raise ValueError('Provide two finite numeric (x,y) points')
    if len(image_shape) < 2 or any(type(v) is not int or v <= 0 for v in image_shape[:2]):
        raise ValueError('image_shape needs positive integer height and width')
    height, width = image_shape[:2]
    if (points < 0).any() or (points[:, 0] > width-1).any() or (points[:, 1] > height-1).any():
        raise ValueError('Calibration points must be inside the original image')
    pixels = abs(float(points[1, 1])-float(points[0, 1]))
    if pixels == 0:
        raise ValueError('Calibration points must have different vertical coordinates')
    return {'method': 'manual_vertical', 'status': 'provided',
            'mm_per_pixel': positive_distance(distance_mm/pixels), 'distance_mm': distance_mm,
            'distance_px': pixels, 'points': points.astype(float).tolist(),
            'coordinate_system': 'original image pixels', 'isotropic_assumption': True}


@lru_cache(maxsize=1)
def load_bar_detector():
    source = ROOT / 'references/DL_Track_US'
    relative = 'DL_Track_US/gui_helpers/calibrate.py'
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('Upstream revision differs from the validated revision')
    original = subprocess.check_output(['git', '-C', str(source), 'show', f'{REVISION}:{relative}'])
    if (source / relative).read_bytes() != original:
        raise ValueError('Upstream calibration source was modified')
    spec = importlib.util.spec_from_file_location('dltrack_calibration', source / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.calibrateDistanceStatic


def bar_calibration(image, spacing_mm=None, *, confirmed=False):
    """Detect a candidate in the ORIGINAL grayscale/BGR/BGRA uint8 image.

    spacing_mm is the known physical distance between the detected ruler marks;
    it is never inferred. confirmed=True is a caller assertion after inspecting
    the actual ruler and spacing. Unconfirmed output is unusable for conversion.
    """
    if type(confirmed) is not bool:
        raise ValueError('confirmed must be boolean')
    if spacing_mm is not None:
        spacing_mm = positive_distance(spacing_mm)
    if confirmed and spacing_mm is None:
        raise ValueError('Confirmation requires known ruler spacing in millimetres')
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError('Expected a uint8 image')
    if image.ndim == 3 and image.shape[2] in (3, 4):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    if image.ndim != 2 or image.shape[0] < 2 or image.shape[1] < 7:
        raise ValueError('Expected a nonempty grayscale, BGR or BGRA image')
    height, width = image.shape
    box = [width-int(width*0.15), int(height*0.4), width, height]
    # ponytail: reuse the upstream maximum-bright-gap heuristic; it can detect
    # text or miss rulers outside this ROI. Require confirmation; upgrade to a
    # validated tick/label detector only if manual calibration becomes a burden.
    pixels, _ = load_bar_detector()(image, spacing_mm)
    result = {'method': 'upstream_bar', 'status': 'not_detected',
              'mm_per_pixel': None, 'candidate_mm_per_pixel': None,
              'distance_px': None, 'distance_mm': spacing_mm, 'confirmed': False,
              'search_box': box, 'coordinate_system': 'original image pixels',
              'isotropic_assumption': True, 'upstream_revision': REVISION}
    if pixels is None or not np.isfinite(pixels) or not 0 < pixels < height:
        return result
    result.update(status='candidate', distance_px=float(pixels))
    if spacing_mm is not None:
        result['candidate_mm_per_pixel'] = positive_distance(spacing_mm/float(pixels))
    if confirmed:
        result.update(status='confirmed', confirmed=True,
                      mm_per_pixel=result['candidate_mm_per_pixel'])
    return result


def from_spec(spec, image):
    """Manifest/app calibration dict -> record; mm_per_pixel stays None unless usable.

    {"method": "manual", "points": [[x, y], [x, y]], "distance_mm": 10}
    {"method": "bar", "spacing_mm": 5, "confirmed": true}
    {"method": "auto"}  (automatic ruler detection for the UMUD test-set layouts)
    Always pass the ORIGINAL full image, before any inference crop.
    """
    if not isinstance(spec, dict):
        raise ValueError('calibration must be an object')
    if spec.get('method') == 'manual':
        return manual_calibration(spec.get('points'), spec.get('distance_mm'), image.shape)
    if spec.get('method') == 'auto':
        return auto_ruler(image)
    if spec.get('method') == 'bar':
        return bar_calibration(image, spec.get('spacing_mm'), confirmed=spec.get('confirmed', False))
    raise ValueError('calibration method must be "manual", "bar" or "auto"')


# Ruler layouts of the UMUD test-set devices (host answers in Kaggle discussions 689019 and 748126:
# vertical scale bars only, square pixels, bottom ticks 1 cm apart where no vertical bar exists).
# Each entry: name, axis ('v' = vertical ruler in a column, 'h' = ticks along a row), fraction range of
# columns/rows to search, accepted tick spacing in px, and the physical tick spacing in mm.
LAYOUTS = {
    (1200, 800): [('telemed', 'v', (40 / 1200, 120 / 1200), (120, 180), 10.0),
                  ('juniper', 'v', (1100 / 1200, 1197 / 1200), (20, 110), None),  # 2 or 5 mm, see below
                  ('lumify', 'v', (0, 40 / 1200), (50, 110), 5.0)],
    (1088, 644): [('telemed', 'v', (0, 40 / 1088), (100, 160), 10.0)],
}
BOTTOM_TICKS = ('bottom_ticks', 'h', (0.85, 0.997), (40, 200), 10.0)


def _tick_chain(gray, axis, lines, spacing_range, threshold):
    """Longest run of evenly spaced short bright marks along one column (v) or row (h)."""
    best = None
    for line in lines:
        band = gray[:, line:line + 3].max(axis=1) if axis == 'v' else gray[line:line + 3, :].max(axis=0)
        idx = np.flatnonzero(band > threshold)
        if len(idx) < 4:
            continue
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        centres = np.array([g.mean() for g in groups if g[-1] - g[0] <= 6])  # ticks are thin
        if len(centres) < 4:
            continue
        gaps = np.diff(centres)
        for spacing in np.unique(np.round(gaps)):
            if not spacing_range[0] <= spacing <= spacing_range[1]:
                continue
            regular = np.abs(gaps - spacing) < 0.08 * spacing
            run = start = longest = first = 0
            for i, ok in enumerate(regular):
                run = run + 1 if ok else 0
                if not ok:
                    start = i + 1
                if run > longest:
                    longest, first = run, start
            if longest >= 3 and (best is None or longest > best['intervals']):
                chain = centres[first:first + longest + 1]
                best = {'intervals': int(longest), 'line': int(line), 'first_px': float(chain[0]),
                        'last_px': float(chain[-1]), 'tick_px': float((chain[-1] - chain[0]) / longest)}
    return best


def auto_ruler(image):
    """Find the on-screen depth ruler automatically; mm_per_pixel is None when not found.

    Works on the ORIGINAL full image. Layouts are those of the UMUD test devices; other devices
    return not_detected and need manual two-point calibration.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError('Expected a uint8 image')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape
    result = {'method': 'auto_ruler', 'status': 'not_detected', 'mm_per_pixel': None, 'layout': None,
              'coordinate_system': 'original image pixels', 'isotropic_assumption': True}
    for name, axis, (lo, hi), spacing_range, tick_mm in LAYOUTS.get((width, height), [BOTTOM_TICKS]):
        size = width if axis == 'v' else height
        lines = range(int(lo * size), max(int(lo * size) + 1, int(hi * size) - 2))
        for threshold in (110, 60):  # some rulers are drawn dimmer
            chain = _tick_chain(gray, axis, lines, spacing_range, threshold)
            if chain is None or chain['intervals'] < (3 if axis == 'h' else 4):
                continue
            # Juniper draws 2 mm ticks at shallow depths (3 cm) and 5 mm ticks otherwise; 5 mm ticks
            # are >= 40 px apart for any depth up to 7.5 cm, beyond the 10-50 mm muscle range.
            mm = tick_mm if tick_mm else (2.0 if chain['tick_px'] < 40 else 5.0)
            scale = mm / chain['tick_px']
            if 1 / 30 <= scale <= 1 / 5:  # 5-30 px/mm covers all test devices and depths
                result.update(chain, status='detected', mm_per_pixel=scale, tick_mm=mm, layout=name,
                              axis=axis, threshold=threshold)
                return result
    return result
