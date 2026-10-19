"""Muscle thickness as the challenge experts measure it (data page: perpendicular distance between the
superficial and deep aponeuroses at three locations over the muscle width, averaged).

Replaces the pinned upstream centre thickness, whose search loop indexes the lower-aponeurosis arrays
with x-coordinates (do_calculations.py: `for val in range(A1, A2): low_x[val]`), so the searched lower
points are shifted by the x where that aponeurosis starts and the minimum is often not perpendicular.
"""
import numpy as np

FRACTIONS = (0.25, 0.5, 0.75)


def point_to_polyline(x, y, xs, ys):
    a = np.c_[xs[:-1], ys[:-1]]
    ab = np.c_[xs[1:], ys[1:]] - a
    t = np.clip(((np.array([x, y]) - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    nearest = a + ab * t[:, None]
    return float(np.min(np.hypot(nearest[:, 0] - x, nearest[:, 1] - y)))


def perpendicular_thickness(upper_x, upper_y, lower_x, lower_y, fractions=FRACTIONS):
    """Mean distance from points on the upper inner edge to the lower inner edge; None without overlap."""
    ux, uy, lx, ly = (np.asarray(v, dtype=float) for v in (upper_x, upper_y, lower_x, lower_y))
    if min(len(ux), len(lx)) < 2 or not all(np.isfinite(v).all() for v in (ux, uy, lx, ly)):
        return None
    order = np.argsort(ux); ux, uy = ux[order], uy[order]
    order = np.argsort(lx); lx, ly = lx[order], ly[order]
    low, high = max(ux[0], lx[0]), min(ux[-1], lx[-1])
    if high <= low:
        return None
    xs = [low + f * (high - low) for f in fractions]
    return float(np.mean([point_to_polyline(x, np.interp(x, ux, uy), lx, ly) for x in xs]))


def thickness_from_lines(lines):
    """lines: iterable of (label, x, y) as drawn by the upstream figure or saved in lines.json."""
    found = {label: (x, y) for label, x, y in lines}
    if 'Upper Aponeurosis' not in found or 'Lower Aponeurosis' not in found:
        return None
    return perpendicular_thickness(*found['Upper Aponeurosis'], *found['Lower Aponeurosis'])


def fascicle_geometry(lines):
    """Exact per-fascicle geometry from the upstream drawn lines: median (FL px, PA deg, count) or None.

    Upstream computes the deep-aponeurosis angle as 90 + abs(int(angle)), dropping its fraction and sign
    (do_calculations.py), so PA is biased whenever that aponeurosis is tilted, and it truncates fascicles at
    a fixed sampling grid. Here each fascicle line (recovered exactly from its drawn points) is intersected
    with the linear fits of both aponeurosis edges, as upstream intends: FL = distance between the two
    intersections (linear extrapolation), PA = angle between fascicle and deep aponeurosis.
    """
    lines = list(lines)
    found = {label: (np.asarray(x, float), np.asarray(y, float)) for label, x, y in lines}
    if 'Upper Aponeurosis' not in found or 'Lower Aponeurosis' not in found:
        return None
    upper = np.polyfit(*found['Upper Aponeurosis'], 1)
    lower = np.polyfit(*found['Lower Aponeurosis'], 1)
    lengths, angles = [], []
    for label, x, y in lines:
        x, y = np.asarray(x, float), np.asarray(y, float)
        if not str(label).startswith('Fascicle') or len(x) < 2 or np.ptp(x) == 0:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        if abs(slope - upper[0]) < 1e-9 or abs(slope - lower[0]) < 1e-9:
            continue
        xu = (upper[1] - intercept) / (slope - upper[0])
        xl = (lower[1] - intercept) / (slope - lower[0])
        lengths.append(float(np.hypot(xu - xl, slope * (xu - xl))))
        angles.append(float(abs(np.degrees(np.arctan(slope) - np.arctan(lower[0])))))
    if not lengths:
        return None
    return float(np.median(lengths)), float(np.median(angles)), len(lengths)
