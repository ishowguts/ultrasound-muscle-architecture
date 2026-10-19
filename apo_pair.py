"""Fascicle-guided aponeurosis pair (fallback when upstream picks the wrong layers).

Upstream takes the two topmost aponeurosis-like contours. On scans with subcutaneous fascia that is
often skin/fat + superficial aponeurosis, leaving no fascicles between them. Here bright aponeurosis
bands are grouped by depth and, among vertically adjacent band pairs, the one enclosing the most
fascicle-probability pixels (the muscle belly) is kept; all other bands are removed from the
aponeurosis map so the unchanged upstream geometry measures that pair.
"""
import cv2
import numpy as np


def bands(apo, threshold=0.2, min_extent=0.25, merge_px=12):
    mask = (apo > threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    width = apo.shape[1]
    parts = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_WIDTH] >= 0.08 * width:
            ys, xs = np.nonzero(labels == i)
            parts.append({'ids': [i], 'y': float(np.median(ys)), 'x0': int(xs.min()), 'x1': int(xs.max())})
    parts.sort(key=lambda p: p['y'])
    merged = []
    for p in parts:  # pieces of one aponeurosis at about the same depth
        if merged and abs(p['y'] - merged[-1]['y']) <= merge_px:
            m = merged[-1]
            m['ids'] += p['ids']; m['x0'] = min(m['x0'], p['x0']); m['x1'] = max(m['x1'], p['x1'])
        else:
            merged.append(p)
    return [m for m in merged if m['x1'] - m['x0'] >= min_extent * width], labels


def guided_apo(apo, fasc, fasc_threshold=0.05, min_gap_px=10):
    """Return (modified aponeurosis map, chosen band indices) or (None, None) if fewer than 2 bands."""
    apo2d, fasc2d = apo[0, :, :, 0], fasc[0, :, :, 0]
    found, labels = bands(apo2d)
    if len(found) < 2:
        return None, None
    best, best_score = None, -1
    for i in range(len(found) - 1):
        top, bottom = found[i], found[i + 1]
        if bottom['y'] - top['y'] < min_gap_px:
            continue
        top_mask = np.isin(labels, top['ids']); bottom_mask = np.isin(labels, bottom['ids'])
        score = 0
        for x in range(max(top['x0'], bottom['x0']), min(top['x1'], bottom['x1']) + 1):
            ty = np.nonzero(top_mask[:, x])[0]; by = np.nonzero(bottom_mask[:, x])[0]
            if len(ty) and len(by) and by.min() > ty.max():
                score += int((fasc2d[ty.max() + 1:by.min(), x] > fasc_threshold).sum())
        if score > best_score:
            best, best_score = (i, i + 1), score
    if best is None or best_score <= 0:
        return None, None
    keep = np.isin(labels, found[best[0]]['ids'] + found[best[1]]['ids'])
    out = apo.copy()
    out[0, :, :, 0] = np.where(keep, apo2d, 0)
    return out, best
