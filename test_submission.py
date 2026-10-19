"""Checks for make_submission.py: .venv/bin/python test_submission.py (no weights needed)."""
import csv
import json
from pathlib import Path
import tempfile

import make_submission as ms


def row(name, status, scale_status='detected', scale=0.1, px=(300, 500), pa=20.0, size=(1200, 800)):
    return {'input': f'data/raw/umud/test_images_v2/test_set_v2/{name}', 'status': status,
            'width': size[0], 'height': size[1], 'calibration_status': scale_status,
            'mm_per_pixel': scale if scale_status == 'detected' else None,
            'thickness_px': px[0] if status != 'invalid_geometry' else None,
            'fascicle_length_px': px[1] if status in ms.MEASURED else None,
            'pennation_deg': pa if status in ms.MEASURED else None}


rows = {r['input'].split('/')[-1]: r for r in [
    row('A.tif', 'ok'),                                        # 30 mm, 50 mm, 20 deg
    row('B.tif', 'ok', px=(200, 700), pa=30.0),                # 20 mm, 70 mm
    row('C.tif', 'no_fascicles', scale_status='not_detected'), # layout median scale 0.1 -> MT 30, PA/FL imputed
    row('D.tif', 'invalid_geometry'),                          # everything imputed
    row('E.tif', 'ok', px=(10, 5000), pa=60.0),                # all out of range -> imputed, not clipped
]}
names = sorted(rows) + ['F.png']                               # F missing from the run
details = {d['image_id']: d for d in ms.build(names, rows)}
assert details['A.tif']['mt_mm'] == 30 and details['A.tif']['fl_mm'] == 50 and details['A.tif']['pa_deg'] == 20
assert details['C.tif']['scale_source'] == 'layout_median' and details['C.tif']['mt_mm'] == 30
assert details['C.tif']['pa_deg_source'] == 'layout_median' and details['C.tif']['pa_deg'] == 25.0  # E excluded
assert details['D.tif']['mt_mm_source'] == 'layout_median'
assert details['E.tif']['mt_mm_source'] == details['E.tif']['pa_deg_source'] == 'out_of_range+layout_median'
assert details['E.tif']['pa_deg'] == 25.0 and details['E.tif']['mt_mm'] == 30.0  # medians of valid A, B (and C for MT)
assert details['F.png']['status'] == 'missing' and details['F.png']['pa_deg_source'] == 'overall_median'
with tempfile.TemporaryDirectory() as directory:
    summary = ms.write(list(details.values()), Path(directory) / 'sub')
    text = (Path(directory) / 'sub/submission.csv').read_text().splitlines()
    assert text[0] == 'image_id,pa_deg,fl_mm,mt_mm' and text[1] == 'A.tif,20.000,50.000,30.000' and len(text) == 7
    assert (Path(directory) / 'sub/submission_comma.csv').read_text().splitlines()[0] == 'image_id,pa_deg,fl_mm,mt_mm'
    assert list(csv.DictReader(text))[0]['image_id'] == 'A.tif'
    assert summary['images'] == 6 and summary['scale_source']['layout_median'] == 1
# FL imputed from MT x layout median FL/MT ratio (A: 50/30, B: 70/20 -> median ratio 2.5833); C has MT 30.
geo = {d['image_id']: d for d in ms.build(names, rows, fl_from_mt=True)}
assert geo['C.tif']['fl_mm_source'] == 'mt_ratio' and abs(geo['C.tif']['fl_mm'] - 30 * (50 / 30 + 70 / 20) / 2) < 1e-9
assert geo['A.tif']['fl_mm_source'] == 'model' and geo['D.tif']['fl_mm_source'] == 'layout_median'
# Re-measured thickness from saved lines replaces the upstream value and keeps it for reference.
with tempfile.TemporaryDirectory() as directory:
    run = Path(directory) / 'run'; (run / '01_A').mkdir(parents=True)
    r = row('A.tif', 'ok'); (run / 'run.json').write_text(json.dumps({'results': [r]}))
    (run / '01_A/measurements.json').write_text(json.dumps(r))
    (run / '01_A/lines.json').write_text(json.dumps({'lines': [
        {'label': 'Upper Aponeurosis', 'x': [0, 400], 'y': [10, 10]},
        {'label': 'Lower Aponeurosis', 'x': [100, 400], 'y': [260, 260]}]}))
    again = ms.load_rows([run], recompute_thickness=True)['A.tif']
    assert again['thickness_px'] == 250 and again['thickness_upstream_px'] == 300
    assert ms.load_rows([run])['A.tif']['thickness_px'] == 300
# Exact fascicle geometry: deep aponeurosis tilted 2 deg the same way as a 12 deg fascicle -> PA 10 deg.
from measure import fascicle_geometry
import numpy as np
xs = np.array([0.0, 1000.0]); t = lambda deg: np.tan(np.radians(deg))
lines = [('Upper Aponeurosis', xs, 100 - xs * t(2)), ('Lower Aponeurosis', xs, 400 - xs * t(2)),
         ('Fascicle 0', xs, 600 - xs * t(12))]
fl, pa, count = fascicle_geometry(lines)
assert abs(pa - 10) < 1e-9 and count == 1, pa
assert abs(fl - 300 * np.cos(np.radians(2)) / np.sin(np.radians(10))) < 1e-6, fl  # thickness / sin(PA)
assert fascicle_geometry(lines[:2]) is None
rows_lc = dict(rows); rows_lc['G.tif'] = row('G.tif', 'low_confidence', pa=40.0)
lc = {d['image_id']: d for d in ms.build(sorted(rows_lc), rows_lc, low_confidence_median=True)}
assert lc['G.tif']['pa_deg_source'] == 'layout_median' and lc['G.tif']['mt_mm_source'] == 'model'
print('Passed: ruler and fallback scale, measured/imputed/out-of-range values, missing images, comma-delimited exports, FL from MT, thickness re-measure, exact fascicle geometry, low-confidence medians.')
