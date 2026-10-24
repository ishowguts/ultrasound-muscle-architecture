"""Build the UMUD Kaggle submission from run_analysis.py outputs on the test set.

Run: .venv/bin/python make_submission.py outputs/test309_run1 --output outputs/submission_v1
Writes comma-separated submission.csv and submission_comma.csv (identical predictions;
columns image_id,pa_deg,fl_mm,mt_mm, as required by the Kaggle uploader), details.csv (where every value came from) and summary.json.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

from measure import fascicle_geometry, thickness_from_lines

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / 'data/raw/umud/test_images_v2/test_set_v2'
# Ranges of the test labels stated on the competition data page.
RANGES = {'pa_deg': (5.0, 45.0), 'fl_mm': (30.0, 200.0), 'mt_mm': (10.0, 50.0)}
MEASURED = ('ok', 'low_confidence')


def test_images(folder=TEST_DIR):
    # Skip hidden files: macOS writes '._<name>' companions on ExFAT/FAT drives.
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in ('.tif', '.tiff', '.png') and not p.name.startswith('.'))


def load_rows(run_dirs, recompute_thickness=False, exact_pa=False, exact_fl=False):
    rows = {}
    for run in run_dirs:  # later runs override earlier ones (e.g. a rerun of failed images)
        folders = {Path(json.loads(m.read_text())['input']).name: m.parent
                   for m in Path(run).glob('*/measurements.json')}
        for row in json.loads((Path(run) / 'run.json').read_text())['results']:
            name = Path(row['input']).name
            # Runs made before measure.py stored the upstream centre thickness; re-measure from saved lines.
            if recompute_thickness and row.get('thickness_px') and 'thickness_upstream_px' not in row:
                lines = json.loads((folders[name] / 'lines.json').read_text())['lines']
                perpendicular = thickness_from_lines((l['label'], l['x'], l['y']) for l in lines)
                row = dict(row, thickness_upstream_px=row['thickness_px'],
                           thickness_px=perpendicular if perpendicular is not None else row['thickness_px'])
            if (exact_pa or exact_fl) and row['status'] in MEASURED:
                lines = json.loads((folders[name] / 'lines.json').read_text())['lines']
                exact = fascicle_geometry((l['label'], l['x'], l['y']) for l in lines)
                if exact:
                    row = dict(row, pennation_upstream_deg=row['pennation_deg'],
                               fascicle_length_upstream_px=row['fascicle_length_px'])
                    if exact_fl:
                        row['fascicle_length_px'] = exact[0]
                    if exact_pa:
                        row['pennation_deg'] = exact[1]
            rows[name] = row
    return rows


def layout(row):
    return f"{row.get('width')}x{row.get('height')}{Path(row['input']).suffix.lower()}"


def build(names, rows, fl_from_mt=False, low_confidence_median=False):
    # Per-layout fallback scale for images whose ruler was not detected (same device and settings family).
    scales = {}
    for row in rows.values():
        if row.get('calibration_status') == 'detected':
            scales.setdefault(layout(row), []).append(row['mm_per_pixel'])
    scales = {k: statistics.median(v) for k, v in scales.items()}
    details = []
    for name in names:
        row = rows.get(name)
        d = {'image_id': name, 'status': row['status'] if row else 'missing', 'scale_source': None, 'mm_per_pixel': None}
        if row and row.get('calibration_status') == 'detected':
            d.update(scale_source='ruler', mm_per_pixel=row['mm_per_pixel'])
        elif row and layout(row) in scales:
            d.update(scale_source='layout_median', mm_per_pixel=scales[layout(row)])
        scale = d['mm_per_pixel']
        measured = row is not None and row['status'] in (('ok',) if low_confidence_median else MEASURED)
        d['pa_deg'] = row['pennation_deg'] if measured else None
        d['fl_mm'] = row['fascicle_length_px'] * scale if measured and scale else None
        d['mt_mm'] = row['thickness_px'] * scale if row and row.get('thickness_px') and scale else None
        details.append(d)
    # A value outside the published label range means the geometry failed (e.g. wrong aponeurosis pair
    # gives MT < 10 mm): treat it as missing rather than clipping it to the boundary.
    for d in details:
        for key, (low, high) in RANGES.items():
            d[key + '_source'] = 'model'
            if d[key] is not None and not low <= d[key] <= high:
                d[key], d[key + '_source'] = None, 'out_of_range'
    # FL missing but MT measured: FL tracks MT/sin(PA), so scale MT by the layout's median FL/MT ratio.
    if fl_from_mt:
        ratios = {}
        for d in details:
            if d['fl_mm_source'] == d['mt_mm_source'] == 'model' and d['fl_mm'] is not None and d['mt_mm'] is not None:
                ratios.setdefault(layout(rows[d['image_id']]), []).append(d['fl_mm'] / d['mt_mm'])
        for d in details:
            group = ratios.get(layout(rows[d['image_id']])) if d['image_id'] in rows else None
            if d['fl_mm'] is None and d['mt_mm'] is not None and d['mt_mm_source'] == 'model' and group:
                estimate = d['mt_mm'] * statistics.median(group)
                if RANGES['fl_mm'][0] <= estimate <= RANGES['fl_mm'][1]:
                    d['fl_mm'], d['fl_mm_source'] = estimate, ('out_of_range+' if d['fl_mm_source'] == 'out_of_range' else '') + 'mt_ratio'
    # Impute what could not be measured with the median of measured values from the same layout, else overall.
    for key in RANGES:
        overall = statistics.median(d[key] for d in details if d[key] is not None)
        by_layout = {}
        for d in details:
            if d[key] is not None and d['image_id'] in rows and d[key + '_source'] == 'model':
                by_layout.setdefault(layout(rows[d['image_id']]), []).append(d[key])
        for d in details:
            if d[key] is None:
                group = by_layout.get(layout(rows[d['image_id']])) if d['image_id'] in rows else None
                prefix = 'out_of_range+' if d[key + '_source'] == 'out_of_range' else ''
                d[key], d[key + '_source'] = ((statistics.median(group), prefix + 'layout_median') if group
                                              else (overall, prefix + 'overall_median'))
    return details


def write(details, out):
    out.mkdir(parents=True, exist_ok=False)
    for name, delimiter in (('submission.csv', ','), ('submission_comma.csv', ',')):
        with (out / name).open('w', newline='') as stream:
            writer = csv.writer(stream, delimiter=delimiter)
            writer.writerow(['image_id', 'pa_deg', 'fl_mm', 'mt_mm'])
            for d in details:
                writer.writerow([d['image_id']] + [f'{d[k]:.3f}' for k in ('pa_deg', 'fl_mm', 'mt_mm')])
    with (out / 'details.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader(); writer.writerows(details)
    count = lambda key, value: sum(d[key] == value for d in details)
    summary = {'images': len(details),
               'status': {s: count('status', s) for s in sorted({d['status'] for d in details})},
               'scale_source': {s: count('scale_source', s) for s in ('ruler', 'layout_median', None)},
               **{key + '_source': {s: count(key + '_source', s) for s in sorted({d[key + '_source'] for d in details})}
                  for key in RANGES}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('runs', nargs='+', type=Path, help='run_analysis.py output directories (later ones win)')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--recompute-thickness', action='store_true',
                        help='re-measure perpendicular thickness from lines.json for runs made before measure.py')
    parser.add_argument('--exact-pa', action='store_true', help='pennation from exact fascicle/deep-aponeurosis angle (lines.json)')
    parser.add_argument('--exact-fl', action='store_true', help='fascicle length from exact aponeurosis intersections (lines.json)')
    parser.add_argument('--low-confidence-median', action='store_true',
                        help='use layout medians for PA/FL of low_confidence rows (1-2 fascicles); MT kept')
    parser.add_argument('--fl-from-mt', action='store_true', help='impute missing FL from measured MT (layout FL/MT ratio)')
    args = parser.parse_args()
    names, rows = test_images(), load_rows(args.runs, args.recompute_thickness, args.exact_pa, args.exact_fl)
    unknown = sorted(set(rows) - set(names))
    if unknown:
        parser.error(f'Rows for non-test images: {unknown[:3]}')
    details = build(names, rows, args.fl_from_mt, args.low_confidence_median)
    assert len(details) == len({d['image_id'] for d in details}) == len(names)  # each test image exactly once
    print(json.dumps(write(details, args.output), indent=2))


if __name__ == '__main__':
    main()
