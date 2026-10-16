"""Run pinned DL_Track_US image inference without launching its GUI."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.local/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.local/cache'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import tensorflow as tf

REVISION = 'b88df46320d822734fed2f1b711ccc2e8300efa1'
CHECKPOINTS = {
    'apo': ('model-apo-VGG16_DiceBCE_512_300.h5', '46281279f5d9b803914b978a7890c5588b1322ab7bb3319c85845a6678172abb'),
    'fasc': ('model-fasc-VGG16_DiceBCE_200_512.h5', 'ba075f0474a2926aa3cf8f5f35993f4d9178c6513a7bbb2b30b334f38bc1521d'),
}
# Published backup settings; linear extrapolation is the implemented image path.
REFERENCE_SETTINGS = {
    'aponeurosis_detection_threshold': 0.2, 'aponeurosis_length_threshold': 400,
    'fascicle_detection_threshold': 0.05, 'fascicle_length_threshold': 40,
    'minimal_muscle_width': 60, 'minimal_pennation_angle': 10,
    'maximal_pennation_angle': 40,
}
# Transcribed from the pinned authors' automated-image-test results.png.
EXPECTED = {
    '1.tif': [175.967, 546.813, 18.4966],
    '2.tif': [162.881, 542.399, 17.9445],
    '3.tif': [283.692, 906.778, 18.6079],
    '4.tif': [196.636, 945.418, 11.8089],
}


def prepare(path, flip):
    if type(flip) is not int or flip not in (0, 1):
        raise ValueError('flip must be 0 or 1')
    original = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f'Unreadable image: {path.name}')
    oriented = np.fliplr(original).copy() if flip else original.copy()
    # Match upstream importAndReshapeImage: BGR channels, float32 then resize.
    batch = resize(oriented.astype(np.float32)[None], (1, 512, 512, 3),
                   mode='constant', preserve_range=True) / 255.0
    return original, oriented, batch


def restore_x(x, width, flip):
    x = np.asarray(x, dtype=float)
    return width - 1 - x if flip else x


def measurement_summary(lengths, angles, thickness, width, height):
    result = {'status': 'invalid_geometry', 'thickness_px': None,
              'fascicle_length_px': None, 'pennation_deg': None}
    # Upstream uses 10000 as an initial distance sentinel. A thickness cannot
    # exceed the image diagonal; never export an uncomputed sentinel as data.
    if thickness is None or not np.isfinite(thickness) or not 0 < thickness <= np.hypot(width, height):
        return result
    result['thickness_px'] = float(thickness)
    if not lengths:
        result['status'] = 'no_fascicles'
        return result
    if (len(lengths) != len(angles) or not np.isfinite(lengths).all()
            or not np.isfinite(angles).all() or min(lengths) <= 0
            or min(angles) < 0 or max(angles) > 90):
        return result
    result.update(status='ok', fascicle_length_px=float(np.median(lengths)),
                  pennation_deg=float(np.median(angles)))
    return result


def load_geometry():
    source = ROOT / 'references/DL_Track_US'
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('Upstream revision differs from the validated revision')
    relative = 'DL_Track_US/gui_helpers/do_calculations.py'
    original = subprocess.check_output(['git', '-C', str(source), 'show', f'{REVISION}:{relative}'])
    if (source / relative).read_bytes() != original:
        raise ValueError('Upstream geometry was modified; review before running')
    spec = importlib.util.spec_from_file_location('dltrack_geometry', source / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.doCalculations


def load_models():
    models = {}
    for key, (name, expected) in CHECKPOINTS.items():
        path = ROOT / 'models' / name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
        model = tf.keras.models.load_model(path, compile=False)
        if model.input_shape != (None, 512, 512, 3) or model.output_shape != (None, 512, 512, 1):
            raise ValueError(f'Unexpected checkpoint shape: {name}')
        models[key] = model
        print(f'Loaded {key}: {model.count_params():,} parameters', flush=True)
    return models


def analyse(item, models, geometry, settings, out):
    path = ROOT / item['path']
    flip = item.get('flip', 0)
    scale = item.get('mm_per_pixel')
    if scale is not None and (type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0):
        raise ValueError('mm_per_pixel must be finite and positive')
    original, oriented, batch = prepare(path, flip)
    height, width = original.shape[:2]
    predictions = {k: m(batch, training=False).numpy() for k, m in models.items()}
    for p in predictions.values():
        if p.shape != (1, 512, 512, 1) or not np.isfinite(p).all():
            raise ValueError('Invalid model prediction')
    out.mkdir()
    np.savez_compressed(out / 'probabilities.npz', **predictions)
    for key, p in predictions.items():
        threshold = settings['aponeurosis_detection_threshold' if key == 'apo' else 'fascicle_detection_threshold']
        mask = cv2.resize((p[0, :, :, 0] > threshold).astype('uint8')*255,
                          (width, height), interpolation=cv2.INTER_NEAREST)
        if flip:
            mask = np.fliplr(mask)
        if not cv2.imwrite(str(out / f'{key}_mask.png'), mask):
            raise OSError('Failed to save mask')
    cached = {k: SimpleNamespace(predict=lambda _, p=p: p) for k, p in predictions.items()}
    result = geometry(batch, oriented, height, width, None, 10,
                      cached['apo'], cached['fasc'], settings, False)
    lengths, angles, _, _, thickness, fig = result
    row = {'input': item['path'], 'flip': flip, 'width': width, 'height': height,
           'length_unit': 'pixels', 'mm_per_pixel': scale, 'fascicle_count': 0,
           'status': 'no_geometry', 'thickness_px': None,
           'fascicle_length_px': None, 'pennation_deg': None,
           'thickness_mm': None, 'fascicle_length_mm': None}
    lines = []
    if fig is not None:
        ax = fig.axes[0]
        ax.images[0].set_data(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        for line in ax.lines:
            x = restore_x(line.get_xdata(), width, flip)
            y = np.asarray(line.get_ydata(), dtype=float)
            line.set_xdata(x)
            lines.append({'label': line.get_label(), 'x': x.tolist(), 'y': y.tolist()})
        ax.set_xlim(0, width); ax.set_ylim(height, 0)
        row['fascicle_count'] = len(lengths)
        row.update(measurement_summary(lengths, angles, thickness, width, height))
        if row['status'] != 'ok':
            for annotation in list(ax.texts):
                annotation.remove()
            ax.text(0.03, 0.07, f"No complete measurement: {row['status']}",
                    transform=ax.transAxes, color='yellow', fontsize=12)
        fig.savefig(out / 'overlay.png', dpi=140)
        plt.close(fig)
    if scale is not None:
        for key in ('thickness', 'fascicle_length'):
            if row[key+'_px'] is not None:
                row[key+'_mm'] = row[key+'_px'] * scale
    (out / 'lines.json').write_text(json.dumps({'coordinate_system': 'original image pixels; top-left origin; extrapolated points may lie outside image', 'lines': lines}, allow_nan=False)+'\n')
    (out / 'measurements.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--examples', action='store_true')
    inputs.add_argument('--manifest', type=Path, help='JSON list of path, flip (0/1), optional verified mm_per_pixel')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite a run')
    parser.add_argument('--profile', choices=['reference', 'current'], default='reference')
    args = parser.parse_args()
    if args.examples:
        flags = [int(x) for x in (ROOT/'data/raw/dltrack_examples/images/flip_flags.txt').read_text().split()]
        if len(flags) != 4:
            parser.error('Expected four example flip flags')
        items = [{'path': f'data/raw/dltrack_examples/images/{i+1}.tif', 'flip': flag} for i, flag in enumerate(flags)]
    else:
        items = json.loads(args.manifest.read_text())
    if not isinstance(items, list) or not items:
        parser.error('Manifest must be a nonempty list')
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str) or not item['path']:
            parser.error('Each manifest entry needs a nonempty path string')
    settings = REFERENCE_SETTINGS.copy()
    if args.profile == 'current':
        settings = json.loads((ROOT/'references/DL_Track_US/DL_Track_US/gui_helpers/gui_files/settings.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    geometry, models = load_geometry(), load_models()
    run = {'upstream_revision': REVISION, 'tensorflow': tf.__version__, 'settings': settings,
           'checkpoints': CHECKPOINTS, 'inputs': items, 'results': []}
    for index, item in enumerate(items):
        start = time.monotonic()
        output = args.output/f'{index+1:02d}_{Path(item["path"]).stem}'
        try:
            row = analyse(item, models, geometry, settings, output)
        except Exception as error:
            plt.close('all')
            row = {'input': item['path'], 'status': 'error', 'error': f'{type(error).__name__}: {error}'}
        row['seconds'] = round(time.monotonic()-start, 3)
        if args.examples and row['status'] == 'ok':
            expected = EXPECTED[Path(item['path']).name]
            row['reference_absolute_errors'] = dict(zip(
                ['thickness_px', 'fascicle_length_px', 'pennation_deg'],
                [abs(row[k]-v) for k,v in zip(['thickness_px','fascicle_length_px','pennation_deg'],expected)]))
        run['results'].append(row)
        (args.output/'run.json').write_text(json.dumps(run, indent=2, allow_nan=False)+'\n')
        print(json.dumps(row), flush=True)
    keys = sorted({k for r in run['results'] for k in r if k != 'reference_absolute_errors'})
    with (args.output/'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(run['results'])
    if not any(r['status']=='ok' for r in run['results']):
        raise SystemExit('No successful measurements; see run.json')


if __name__ == '__main__':
    main()
