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
from calibration import from_spec
from measure import thickness_from_lines
from apo_pair import guided_apo

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
# Adapter-only rules, kept outside upstream settings (D015, D016, D018).
MIN_FASCICLES = 3
STATUS_RANK = {'ok': 3, 'low_confidence': 2, 'no_fascicles': 1, 'error': -1}


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Unreadable image: {path.name}')
    return image


def to_batch(image, flip):
    if type(flip) is not int or flip not in (0, 1):
        raise ValueError('flip must be 0 or 1')
    oriented = np.fliplr(image).copy() if flip else image.copy()
    # Match upstream importAndReshapeImage: BGR channels, float32 then resize.
    batch = resize(oriented.astype(np.float32)[None], (1, 512, 512, 3),
                   mode='constant', preserve_range=True) / 255.0
    return oriented, batch


def prepare(path, flip):
    if type(flip) is not int or flip not in (0, 1):
        raise ValueError('flip must be 0 or 1')
    original = read_image(path)
    return (original, *to_batch(original, flip))


def detect_scan_field(image):
    """Box (x0, y0, x1, y1) of the largest non-black region, else the full image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    full = (0, 0, width, height)
    # Blurring merges speckle into one region; thin UI text on black stays dim.
    # shortcut: grey (non-black) UI panels are not removed, pass a manual crop for those.
    k = max(15, (min(height, width)//25) | 1)
    mask = (cv2.blur(gray, (k, k)) > 18).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return full
    x, y, w, h, _ = stats[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    # Crop only letterboxed frames: trimming thin slivers shifted author-example results (D024).
    if w < 64 or h < 64 or not 0.1*width*height <= w*h < 0.75*width*height:
        return full
    return (int(x), int(y), int(x+w), int(y+h))


def resolve_crop(crop, image):
    height, width = image.shape[:2]
    if crop == 'auto':
        return detect_scan_field(image)
    if crop == 'none':
        return (0, 0, width, height)
    if (not isinstance(crop, list) or len(crop) != 4 or any(type(v) is not int for v in crop)
            or not (0 <= crop[0] < crop[2] <= width and 0 <= crop[1] < crop[3] <= height)):
        raise ValueError('crop must be "auto", "none" or [x0, y0, x1, y1] inside the image')
    return tuple(crop)


def choose(summaries):
    """Best orientation: usable status, then more fascicles; ties keep the first (flip 0)."""
    return max(range(len(summaries)), key=lambda i: (
        STATUS_RANK.get(summaries[i]['status'], 0), summaries[i]['fascicle_count'], -i))


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
    result.update(status='ok' if len(lengths) >= MIN_FASCICLES else 'low_confidence',
                  fascicle_length_px=float(np.median(lengths)),
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


def load_weights_any(model, path):
    """Load weights saved by Keras 2.10 (legacy h5) or by tf_keras>=2.16 (`.weights.h5`: layers/<name>/vars/<i>)."""
    import h5py
    with h5py.File(path, 'r') as f:
        if 'layers' not in f:
            model.load_weights(str(path))
            return
        groups = f['layers']
        # The new format names layers generically by type and order (conv2d, conv2d_1, ...), not by layer.name.
        import re
        seen, used = {}, set()
        for layer in model.layers:
            base = re.sub(r'([a-z])([A-Z])', r'\1_\2', re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', type(layer).__name__)).lower()  # Keras' to_snake_case
            key = base if base not in seen else f'{base}_{seen[base]}'
            seen[base] = seen.get(base, 0) + 1
            if not layer.weights:
                continue
            saved = groups[key]['vars']
            values = [saved[str(i)][()] for i in range(len(saved))]
            if [v.shape for v in values] != [tuple(w.shape) for w in layer.weights]:
                raise ValueError(f'Weight shape mismatch: {layer.name} vs {key} ({path})')
            layer.set_weights(values)
            used.add(key)
        unused = [k for k in groups if len(groups[k]['vars']) and k not in used]
        if unused:
            raise ValueError(f'Saved weights not loaded: {unused[:5]} ({path})')


def load_models(weights_dir=None):
    """Official checkpoints (hash-checked); optionally overlay fine-tuned weights-only files from weights_dir."""
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
        tuned = Path(weights_dir) / f'{key}_finetuned.weights.h5' if weights_dir else None
        if tuned and tuned.exists():  # finetune.py saves only when validation Dice beat the checkpoint
            load_weights_any(model, tuned)
            print(f'Fine-tuned {key} weights: {tuned.name} sha256 {hashlib.sha256(tuned.read_bytes()).hexdigest()[:16]}', flush=True)
        models[key] = model
        print(f'Loaded {key}: {model.count_params():,} parameters', flush=True)
    return models


def measure_geometry(batch, oriented, apo, fasc, geometry, settings):
    height, width = oriented.shape[:2]
    cached = {'apo': SimpleNamespace(predict=lambda _: apo), 'fasc': SimpleNamespace(predict=lambda _: fasc)}
    lengths, angles, _, _, thickness, fig = geometry(
        batch, oriented, height, width, None, 10, cached['apo'], cached['fasc'], settings, False)
    summary = {'fascicle_count': 0, 'status': 'no_geometry', 'thickness_px': None,
               'fascicle_length_px': None, 'pennation_deg': None, 'thickness_upstream_px': None}
    if fig is not None:
        summary['fascicle_count'] = len(lengths)
        upstream = measurement_summary([], [], thickness, width, height)['thickness_px']
        if upstream is not None:  # only re-measure where upstream found a valid aponeurosis pair
            summary['thickness_upstream_px'] = upstream
            perpendicular = thickness_from_lines(
                (l.get_label(), l.get_xdata(), l.get_ydata()) for l in fig.axes[0].lines)
            thickness = perpendicular if perpendicular is not None else upstream
        summary.update(measurement_summary(lengths, angles, thickness, width, height))
    return summary, fig


def thickness_ok(summary, min_thickness_px):
    t = summary.get('thickness_px')
    return t is not None and (min_thickness_px is None or t >= min_thickness_px)


def pick(upstream, guided, min_thickness_px):
    """0 keeps the upstream result, 1 the fascicle-guided one: status, plausible thickness, fascicles; ties upstream."""
    key = lambda s: (STATUS_RANK.get(s['status'], 0), thickness_ok(s, min_thickness_px), s['fascicle_count'])
    return 1 if key(guided) > key(upstream) else 0


def ifss_fascicles(oriented, ifss):
    """Fascicle map from the video IFSS model: one frame stacked x3, middle frame, channel 0 (as the upstream video routine)."""
    gray = cv2.resize(cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY), (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    stack = np.repeat(gray[None, None, :, :, None], 3, axis=1)
    return tf.clip_by_value(ifss([stack], training=False), 0, 1).numpy()[:, 1, :, :, :1]


def infer(region, flip, models, geometry, settings, min_thickness_px=None, ifss=None):
    oriented, batch = to_batch(region, flip)
    predictions = {k: m(batch, training=False).numpy() for k, m in models.items() if not (ifss is not None and k == 'fasc')}
    if ifss is not None:
        predictions['fasc'] = ifss_fascicles(oriented, ifss)
    for p in predictions.values():
        if p.shape != (1, 512, 512, 1) or not np.isfinite(p).all():
            raise ValueError('Invalid model prediction')
    summary, fig = measure_geometry(batch, oriented, predictions['apo'], predictions['fasc'], geometry, settings)
    summary['aponeurosis_pair'] = 'upstream'
    # Fallback only for failures (D045): upstream may pair skin/fat layers instead of the muscle's aponeuroses.
    if summary['status'] != 'ok' or not thickness_ok(summary, min_thickness_px):
        apo, _ = guided_apo(predictions['apo'], predictions['fasc'])
        if apo is not None:
            try:
                guided, guided_fig = measure_geometry(batch, oriented, apo, predictions['fasc'], geometry, settings)
            except Exception:  # the upstream geometry can raise on unusual masks; keep the upstream result
                guided, guided_fig = None, None
            if guided is not None and pick(summary, guided, min_thickness_px):
                if fig is not None:
                    plt.close(fig)
                summary, fig = dict(guided, aponeurosis_pair='fascicle_guided'), guided_fig
                predictions = dict(predictions, apo=apo)
            elif guided_fig is not None:
                plt.close(guided_fig)
    return SimpleNamespace(flip=flip, predictions=predictions, fig=fig, summary=summary)


def analyse(item, models, geometry, settings, out, ifss=None):
    path = ROOT / item['path']
    flip = item.get('flip', 'auto')
    if flip != 'auto' and (type(flip) is not int or flip not in (0, 1)):
        raise ValueError('flip must be 0, 1 or "auto"')
    scale = item.get('mm_per_pixel')
    if scale is not None and (type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0):
        raise ValueError('mm_per_pixel must be finite and positive')
    if scale is not None and item.get('calibration') is not None:
        raise ValueError('Give either mm_per_pixel or calibration, not both')
    original = read_image(path)
    # Calibrate on the full original: the inference crop may remove the ruler (D022).
    calibration = from_spec(item['calibration'], original) if item.get('calibration') is not None else None
    if calibration is not None:
        scale = calibration['mm_per_pixel']
    full_height, full_width = original.shape[:2]
    box = resolve_crop(item.get('crop', 'auto'), original)
    x0, y0, x1, y1 = box
    region = original[y0:y1, x0:x1]
    height, width = region.shape[:2]
    min_thickness_px = 10 / scale if scale else None  # published lower bound of test-set thickness
    def run(fascicle_model):
        found = []
        for f in ((0, 1) if flip == 'auto' else (flip,)):
            try:
                found.append(infer(region, f, models, geometry, settings, min_thickness_px, fascicle_model))
            except Exception as error:
                # Upstream geometry can raise in one orientation only (IMG_00232); keep the other.
                if flip != 'auto':
                    raise
                plt.close('all')
                found.append(SimpleNamespace(flip=f, error=error, fig=None, summary={
                    'fascicle_count': 0, 'status': 'error', 'error': f'{type(error).__name__}: {error}'}))
        return found
    candidates = run(None)
    chosen = candidates[choose([c.summary for c in candidates])]
    chosen.summary['fascicle_model'] = 'image'
    # Last resort (D048/D050): the video IFSS fascicle model, only when the image model gives no estimate.
    if ifss is not None and chosen.summary['status'] not in ('ok', 'low_confidence'):
        video = run(ifss)
        best = video[choose([c.summary for c in video])]
        if choose([chosen.summary, best.summary]) == 1:
            for c in candidates:
                if c.fig is not None:
                    plt.close(c.fig)
            candidates, chosen = video, best
            chosen.summary['fascicle_model'] = 'ifss'
        else:
            for c in video:
                if c.fig is not None:
                    plt.close(c.fig)
    if chosen.summary['status'] == 'error':
        raise chosen.error
    for c in candidates:
        if c is not chosen and c.fig is not None:
            plt.close(c.fig)
    flip, fig = chosen.flip, chosen.fig
    out.mkdir()
    np.savez_compressed(out / 'probabilities.npz', **chosen.predictions)
    for key, p in chosen.predictions.items():
        threshold = settings['aponeurosis_detection_threshold' if key == 'apo' else 'fascicle_detection_threshold']
        mask = cv2.resize((p[0, :, :, 0] > threshold).astype('uint8')*255,
                          (width, height), interpolation=cv2.INTER_NEAREST)
        if flip:
            mask = np.fliplr(mask)
        full = np.zeros((full_height, full_width), dtype=np.uint8)
        full[y0:y1, x0:x1] = mask
        if not cv2.imwrite(str(out / f'{key}_mask.png'), full):
            raise OSError('Failed to save mask')
    row = {'input': item['path'], 'flip': flip, 'flip_mode': 'auto' if len(candidates) == 2 else 'manual',
           'width': full_width, 'height': full_height, 'crop': list(box),
           'length_unit': 'pixels', 'mm_per_pixel': scale, **chosen.summary,
           'thickness_mm': None, 'fascicle_length_mm': None,
           'calibration_method': calibration['method'] if calibration else ('mm_per_pixel' if scale else None),
           'calibration_status': calibration['status'] if calibration else ('provided' if scale else None)}
    if calibration is not None:
        row['calibration'] = calibration
    if len(candidates) == 2:
        row['flip_candidates'] = [{'flip': c.flip, 'status': c.summary['status'],
                                   'fascicle_count': c.summary['fascicle_count'],
                                   **({'error': c.summary['error']} if 'error' in c.summary else {})}
                                  for c in candidates]
    lines = []
    if fig is not None:
        ax = fig.axes[0]
        ax.images[0].set_data(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax.images[0].set_extent([0, full_width, full_height, 0])
        for line in ax.lines:
            x = restore_x(line.get_xdata(), width, flip) + x0
            y = np.asarray(line.get_ydata(), dtype=float) + y0
            line.set_xdata(x)
            line.set_ydata(y)
            lines.append({'label': line.get_label(), 'x': x.tolist(), 'y': y.tolist()})
        ax.set_xlim(0, full_width); ax.set_ylim(full_height, 0)
        for annotation in list(ax.texts):
            if row['status'] in ('ok', 'low_confidence'):
                tx, ty = annotation.get_position()
                annotation.set_position((tx + x0, ty + y0))
            else:
                annotation.remove()
        if scale is not None and row['status'] in ('ok', 'low_confidence'):
            ax.text(0.97, 0.03, f"MT {row['thickness_px']*scale:.1f} mm, FL {row['fascicle_length_px']*scale:.1f} mm "
                    f"({scale:.4f} mm/px, {row['calibration_method']})", transform=ax.transAxes,
                    color='cyan', fontsize=11, ha='right')
        if row['status'] == 'low_confidence':
            ax.text(0.03, 0.97, f"Low confidence: {row['fascicle_count']} fascicle(s), fewer than {MIN_FASCICLES}",
                    transform=ax.transAxes, color='yellow', fontsize=12, va='top')
        elif row['status'] != 'ok':
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
    inputs.add_argument('--manifest', type=Path, help='JSON list of path, optional flip (0, 1 or "auto", default auto), crop ("auto", "none" or [x0,y0,x1,y1], default auto), and either verified mm_per_pixel or calibration (see calibration.from_spec)')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite a run')
    parser.add_argument('--profile', choices=['reference', 'current'], default='reference')
    parser.add_argument('--ifss-fallback', action='store_true', help='retry images without an estimate using the video IFSS fascicle model (v10)')
    parser.add_argument('--weights-dir', type=Path, help='folder with apo/fasc _finetuned.weights.h5 from finetune.py')
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
    geometry, models = load_geometry(), load_models(args.weights_dir)
    ifss = None
    if args.ifss_fallback:
        from run_video import load_video_model  # lazy: run_video imports this module
        ifss = load_video_model()
    run = {'upstream_revision': REVISION, 'tensorflow': tf.__version__, 'settings': settings,
           'checkpoints': CHECKPOINTS, 'inputs': items, 'results': [],
           'ifss_fallback': bool(args.ifss_fallback),
           'finetuned_weights': sorted(str(p) for p in Path(args.weights_dir).glob('*_finetuned.weights.h5')) if args.weights_dir else [],
           'adapter_settings': {'minimal_fascicle_count': MIN_FASCICLES, 'default_flip': 'auto', 'default_crop': 'auto'}}
    for index, item in enumerate(items):
        start = time.monotonic()
        output = args.output/f'{index+1:02d}_{Path(item["path"]).stem}'
        try:
            row = analyse(item, models, geometry, settings, output, ifss)
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
    keys = sorted({k for r in run['results'] for k in r if k not in ('reference_absolute_errors', 'flip_candidates', 'calibration')})
    with (args.output/'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(run['results'])
    if not any(r['status']=='ok' for r in run['results']):
        raise SystemExit('No successful measurements; see run.json')


if __name__ == '__main__':
    main()
