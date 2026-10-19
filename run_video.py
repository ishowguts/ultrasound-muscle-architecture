"""Offline video analysis with the pinned DL_Track_US video routine (stacked IFSS fascicle model).

Run: .venv/bin/python run_video.py data/raw/dltrack_examples/video/video_calf_raise.mp4 --output outputs/video_run1
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import run_analysis as ra  # sets matplotlib/TF environment first
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = ra.ROOT
VIDEO_CHECKPOINT = ('model-fasc-IFSS_1Encoder_3d_a08_g2_noaug_nocrop_HL.h5',
                    '6d50625e2c58b5e9540c4b417ee96ad764f10aef903a3351b9772ae47bb0d4a8')


def pinned_module(relative, name):
    source = ROOT / 'references/DL_Track_US'
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != ra.REVISION:
        raise ValueError('Upstream revision differs from the validated revision')
    original = subprocess.check_output(['git', '-C', str(source), 'show', f'{ra.REVISION}:{relative}'])
    if (source / relative).read_bytes() != original:
        raise ValueError(f'Upstream {relative} was modified; review before running')
    spec = importlib.util.spec_from_file_location(name, source / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_video_routine():
    # The video module imports from the DL_Track_US package, whose __init__ starts the GUI (D007);
    # register only the unchanged geometry module under that name instead.
    calculations = pinned_module('DL_Track_US/gui_helpers/do_calculations.py', 'DL_Track_US.gui_helpers.do_calculations')
    for name in ('DL_Track_US', 'DL_Track_US.gui_helpers'):
        sys.modules.setdefault(name, ModuleType(name))
    sys.modules['DL_Track_US.gui_helpers.do_calculations'] = calculations
    module = pinned_module('DL_Track_US/gui_helpers/do_calculations_video.py', 'dltrack_video')
    # Headless shims: upstream reports errors with a Tk dialog and closes OpenCV windows,
    # neither of which exists here. Errors are printed instead; analysis code is unchanged.
    module.tk = SimpleNamespace(messagebox=SimpleNamespace(
        showerror=lambda *a, **k: print('Upstream video error:', *a, file=sys.stderr)))
    module.cv2 = SimpleNamespace(**{n: getattr(cv2, n) for n in dir(cv2) if not n.startswith('__')})
    module.cv2.destroyAllWindows = lambda: None
    return module.doCalculationsVideo


def load_video_model():
    name, expected = VIDEO_CHECKPOINT
    path = ROOT / 'models' / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Checkpoint checksum mismatch: {name}')
    model = tf.keras.models.load_model(path, compile=False)
    if model.input_shape != (None, None, 512, 512, 1) or model.output_shape != (None, None, 512, 512, 2):
        raise ValueError(f'Unexpected checkpoint shape: {name}')
    return model


def analysed_frames(count, step):
    """Frame offsets that have upstream results: step frames, minus the first two (stack warm-up)."""
    return [i for i in range(count) if not (i % step == 0 and i // step < 2)]


def frame_rows(results, count, step, fps, width, height, start=0, scale=None):
    lengths_all, angles_all, _, _, thickness_all = results
    offsets = analysed_frames(count, step)
    if len(lengths_all) > len(offsets):
        raise ValueError('Upstream returned more frames than were read')
    rows = []
    for i in range(count):
        rows.append({'frame': start + i, 'time_s': round((start + i) / fps, 4), 'status': 'not_analysed',
                     'fascicle_count': 0, 'thickness_px': None, 'fascicle_length_px': None, 'pennation_deg': None})
    for offset, lengths, angles, thickness in zip(offsets, lengths_all, angles_all, thickness_all):
        if offset % step:
            continue  # skipped by step: upstream stores NaN placeholders
        pairs = [(l, a) for l, a in zip(lengths, angles) if np.isfinite(l) and np.isfinite(a)]
        summary = ra.measurement_summary([p[0] for p in pairs], [p[1] for p in pairs], thickness, width, height)
        rows[offset].update(summary, fascicle_count=len(pairs))
    for row in rows:
        for key in ('thickness', 'fascicle_length'):
            row[key + '_mm'] = row[key + '_px'] * scale if scale and row[key + '_px'] is not None else None
    return rows


def plot(rows, path, unit):
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    keys = [('thickness', 'Muscle thickness'), ('fascicle_length', 'Fascicle length'), ('pennation_deg', 'Pennation angle')]
    for ax, (key, label) in zip(axes, keys):
        column = key if key == 'pennation_deg' else f'{key}_{unit}'
        for status, marker in (('ok', 'o'), ('low_confidence', 'x')):
            points = [(r['time_s'], r[column]) for r in rows if r['status'] == status and r[column] is not None]
            if points:
                ax.plot(*zip(*points), marker, markersize=3, label=status)
        ax.set_ylabel(f"{label} ({'deg' if key == 'pennation_deg' else unit})")
        ax.grid(alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite a run')
    parser.add_argument('--flip', choices=['auto', '0', '1'], default='auto',
                        help='auto: choose with the image pipeline on the first frame')
    parser.add_argument('--start', type=int, default=0, help='First frame index')
    parser.add_argument('--frames', type=int, help='Number of frames (default: to the end)')
    parser.add_argument('--step', type=int, default=1, help='Analyse every Nth frame')
    parser.add_argument('--mm-per-pixel', type=float, help='Verified scale for the video frames')
    args = parser.parse_args()
    if args.start < 0 or args.step < 1 or (args.frames is not None and args.frames < 3):
        parser.error('start must be >= 0, step >= 1, frames >= 3')
    if args.mm_per_pixel is not None and not (np.isfinite(args.mm_per_pixel) and args.mm_per_pixel > 0):
        parser.error('mm-per-pixel must be finite and positive')
    cap = cv2.VideoCapture(str(args.video))
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total < 3 or not fps > 0 or args.start >= total:
        parser.error('Unreadable video or start beyond its end')
    count = min(args.frames or total, total - args.start)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ok, first = cap.read()
    if not ok:
        parser.error('Could not read the first frame')
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    args.output.mkdir(parents=True, exist_ok=False)
    # Same documented reference settings as the image path (D008). The upstream current
    # settings' 200 px minimum width crashes its own video routine (do_calculations_video.py:815).
    settings = ra.REFERENCE_SETTINGS.copy()
    image_models = ra.load_models()
    candidates = []
    if args.flip == 'auto':
        geometry = ra.load_geometry()
        candidates = [ra.infer(first, f, image_models, geometry, ra.REFERENCE_SETTINGS) for f in (0, 1)]
        for c in candidates:
            if c.fig is not None:
                plt.close(c.fig)
        flip = candidates[ra.choose([c.summary for c in candidates])].flip
    else:
        flip = int(args.flip)
    apo = image_models['apo']
    del image_models['fasc']
    video_fn, ifss = load_video_routine(), load_video_model()
    writer = cv2.VideoWriter(str(args.output / 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    gui = SimpleNamespace(should_stop=False, is_running=True, do_break=lambda: None)
    started = time.monotonic()
    results = video_fn(vid_len=count, cap=cap, vid_out=writer, flip='flip' if flip else 'no_flip',
                       apo_model=apo, fasc_model=ifss, calib_dist=None, dic=settings, step=args.step,
                       filter_fasc=False, gui=gui, segmentation_mode='stacked', frame_callback=None)
    seconds = round(time.monotonic() - started, 1)
    if results is None:
        raise SystemExit('Upstream video analysis failed; see console output')
    rows = frame_rows(results, count, args.step, fps, width, height, args.start, args.mm_per_pixel)
    keys = list(rows[0])
    with (args.output / 'frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    plot(rows, args.output / 'timeseries.png', 'mm' if args.mm_per_pixel else 'px')
    counts = {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})}
    run = {'video': args.video.name, 'frames_in_video': total, 'fps': fps, 'width': width, 'height': height,
           'start': args.start, 'frames': count, 'step': args.step, 'flip': flip, 'flip_mode': args.flip,
           'flip_candidates': [{'flip': c.flip, **{k: c.summary[k] for k in ('status', 'fascicle_count')}} for c in candidates],
           'mm_per_pixel': args.mm_per_pixel, 'status_counts': counts, 'seconds': seconds,
           'seconds_per_analysed_frame': round(seconds / max(1, len(analysed_frames(count, args.step))), 2),
           'upstream_revision': ra.REVISION, 'segmentation_mode': 'stacked', 'settings': settings,
           'checkpoints': {'apo': ra.CHECKPOINTS['apo'], 'fasc_video': VIDEO_CHECKPOINT},
           'notes': 'First two analysed frames are stack warm-up (no result). Frames are not cropped. '
                    'annotated.mp4 is written by the upstream routine and omits warm-up frames.'}
    (args.output / 'run.json').write_text(json.dumps(run, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: run[k] for k in ('frames', 'flip', 'status_counts', 'seconds', 'seconds_per_analysed_frame')}))


if __name__ == '__main__':
    main()
