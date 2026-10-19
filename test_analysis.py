"""Small runnable checks: .venv/bin/python test_analysis.py"""
from pathlib import Path
import tempfile
import cv2
import numpy as np
from run_analysis import (prepare, restore_x, load_geometry, measurement_summary, analyse, REFERENCE_SETTINGS,
                          detect_scan_field, resolve_crop, choose)
import json
import matplotlib.pyplot as plt
from types import SimpleNamespace

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)/'colour.png'
    pixels = np.zeros((16, 24, 3), dtype=np.uint8)
    pixels[:, :12] = (15, 70, 200)
    pixels[:, 12:] = (100, 50, 20)
    assert cv2.imwrite(str(path), pixels)
    original, oriented, batch = prepare(path, 1)
    assert np.array_equal(original, pixels)
    assert np.array_equal(oriented, pixels[:, ::-1])
    assert batch.shape == (1, 512, 512, 3) and batch.dtype == np.float32
    np.testing.assert_allclose(batch[0, 250, 40], np.array([100,50,20])/255, atol=1e-6)
    gray = Path(directory)/'gray.tif'
    assert cv2.imwrite(str(gray), np.full((16,24),127,dtype=np.uint8))
    _, _, gray_batch = prepare(gray, 0)
    assert np.array_equal(gray_batch[...,0],gray_batch[...,1])
    for flip in (-1, 2, True):
        try: prepare(path, flip)
        except ValueError: pass
        else: raise AssertionError('Invalid orientation accepted')
    try: prepare(Path(directory)/'missing.png',0)
    except ValueError: pass
    else: raise AssertionError('Missing image accepted')

for scale in (True, '1', 0, -1, float('nan'), float('inf')):
    try: analyse({'path': 'unused.png', 'mm_per_pixel': scale}, None, None, None, None)
    except ValueError: pass
    else: raise AssertionError('Invalid calibration accepted')

x = np.array([-10,0,12,23,30])
np.testing.assert_array_equal(restore_x(restore_x(x,24,1),24,1),x)
zeros = SimpleNamespace(predict=lambda _: np.zeros((1,512,512,1),dtype=np.float32))
result = load_geometry()(np.zeros((1,512,512,3)),np.zeros((128,128,3),dtype=np.uint8),
                         128,128,None,10,zeros,zeros,REFERENCE_SETTINGS,False)
assert result == (None,)*6, 'Empty masks must yield a failure, not zero measurements'
assert measurement_summary([], [], 10000, 465, 513)['thickness_px'] is None
assert measurement_summary([100], [20], 40, 465, 513)['status'] == 'low_confidence'
assert measurement_summary([100, 110, 120], [20, 21, 22], 40, 465, 513)['status'] == 'ok'
assert measurement_summary([float('nan')], [20], 40, 465, 513)['status'] == 'invalid_geometry'
assert measurement_summary([], [], 40, 465, 513)['status'] == 'no_fascicles'

# Scan-field crop: speckle block on black with thin UI text stays tight; all-black keeps full frame.
rng = np.random.default_rng(0)
frame = np.zeros((300, 400, 3), dtype=np.uint8)
frame[60:240, 120:330] = rng.integers(40, 220, (180, 210, 1), dtype=np.uint8)
cv2.putText(frame, 'MI 0.9', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
box = detect_scan_field(frame)
assert all(abs(a-b) <= 8 for a, b in zip(box, (120, 60, 330, 240))), box
assert detect_scan_field(np.zeros((300, 400, 3), np.uint8)) == (0, 0, 400, 300)
sliver = rng.integers(40, 220, (300, 400, 3), dtype=np.uint8); sliver[270:] = 0
assert detect_scan_field(sliver) == (0, 0, 400, 300), 'Mild trims must keep the full frame'
assert resolve_crop('none', frame) == (0, 0, 400, 300)
assert resolve_crop([10, 20, 110, 220], frame) == (10, 20, 110, 220)
for bad in ([0, 0, 500, 300], [10, 10, 5, 50], [0, 0, 1.5, 9], 'yes', None):
    try: resolve_crop(bad, frame)
    except ValueError: pass
    else: raise AssertionError(f'Invalid crop accepted: {bad}')

# Orientation choice: usable status first, then fascicle count, ties keep flip 0.
s = lambda status, n: {'status': status, 'fascicle_count': n}
assert choose([s('no_fascicles', 0), s('ok', 5)]) == 1
assert choose([s('ok', 3), s('low_confidence', 2)]) == 0
assert choose([s('ok', 4), s('ok', 9)]) == 1
assert choose([s('ok', 7), s('ok', 7)]) == 0
assert choose([s('invalid_geometry', 0), s('no_geometry', 0)]) == 0
assert choose([s('error', 0), s('no_geometry', 0)]) == 1

# End to end with stub models/geometry: crop offset and flip map back to original pixels.
ones = SimpleNamespace(numpy=lambda: np.ones((1, 512, 512, 1), np.float32))
stub_models = {'apo': lambda b, training: ones, 'fasc': lambda b, training: ones}
def stub_geometry(batch, oriented, h, w, *_):
    fig, ax = plt.subplots()
    ax.imshow(oriented, extent=[0, w, h, 0])
    ax.plot([0, w-1], [0, h-1], label='Fascicle 0')
    ax.text(50, h-150, 'median')
    return [100.0, 110.0, 120.0], [20.0, 21.0, 22.0], None, None, 50.0, fig
with tempfile.TemporaryDirectory() as directory:
    image = Path(directory)/'scan.png'
    assert cv2.imwrite(str(image), frame)
    row = analyse({'path': str(image), 'flip': 1, 'crop': [100, 50, 350, 250]}, stub_models,
                  stub_geometry, REFERENCE_SETTINGS, Path(directory)/'out')
    assert row['status'] == 'ok' and row['crop'] == [100, 50, 350, 250] and (row['width'], row['height']) == (400, 300)
    line = json.loads((Path(directory)/'out/lines.json').read_text())['lines'][0]
    assert line['x'] == [349.0, 100.0] and line['y'] == [50.0, 249.0], line
    mask = cv2.imread(str(Path(directory)/'out/apo_mask.png'), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (300, 400) and mask[50:250, 100:350].min() == 255
    assert mask[:50].max() == 0 and mask[:, :100].max() == 0 and mask[:, 350:].max() == 0
    auto = analyse({'path': str(image)}, stub_models, stub_geometry, REFERENCE_SETTINGS, Path(directory)/'auto')
    assert auto['flip_mode'] == 'auto' and auto['flip'] == 0 and len(auto['flip_candidates']) == 2
    assert auto['crop'] == list(detect_scan_field(frame))
    def apo_geometry(batch, oriented, h, w, *_):
        fig, ax = plt.subplots()
        ax.imshow(oriented, extent=[0, w, h, 0])
        ax.plot([0, w - 1], [20, 20], label='Upper Aponeurosis')
        ax.plot([60, w - 1], [90, 90], label='Lower Aponeurosis')
        return [100.0, 110.0, 120.0], [20.0, 21.0, 22.0], None, None, 50.0, fig
    apo = analyse({'path': str(image), 'flip': 0, 'crop': 'none'}, stub_models, apo_geometry, REFERENCE_SETTINGS, Path(directory)/'apo')
    assert apo['thickness_px'] == 70 and apo['thickness_upstream_px'] == 50, apo
    calls = []
    def flaky_geometry(*args):
        calls.append(1)
        if len(calls) == 1:
            raise IndexError('list index out of range')
        return stub_geometry(*args)
    recovered = analyse({'path': str(image)}, stub_models, flaky_geometry, REFERENCE_SETTINGS, Path(directory)/'flaky')
    assert recovered['flip'] == 1 and recovered['status'] == 'ok' and recovered['flip_candidates'][0]['status'] == 'error'
    def broken_geometry(*args):
        raise IndexError('list index out of range')
    try: analyse({'path': str(image)}, stub_models, broken_geometry, REFERENCE_SETTINGS, Path(directory)/'broken')
    except IndexError: pass
    else: raise AssertionError('Both orientations failing must raise')
    spec = {'method': 'manual', 'points': [[10, 20], [12, 120]], 'distance_mm': 10}
    mm = analyse({'path': str(image), 'flip': 0, 'crop': [100, 50, 350, 250], 'calibration': spec},
                 stub_models, stub_geometry, REFERENCE_SETTINGS, Path(directory)/'mm')
    # Points lie outside the crop: calibration must use original-image pixels.
    assert mm['mm_per_pixel'] == 0.1 and mm['calibration_status'] == 'provided'
    assert mm['thickness_mm'] == 5.0 and abs(mm['fascicle_length_mm'] - 11.0) < 1e-9
    bar = analyse({'path': str(image), 'flip': 0, 'calibration': {'method': 'bar', 'spacing_mm': 5}},
                  stub_models, stub_geometry, REFERENCE_SETTINGS, Path(directory)/'bar')
    assert bar['mm_per_pixel'] is None and bar['thickness_mm'] is None and bar['calibration_method'] == 'upstream_bar'
    for bad in ({'path': str(image), 'mm_per_pixel': 0.1, 'calibration': spec},
                {'path': str(image), 'calibration': {'method': 'dpi'}}):
        try: analyse(bad, stub_models, stub_geometry, REFERENCE_SETTINGS, Path(directory)/'bad')
        except ValueError: pass
        else: raise AssertionError('Invalid calibration request accepted')
    for bad_flip in (2, 'yes', True):
        try: analyse({'path': str(image), 'flip': bad_flip}, None, None, None, None)
        except ValueError: pass
        else: raise AssertionError('Invalid flip accepted')
# Perpendicular thickness (measure.py): parallel, sloped and offset-start aponeuroses.
from measure import perpendicular_thickness, thickness_from_lines
xs = np.arange(0, 400)
assert abs(perpendicular_thickness(xs, np.full(400, 10.0), xs, np.full(400, 60.0)) - 50) < 1e-9
slope = np.radians(10)
t = perpendicular_thickness(xs, xs * np.tan(slope), xs, xs * np.tan(slope) + 50)
assert abs(t - 50 * np.cos(slope)) < 1e-6, t
# Lower aponeurosis starting at x=150 (upstream's index bug shifts its search there) still measures 50.
assert abs(perpendicular_thickness(xs, np.full(400, 10.0), xs[150:], np.full(250, 60.0)) - 50) < 1e-9
assert perpendicular_thickness(xs[:100], np.zeros(100), xs[200:], np.zeros(200)) is None
assert thickness_from_lines([('Upper Aponeurosis', xs, np.full(400, 0.0))]) is None
# Fascicle-guided aponeurosis pair: skin band (y 30), superficial (y 120), deep (y 400); fascicles only in
# the muscle between y 130 and 390 -> keep bands 1 and 2, drop the skin band.
from apo_pair import guided_apo
from run_analysis import pick
apo_map = np.zeros((1, 512, 512, 1), np.float32)
for y in (30, 120, 400):
    apo_map[0, y:y + 6, 20:490, 0] = 0.9
fasc_map = np.zeros_like(apo_map); fasc_map[0, 150:380:20, 40:470, 0] = 0.8
guided, chosen = guided_apo(apo_map, fasc_map)
assert chosen == (1, 2) and guided[0, 30:36, :, 0].max() == 0 and guided[0, 120:126, 100, 0].min() > 0.5
assert guided_apo(apo_map[:, :, :, :] * 0, fasc_map) == (None, None)
state = lambda status, n, t: {'status': status, 'fascicle_count': n, 'thickness_px': t}
assert pick(state('no_fascicles', 0, 80), state('ok', 6, 280), 130) == 1      # recovered muscle
assert pick(state('no_fascicles', 0, 80), state('no_fascicles', 0, 280), 130) == 1  # at least plausible MT
assert pick(state('low_confidence', 2, 300), state('no_fascicles', 0, 300), 130) == 0
assert pick(state('no_fascicles', 0, 280), state('no_fascicles', 0, 280), 130) == 0  # tie keeps upstream
print('Passed: preprocessing, colour order, grayscale, invalid input, coordinate restoration, empty geometry, '
      'scan-field crop, orientation choice, low confidence, crop/flip mapping, calibration before crop, one-orientation crash, perpendicular thickness, fascicle-guided aponeurosis fallback.')
