"""Small runnable checks: .venv/bin/python test_analysis.py"""
from pathlib import Path
import tempfile
import cv2
import numpy as np
from run_analysis import prepare, restore_x, load_geometry, measurement_summary, analyse, REFERENCE_SETTINGS
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
assert measurement_summary([100], [20], 40, 465, 513)['status'] == 'ok'
assert measurement_summary([float('nan')], [20], 40, 465, 513)['status'] == 'invalid_geometry'
assert measurement_summary([], [], 40, 465, 513)['status'] == 'no_fascicles'
print('Passed: preprocessing, colour order, grayscale, invalid input, coordinate restoration, empty geometry.')
