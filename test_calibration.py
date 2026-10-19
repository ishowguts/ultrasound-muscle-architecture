"""Run: .venv/bin/python test_calibration.py (no model weights needed)."""
import cv2
import numpy as np
from calibration import manual_calibration, bar_calibration


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Invalid calibration was accepted')


points = [[10, 20], [12, 120]]
manual = manual_calibration(points, 10, (200, 200))
assert manual['mm_per_pixel'] == 0.1
assert manual['distance_px'] == 100
assert manual_calibration(points[::-1], 10, (200, 200))['mm_per_pixel'] == 0.1
assert manual_calibration([[60, 70], [62, 170]], 10, (250, 250))['mm_per_pixel'] == 0.1
rejects(lambda: manual_calibration([[1, 0], [1, 1e-300]], 1e300, (200, 200)))
for value in (True, '10', 0, -1, float('nan'), float('inf')):
    rejects(lambda: manual_calibration(points, value, (200, 200)))
for bad in ([[10, 20], [40, 20]], [[-1, 20], [12, 120]], [[10, 20], [200, 120]],
            [[10, float('nan')], [12, 120]], [[10, 20]], [['10', '20'], ['12', '120']]):
    rejects(lambda: manual_calibration(bad, 10, (200, 200)))

image = np.zeros((200, 200), dtype=np.uint8)
for y in (90, 110, 130):
    image[y, 180:195] = 255
candidate = bar_calibration(image)
assert candidate['status'] == 'candidate' and candidate['distance_px'] == 20
assert candidate['mm_per_pixel'] is None and candidate['candidate_mm_per_pixel'] is None
candidate = bar_calibration(image, 5)
assert candidate['candidate_mm_per_pixel'] == 0.25 and candidate['mm_per_pixel'] is None
assert bar_calibration(image, 5, confirmed=True)['mm_per_pixel'] == 0.25
for colour in (cv2.COLOR_GRAY2BGR, cv2.COLOR_GRAY2BGRA):
    assert bar_calibration(cv2.cvtColor(image, colour), 5)['distance_px'] == 20
empty = bar_calibration(np.zeros_like(image), 5, confirmed=True)
assert empty['status'] == 'not_detected' and empty['mm_per_pixel'] is None and not empty['confirmed']
rejects(lambda: bar_calibration(image, confirmed=True))
rejects(lambda: bar_calibration(image, 0))
rejects(lambda: bar_calibration(image.astype(float), 5))
rejects(lambda: bar_calibration(image, 5, confirmed='yes'))
rejects(lambda: bar_calibration(np.zeros((0, 0), dtype=np.uint8), 5))
# Even a successful bright-gap detection is not proof of ruler semantics.
assert bar_calibration(image, 5)['status'] != 'confirmed'
from calibration import auto_ruler, from_spec


def ruler(width, height, column, start, spacing, count, axis='v'):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(count):
        p = int(start + i * spacing)
        if axis == 'v':
            canvas[p:p + 2, column:column + 10] = 200
        else:
            canvas[column:column + 8, p:p + 2] = 200
    return canvas


juniper = auto_ruler(ruler(1200, 800, 1150, 87, 76, 9))
assert juniper['layout'] == 'juniper' and juniper['tick_mm'] == 5 and abs(juniper['mm_per_pixel'] - 5 / 76) < 1e-6
shallow = auto_ruler(ruler(1200, 800, 1150, 87, 32, 16))
assert shallow['tick_mm'] == 2 and abs(shallow['mm_per_pixel'] - 2 / 32) < 1e-6, shallow
telemed = auto_ruler(ruler(1088, 644, 2, 5, 126, 6))
assert telemed['layout'] == 'telemed' and abs(telemed['mm_per_pixel'] - 10 / 126) < 1e-6
bottom = auto_ruler(ruler(465, 513, 503, 53, 78, 6, axis='h'))
assert bottom['layout'] == 'bottom_ticks' and abs(bottom['mm_per_pixel'] - 10 / 78) < 1e-6
assert auto_ruler(np.zeros((800, 1200, 3), np.uint8))['mm_per_pixel'] is None
assert from_spec({'method': 'auto'}, ruler(1200, 800, 1150, 87, 76, 9))['status'] == 'detected'
rejects(lambda: auto_ruler(np.zeros((10, 10), float)))
print('PASS: manual vertical scale, reversed/translated points, invalid inputs, '
      'upstream synthetic 20px gap, grayscale/BGR/BGRA, confirmation gate, empty detection, '
      'automatic Juniper 5/2 mm, Telemed, bottom-tick rulers.')
