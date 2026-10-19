"""Frame alignment checks for run_video.py: .venv/bin/python test_video.py (no weights needed)."""
import math

from run_video import analysed_frames, frame_rows

nan = math.nan
assert analysed_frames(6, 1) == [2, 3, 4, 5]
assert analysed_frames(7, 2) == [1, 3, 4, 5, 6]  # step frames 0 and 2 are warm-up
# Two analysed frames for count=4, step=1: offsets 2 (ok, 3 fascicles) and 3 (one NaN fascicle dropped).
results = ([[100, 110, 120], [90, nan]], [[20, 21, 22], [18, nan]], None, None, [50.0, 40.0])
rows = frame_rows(results, 4, 1, fps=10, width=100, height=100, start=20, scale=0.1)
assert [r['frame'] for r in rows] == [20, 21, 22, 23] and rows[3]['time_s'] == 2.3
assert [r['status'] for r in rows] == ['not_analysed', 'not_analysed', 'ok', 'low_confidence']
assert rows[2]['fascicle_length_px'] == 110 and rows[2]['thickness_mm'] == 5.0
assert rows[3]['fascicle_count'] == 1 and rows[0]['thickness_mm'] is None
# Step placeholders (NaN thickness) stay not_analysed; the 10000 sentinel becomes invalid.
# Upstream order for 7 frames, step 2: NaN(1), NaN(3), frame 4, NaN(5), frame 6.
results = ([[nan], [nan], [100, 110, 120], [nan], [100]], [[nan], [nan], [20, 21, 22], [nan], [20]], None, None,
           [nan, nan, 60.0, nan, 10000])
rows = frame_rows(results, 7, 2, fps=1, width=100, height=100)
assert [r['status'] for r in rows] == ['not_analysed'] * 4 + ['ok', 'not_analysed', 'invalid_geometry'], [r['status'] for r in rows]
try: frame_rows(([[1]] * 5, [[1]] * 5, None, None, [1] * 5), 4, 1, 1, 10, 10)
except ValueError: pass
else: raise AssertionError('Too many upstream frames accepted')
print('Passed: warm-up and step alignment, NaN fascicles, low confidence, sentinel rejection, mm scaling.')
