"""App handler checks with stub models: .venv/bin/python test_app.py (no weights or server needed)."""
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

import app

ones = SimpleNamespace(numpy=lambda: np.ones((1, 512, 512, 1), np.float32))
app.STATE['models'] = {'apo': lambda b, training: ones, 'fasc': lambda b, training: ones}


def geometry(batch, oriented, h, w, *_):
    fig, ax = plt.subplots()
    ax.imshow(oriented, extent=[0, w, h, 0])
    ax.plot([0, w-1], [0, h-1], label='Fascicle 0')
    return [100.0, 110.0, 120.0], [20.0, 21.0, 22.0], None, None, 50.0, fig


app.STATE['geometry'] = geometry
with patch("app.time.strftime", return_value="20000101-000000"), tempfile.TemporaryDirectory() as directory:
    app.RUNS = Path(directory) / 'runs'
    scan = Path(directory) / 'scan.tif'
    assert cv2.imwrite(str(scan), np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8))
    preview, path, points, _ = app.on_upload(str(scan))
    assert preview.shape == (200, 300, 3) and path == str(scan) and points == []
    try: app.on_upload(str(Path(directory) / 'missing.tif'))
    except gr.Error: pass
    else: raise AssertionError('Unreadable upload accepted')
    _, points, _ = app.on_click(path, points, SimpleNamespace(index=[150, 20]))
    _, points, note = app.on_click(path, points, SimpleNamespace(index=[152, 120]))
    assert points == [[150.0, 20.0], [152.0, 120.0]] and '100 px' in note
    _, restarted, _ = app.on_click(path, points, SimpleNamespace(index=[5, 5]))
    assert restarted == [[5.0, 5.0]], 'Third click must start a new pair'
    overlay, status, table, files = app.analyse(path, 'Auto (try both)', 'Full image', True, 10, points)
    assert overlay and 'Complete measurement' in status and '0.10000 mm/px' in status
    assert table[0] == ['Muscle thickness', '50.00 px', '5.00 mm'], table[0]
    assert {Path(f).name for f in files} == {'measurements.csv', 'measurements.json', 'overlay.png', 'lines.json'}
    csv_text = Path(files[0]).read_text()
    assert directory not in csv_text and 'scan.tif' in csv_text, 'Exports must not contain local paths'
    assert all(directory not in Path(f).read_text() for f in files if f.endswith(('.json', '.csv')))
    _, status, table, second_files = app.analyse(path, 'As uploaded', 'Full image', False, 10, [])
    assert set(files).isdisjoint(second_files), 'Same-second analyses must have distinct output paths'
    assert all(Path(f).exists() for f in files), 'The earlier run must remain intact'
    assert table[0][2] == 'not calibrated' and 'mm/px' not in status
    ruled = Path(directory) / 'ruled.tif'
    canvas = np.zeros((800, 1200, 3), np.uint8)
    for i in range(9):
        canvas[87 + 76 * i:89 + 76 * i, 1150:1160] = 200
    assert cv2.imwrite(str(ruled), canvas)
    _, cleared, auto_on, ticked, note = app.on_auto(str(ruled))
    assert auto_on and ticked and cleared == [] and 'juniper' in note and '40 mm' in note, note
    _, _, auto_off, _, note = app.on_auto(path)  # random noise: no ruler
    assert not auto_off and 'No ruler found' in note
    _, status, table, _ = app.analyse(str(ruled), 'As uploaded', 'Full image', True, 10, [], True)
    assert 'automatically detected ruler' in status and '0.06579 mm/px' in status, status
    marked = app.show(path, points)
    assert (marked[120] == [255, 220, 0]).all(axis=1).sum() > 250, 'Guide line must span the width'
    for bad in ((None, 'Auto (try both)', 'Full image', False, 10, []),
                (path, 'Auto (try both)', 'Full image', True, 45, [[216, 11], [142, 21]]),  # observed user case
                (path, 'Auto (try both)', 'Full image', True, 450, points),
                (path, 'Auto (try both)', 'Full image', True, 10, [[1, 1]]),
                (path, 'Auto (try both)', 'Full image', True, 0, points)):
        try: app.analyse(*bad)
        except gr.Error: pass
        else: raise AssertionError(f'Invalid request accepted: {bad[3:]}')
print('Passed: upload, unreadable upload, ruler clicks, restart, automatic ruler, calibrated and uncalibrated analysis, exports without local paths, invalid requests.')
