"""Local web app: upload a scan, optionally calibrate on its ruler, analyse, export.

Run: .venv/bin/python app.py  (opens http://127.0.0.1:7860, local only)
"""
import csv
import json
import os
from pathlib import Path
import time
from uuid import uuid4

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import matplotlib
matplotlib.use('Agg')  # handlers run in worker threads; GUI backends crash there on macOS
import cv2
import gradio as gr
import numpy as np

from calibration import auto_ruler
import run_analysis as ra

ROOT = ra.ROOT
RUNS = ROOT / 'outputs/app_runs'
STATE = {}  # models and geometry, loaded once in main()
ORIENTATION = {'Auto (try both)': 'auto', 'As uploaded': 0, 'Mirrored': 1}
CROP = {'Auto (remove black borders)': 'auto', 'Full image': 'none'}
# Plausible ultrasound pixel spacing; outside this the clicks almost certainly missed the ruler.
SCALE_RANGE = (0.01, 0.5)
MIN_RULER_PX = 50
ROWS = [('Muscle thickness', 'thickness'), ('Fascicle length (median)', 'fascicle_length'),
        ('Pennation angle (median)', 'pennation_deg')]


def show(path, points=(), ruler=None):
    """RGB preview of the original image with calibration points or the detected ruler marked."""
    image = cv2.cvtColor(ra.read_image(Path(path)), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    size = max(8, min(height, width) // 40)  # must stay visible after the browser scales the preview down
    if ruler:
        ends = [(ruler['line'], ruler['first_px']), (ruler['line'], ruler['last_px'])]
        for x, y in (ends if ruler['axis'] == 'v' else [(e[1], e[0]) for e in ends]):
            cv2.circle(image, (int(x), int(y)), size, (0, 255, 0), max(2, size // 4))
    for number, (x, y) in enumerate(points, 1):
        x, y = int(x), int(y)
        cv2.line(image, (0, y), (width - 1, y), (255, 220, 0), max(1, size // 6))  # only vertical distance counts
        cv2.circle(image, (x, y), size, (255, 0, 255), -1)
        cv2.putText(image, str(number), (x + size + 4, y + size), cv2.FONT_HERSHEY_SIMPLEX,
                    size / 10, (255, 0, 255), max(2, size // 5))
    return image


def on_upload(file):
    if file is None:
        return None, None, [], 'Upload a scan to begin.'
    try:
        return show(file), file, [], 'Scan loaded. Optional: click two ruler marks for millimetres.'
    except ValueError as error:
        raise gr.Error(str(error))


def on_click(path, points, evt: gr.SelectData):
    if path is None:
        return None, [], 'Upload a scan first.'
    points = [] if len(points) >= 2 else list(points)  # a third click starts over
    points.append([float(evt.index[0]), float(evt.index[1])])
    if len(points) == 1:
        note = f'Ruler point 1 at ({points[0][0]:.0f}, {points[0][1]:.0f}). Click the second mark.'
    else:
        note = (f'Ruler points set: {abs(points[1][1]-points[0][1]):.0f} px apart vertically. '
                'Enter their real distance in mm.')
    return show(path, points), points, note


def on_clear(path):
    return (show(path) if path else None), [], 'Calibration points cleared.'


def on_auto(path):
    """Detect the ruler; the user confirms the green circles sit on its first and last marks."""
    if path is None:
        raise gr.Error('Upload a scan first.')
    ruler = auto_ruler(ra.read_image(Path(path)))
    if ruler['mm_per_pixel'] is None:
        return show(path), [], False, False, ('No ruler found automatically for this scan layout. '
                                              'Click two ruler marks instead.')
    span_mm = (ruler['last_px'] - ruler['first_px']) * ruler['mm_per_pixel']
    note = (f"Found the {ruler['layout']} ruler: {ruler['intervals']} ticks of {ruler['tick_mm']:g} mm "
            f"({span_mm:.0f} mm, {ruler['mm_per_pixel']:.5f} mm/px). Check that the **green circles** sit on the "
            "first and last ruler marks, then Analyse. Clicking the scan switches back to manual points.")
    return show(path, ruler=ruler), [], True, True, note


def fmt(value, unit):
    return '—' if value is None else f'{value:.2f} {unit}'


def analyse(path, orientation, crop, calibrate, distance_mm, points, auto=False):
    if path is None:
        raise gr.Error('Upload a scan first.')
    item = {'path': str(Path(path).resolve()), 'flip': ORIENTATION[orientation], 'crop': CROP[crop]}
    if calibrate and auto and not points:
        item['calibration'] = {'method': 'auto'}
    elif calibrate:
        if len(points) != 2:
            raise gr.Error('Click exactly two ruler marks, or untick calibration.')
        pixels = abs(points[1][1] - points[0][1])
        if pixels < MIN_RULER_PX:
            raise gr.Error(f'Your two points are only {pixels:.0f} px apart vertically. Click two ruler marks far apart '
                           '(e.g. the top and bottom of the depth scale), then try again.')
        if distance_mm is not None and distance_mm > 0 and not SCALE_RANGE[0] <= distance_mm / pixels <= SCALE_RANGE[1]:
            raise gr.Error(f'{distance_mm:g} mm over {pixels:.0f} px gives {distance_mm / pixels:.4f} mm/px, which is not a '
                           f'plausible ultrasound scale ({SCALE_RANGE[0]}–{SCALE_RANGE[1]} mm/px). Check the points and distance.')
        item['calibration'] = {'method': 'manual', 'points': points, 'distance_mm': distance_mm}
    out = RUNS / f'{time.strftime("%Y%m%d-%H%M%S")}_{Path(path).stem}_{uuid4().hex}'
    RUNS.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        row = ra.analyse(item, STATE['models'], STATE['geometry'], ra.REFERENCE_SETTINGS, out)
    except ValueError as error:
        raise gr.Error(str(error))
    row['seconds'] = round(time.monotonic() - start, 3)
    row['input'] = Path(path).name  # never export local paths
    (out / 'measurements.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
    keys = [k for k in row if k not in ('flip_candidates', 'calibration')]
    with (out / 'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerow(row)
    scale = row['mm_per_pixel']
    table = []
    for label, key in ROWS:
        if key == 'pennation_deg':
            table.append([label, fmt(row[key], '°'), '—'])
        else:
            table.append([label, fmt(row[key+'_px'], 'px'), fmt(row[key+'_mm'], 'mm') if scale else 'not calibrated'])
    table.append(['Fascicles used', str(row['fascicle_count']), ''])
    messages = {
        'ok': '✅ **Complete measurement.** Experimental tool; not clinically validated.',
        'low_confidence': f"⚠️ **Low confidence:** only {row['fascicle_count']} fascicle(s) detected (fewer than {ra.MIN_FASCICLES}). Review the overlay before using these numbers.",
    }
    status = messages.get(row['status'], f"❌ **No complete measurement** (`{row['status']}`). Try another orientation or crop setting.")
    status += (f"\n\nOrientation used: {'mirrored' if row['flip'] else 'as uploaded'} ({row['flip_mode']}); "
               f"analysed region x {row['crop'][0]}–{row['crop'][2]}, y {row['crop'][1]}–{row['crop'][3]}; "
               f"{row['seconds']:.1f} s.")
    if scale:
        source = 'the automatically detected ruler' if row['calibration_method'] == 'auto_ruler' else 'your two ruler points'
        status += f" Scale {scale:.5f} mm/px from {source} (assumes square pixels)."
    files = [str(out / name) for name in ('measurements.csv', 'measurements.json', 'overlay.png', 'lines.json')
             if (out / name).exists()]
    overlay = str(out / 'overlay.png') if (out / 'overlay.png').exists() else None
    return overlay, status, table, files


def build():
    with gr.Blocks(title='Muscle ultrasound analysis') as demo:
        gr.Markdown('## Muscle ultrasound analysis\nAponeuroses, fascicles, thickness, fascicle length and '
                    'pennation angle using the pretrained DL_Track_US models (Ritsche et al.). Experimental college '
                    'project; not a medical device.')
        path, points, auto = gr.State(None), gr.State([]), gr.State(False)
        with gr.Row():
            with gr.Column():
                upload = gr.File(label='Ultrasound image (TIFF, PNG, JPG)', file_types=['image', '.tif', '.tiff'])
                preview = gr.Image(label='Scan: to calibrate, click the top and the bottom mark of the depth ruler', interactive=False, format='png')
                note = gr.Markdown('Upload a scan to begin.')
                with gr.Row():
                    calibrate = gr.Checkbox(label='Convert to mm using my two ruler points')
                    distance = gr.Number(label='Real distance between the two marks (mm), e.g. full depth 45', value=10, minimum=0.001)
                    with gr.Column(min_width=160):
                        find = gr.Button('Find ruler automatically')
                        clear = gr.Button('Clear points')
                orientation = gr.Radio(list(ORIENTATION), value='Auto (try both)', label='Orientation')
                crop = gr.Radio(list(CROP), value='Auto (remove black borders)', label='Region')
                run = gr.Button('Analyse', variant='primary')
            with gr.Column():
                overlay = gr.Image(label='Result', interactive=False, format='png')
                status = gr.Markdown()
                table = gr.Dataframe(headers=['Measurement', 'Pixels', 'Millimetres'], interactive=False)
                files = gr.File(label='Download results', file_count='multiple')
        upload.change(on_upload, upload, [preview, path, points, note])
        upload.change(lambda: False, None, auto)
        preview.select(on_click, [path, points], [preview, points, note]).then(lambda: False, None, auto)
        clear.click(on_clear, path, [preview, points, note]).then(lambda: False, None, auto)
        find.click(on_auto, path, [preview, points, auto, calibrate, note])
        run.click(analyse, [path, orientation, crop, calibrate, distance, points, auto], [overlay, status, table, files])
    return demo


def main():
    STATE['geometry'], STATE['models'] = ra.load_geometry(), ra.load_models()
    build().launch(server_name='127.0.0.1', share=False, inbrowser=True, allowed_paths=[str(RUNS)])


if __name__ == '__main__':
    main()
