"""Audit downloaded public examples without changing originals.

Run: python3 reports/audit_examples.py
Check: python3 reports/audit_examples.py --self-check
Requires Pillow and NumPy, already available on the preparation computer.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def inspect_image(path):
    with Image.open(path) as image:
        image.load()
        pixels = np.asarray(image)
        decoded = np.asarray(image.convert('RGB')) if image.mode == 'P' else pixels
        signature = f'{decoded.shape}:{decoded.dtype}:'.encode() + decoded.tobytes()
        tags = getattr(image, 'tag_v2', {})
        return {
            'width': image.width, 'height': image.height, 'mode': image.mode,
            'frames': getattr(image, 'n_frames', 1),
            'min': int(pixels.min()), 'max': int(pixels.max()),
            'unique_values': int(np.unique(pixels).size),
            'pixel_sha256': hashlib.sha256(signature).hexdigest(),
            'resolution_tags': {str(k): str(tags[k]) for k in (282, 283, 296) if k in tags},
        }


def self_check():
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / 'a.tif'
        second = Path(directory) / 'b.png'
        pixels = np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8)
        Image.fromarray(pixels).save(first)
        Image.fromarray(pixels).save(second)
        a, b = inspect_image(first), inspect_image(second)
        assert (a['width'], a['height'], a['unique_values']) == (3, 2, 2)
        assert a['pixel_sha256'] == b['pixel_sha256'], 'Decoded duplicates must match across formats'
        pixels[0, 0] = 255
        Image.fromarray(pixels).save(second)
        assert inspect_image(second)['pixel_sha256'] != a['pixel_sha256']
        palette_image = Image.new('P', (3, 2))
        palette_image.putpalette([0, 0, 0] * 256)
        palette_image.save(first)
        black = inspect_image(first)['pixel_sha256']
        palette_image.putpalette([255, 255, 255] * 256)
        palette_image.save(first)
        assert inspect_image(first)['pixel_sha256'] != black, 'Palette colours affect decoded pixels'
    print('Self-check passed: dimensions, mask values, cross-format duplicates and changed pixels.')


def main():
    rows, byte_groups, pixel_groups = [], defaultdict(list), defaultdict(list)
    for path in sorted((ROOT / 'data/raw/dltrack_examples').rglob('*')):
        if not path.is_file() or path.name.endswith('.part'):
            continue
        relative = str(path.relative_to(ROOT))
        row = {'path': relative, 'bytes': path.stat().st_size,
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        byte_groups[row['sha256']].append(relative)
        if path.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
            row.update(inspect_image(path))
            pixel_groups[row['pixel_sha256']].append(relative)
        rows.append(row)
    pairs = []
    base = ROOT / 'data/raw/dltrack_examples'
    images = {p.name: p for p in (base / 'apo_images').glob('*.tif')}
    masks = {p.name: p for p in (base / 'apo_masks').glob('*.tif')}
    for name in sorted(images.keys() | masks.keys()):
        pair = {'name': name, 'image_present': name in images, 'mask_present': name in masks}
        if name in images and name in masks:
            with Image.open(images[name]) as image, Image.open(masks[name]) as mask:
                pair['same_dimensions'] = image.size == mask.size
        pairs.append(pair)
    result = {'scope': 'Official public DL_Track_US examples only; not the Kaggle dataset.',
              'files': rows, 'aponeurosis_pairs': pairs,
              'byte_duplicate_groups': [v for v in byte_groups.values() if len(v) > 1],
              'pixel_duplicate_groups': [v for v in pixel_groups.values() if len(v) > 1],
              'calibration_note': 'TIFF resolution/DPI tags alone do not establish ultrasound millimetres per pixel. Verify a physical scale or use known-distance calibration.'}
    (ROOT / 'reports/example_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    previews = [row for row in rows if 'width' in row]
    if previews:
        sheet = Image.new('RGB', (1000, 280 * ((len(previews) + 2) // 3)), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(previews):
            with Image.open(ROOT / row['path']) as image:
                thumb = image.convert('RGB'); thumb.thumbnail((320, 240))
                x, y = (index % 3) * 333, (index // 3) * 280
                sheet.paste(thumb, (x, y + 35))
                draw.text((x + 4, y + 4), '/'.join(Path(row['path']).parts[-2:]), fill='black')
        sheet.save(ROOT / 'reports/example_contact_sheet.jpg')
    print(json.dumps({'files': len(rows), 'pairs': pairs,
                      'byte_duplicates': result['byte_duplicate_groups'],
                      'pixel_duplicates': result['pixel_duplicate_groups']}, indent=2))


if __name__ == '__main__':
    self_check() if '--self-check' in sys.argv else main()
