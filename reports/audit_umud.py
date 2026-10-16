"""Read-only UMUD audit. Run: python3 reports/audit_umud.py [--self-check]."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from audit_examples import ROOT, inspect_image, self_check

BASE = ROOT / 'data/raw/umud'


def pair_rows(images, masks):
    images = {Path(r['path']).name: r for r in images}
    masks = {Path(r['path']).name: r for r in masks}
    missing_images = sorted(masks.keys() - images.keys())
    missing_masks = sorted(images.keys() - masks.keys())
    mismatches = []
    for name in images.keys() & masks.keys():
        a, b = images[name], masks[name]
        if (a.get('width'), a.get('height')) != (b.get('width'), b.get('height')):
            mismatches.append(name)
    return {'pairs': len(images.keys() & masks.keys()),
            'missing_images': missing_images, 'missing_masks': missing_masks,
            'dimension_mismatches': sorted(mismatches)}


def check():
    self_check()
    a = {'path': 'images/a.tif', 'width': 3, 'height': 2}
    b = {**a, 'path': 'masks/a.tif'}
    assert pair_rows([a], [b])['pairs'] == 1
    assert not pair_rows([a], [b])['dimension_mismatches']
    assert pair_rows([a], [{**b, 'height': 4}])['dimension_mismatches'] == ['a.tif']
    assert pair_rows([a], [])['missing_masks'] == ['a.tif']
    assert pair_rows([], [b])['missing_images'] == ['a.tif']
    print('Pairing self-check passed.')


def main():
    assert BASE.is_dir(), 'Extract the UMUD archive first.'
    rows, groups = [], defaultdict(list)
    byte_groups, pixel_groups = defaultdict(list), defaultdict(list)
    for path in sorted(p for p in BASE.rglob('*') if p.is_file()):
        rel = str(path.relative_to(BASE))
        with path.open('rb') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
            sha = digest.hexdigest()
        row = {'path': rel, 'bytes': path.stat().st_size, 'sha256': sha}
        byte_groups[sha].append(rel)
        if path.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
            try:
                row.update(inspect_image(path))
                pixel_groups[row['pixel_sha256']].append(rel)
            except Exception as error:
                row['error'] = str(error)
        groups[Path(rel).parts[0]].append(row)
        rows.append(row)
        if len(rows) % 1000 == 0:
            print(f'Inspected {len(rows)} files', flush=True)
    pairs = {kind: pair_rows([r for r in groups[f'{kind}_imgs_v1'] if 'width' in r],
                             [r for r in groups[f'{kind}_masks_v1'] if 'width' in r])
             for kind in ('apo', 'fasc')}
    summaries = {}
    for name, items in groups.items():
        summaries[name] = {
            'files': len(items), 'images': sum('width' in r for r in items),
            'dimensions': dict(Counter(f"{r['width']}x{r['height']}" for r in items if 'width' in r)),
            'modes': dict(Counter(r['mode'] for r in items if 'mode' in r)),
            'pixel_value_counts': dict(Counter(str(r['unique_values']) for r in items if 'unique_values' in r)),
            'resolution_tags': dict(Counter(json.dumps(r['resolution_tags'], sort_keys=True)
                                            for r in items if 'resolution_tags' in r)),
        }
    result = {'groups': summaries, 'pairs': pairs,
              'decode_errors': [r for r in rows if 'error' in r],
              'byte_duplicate_groups': [v for v in byte_groups.values() if len(v) > 1],
              'pixel_duplicate_groups': [v for v in pixel_groups.values() if len(v) > 1],
              'files': rows,
              'calibration_note': 'DPI is not confirmed anatomical scale. Verify ruler spacing before reporting mm.'}
    (ROOT / 'reports/umud_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    # Representative evenly spaced examples; this is a visual sample, not quality grading.
    panels = []
    for kind in ('apo', 'fasc'):
        items = [r for r in groups[f'{kind}_imgs_v1'] if 'width' in r]
        masks = {Path(r['path']).name: r for r in groups[f'{kind}_masks_v1']}
        for i in (0, len(items)//3, 2*len(items)//3, len(items)-1):
            row = items[i]
            panels.append(row)
            mask = masks.get(Path(row['path']).name)
            if mask and 'width' in mask:
                panels.append(mask)
    tests = [r for r in groups['test_images_v2'] if 'width' in r]
    panels += [tests[i] for i in (0, len(tests)//3, 2*len(tests)//3, len(tests)-1)]
    sheet = Image.new('RGB', (1280, 270*((len(panels)+3)//4)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(panels):
        x, y = (i % 4)*320, (i // 4)*270
        label = Path(row['path']).parts[0] + '/' + Path(row['path']).name
        draw.text((x+3, y+3), label, fill='black')
        with Image.open(BASE / row['path']) as source:
            thumb = source.convert('RGB'); thumb.thumbnail((312, 240))
            sheet.paste(thumb, (x+3, y+25))
    sheet.save(ROOT / 'reports/umud_contact_sheet.jpg')
    print(json.dumps({'pairs': {k: {a: len(b) if isinstance(b, list) else b
                                   for a, b in v.items()} for k, v in pairs.items()},
                      'decode_errors': len(result['decode_errors']),
                      'byte_duplicate_groups': len(result['byte_duplicate_groups']),
                      'pixel_duplicate_groups': len(result['pixel_duplicate_groups'])}, indent=2))


if __name__ == '__main__':
    check() if '--self-check' in sys.argv else main()
