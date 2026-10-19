"""Fine-tune the DL_Track_US aponeurosis and fascicle checkpoints on the UMUD training masks.

Kaggle (GPU): run fine_tune_kaggle.ipynb. Local smoke test: python finetune.py --smoke
Outputs <out>/<target>_finetuned.weights.h5 only when its validation Dice beats the pretrained
checkpoint, plus <out>/report.json. Weights-only files load into the same architecture with Keras 2.10.
Preprocessing matches run_analysis.to_batch: OpenCV BGR, float32, resize to 512, divide by 255.
"""
import argparse
import glob
import hashlib
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
try:  # Kaggle images ship Keras 3; the Keras-2 checkpoints need the legacy API
    import tf_keras as keras
except ImportError:
    keras = tf.keras

TARGETS = {  # images folder, masks folder, checkpoint filename
    'apo': ('apo_imgs_v1/apo_images_new_model_v1', 'apo_masks_v1/apo_masks_new_model_v1', 'model-apo-VGG16_DiceBCE_512_300.h5'),
    'fasc': ('fasc_imgs_v1/fasc_images_new_model_v1', 'fasc_masks_v1/fasc_masks_new_model_v1', 'model-fasc-VGG16_DiceBCE_200_512.h5'),
}
SIZE = 512


def pairs(data_root, target):
    images, masks, _ = TARGETS[target]
    found = {}
    for path in sorted(glob.glob(os.path.join(data_root, images, '*'))):
        mask = os.path.join(data_root, masks, os.path.basename(path))
        if path.lower().endswith(('.tif', '.tiff', '.png', '.jpg')) and os.path.exists(mask):
            found[os.path.basename(path)] = (path, mask)
    return found


def find_data_root(target):
    # Kaggle's mount path differs by account/version; pick the folder that yields the most pairs.
    first = TARGETS[target][0].split('/')[0]
    roots = {os.path.dirname(h) for h in glob.glob(f'/kaggle/input/**/{first}', recursive=True)}
    best = max(roots, key=lambda r: len(pairs(r, target)), default=None)
    if not best or not pairs(best, target):
        raise SystemExit(f'No {target} image/mask pairs under /kaggle/input; attach the UMUD competition data')
    return best


def load_pair(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.max(axis=2)  # palette/RGB masks: any non-zero channel is foreground
    x = cv2.resize(image.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR) / 255.0
    y = (cv2.resize((mask > 0).astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
    return x, y[..., None]


def dataset_split(found, val_fraction=0.15, block=20, seed=2026):
    """Deduplicate by image bytes, drop blank masks, split in blocks of consecutive names (shared scans/frames)."""
    seen, keep = set(), []
    for name, (image, mask) in sorted(found.items()):
        digest = hashlib.sha256(open(image, 'rb').read()).hexdigest()
        m = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
        if digest in seen or m is None or not m.any():
            continue
        seen.add(digest); keep.append(name)
    blocks = [keep[i:i + block] for i in range(0, len(keep), block)]
    random.Random(seed).shuffle(blocks)
    n_val = max(1, round(len(blocks) * val_fraction))
    val = sorted(n for b in blocks[:n_val] for n in b)
    train = sorted(n for b in blocks[n_val:] for n in b)
    return train, val


_CACHE = {}


def cached_pair(image_path, mask_path):
    # Decode each TIFF once per target; float16 keeps ~2k pairs under ~5 GB RAM.
    key = (image_path, mask_path)
    if key not in _CACHE:
        loaded = load_pair(image_path, mask_path)
        _CACHE[key] = None if loaded is None else tuple(a.astype(np.float16) for a in loaded)
    return _CACHE[key]


def tf_data(found, names, batch, augment):
    def gen():
        order = list(names)
        if augment:
            random.shuffle(order)
        for name in order:
            loaded = cached_pair(*found[name])
            if loaded is None:
                continue
            x, y = loaded
            if augment:
                if random.random() < 0.5:
                    x, y = x[:, ::-1], y[:, ::-1]
                x = np.clip(x * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08), 0, 1)
            yield x.astype(np.float32), y.astype(np.float32)
    spec = (tf.TensorSpec((SIZE, SIZE, 3), tf.float32), tf.TensorSpec((SIZE, SIZE, 1), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=spec).batch(batch).prefetch(2)


def dice_bce(y_true, y_pred, smooth=1.0):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    dice = (2 * inter + smooth) / (tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) + smooth)
    return tf.reduce_mean(bce) + 1 - tf.reduce_mean(dice)


def mean_dice(model, data, threshold):
    scores = []
    for x, y in data:
        p = (model.predict(x, verbose=0) > threshold).astype(np.float32)
        y = y.numpy()
        inter = (p * y).sum(axis=(1, 2, 3)); total = p.sum(axis=(1, 2, 3)) + y.sum(axis=(1, 2, 3))
        scores += list(np.where(total > 0, 2 * inter / np.maximum(total, 1e-9), 1.0))
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None, help='UMUD data root (searched under /kaggle/input by default)')
    parser.add_argument('--checkpoints', default=None, help='folder with the DL_Track .h5 files (searched under /kaggle/input by default)')
    parser.add_argument('--out', default='/kaggle/working/finetuned')
    parser.add_argument('--targets', default='apo,fasc')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--fasc-epochs', type=int, default=None, help='override --epochs for the fascicle model')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--smoke', action='store_true', help='tiny local run: 2 train / 2 val images, 1 epoch, batch 1')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    thresholds = {'apo': 0.2, 'fasc': 0.05}  # the runner's reference detection thresholds
    report = {'tensorflow': tf.__version__, 'keras': keras.__version__, 'settings': vars(args)}
    for target in args.targets.split(','):
        _CACHE.clear()
        name = TARGETS[target][2]
        ckpt = os.path.join(args.checkpoints, name) if args.checkpoints else (glob.glob(f'/kaggle/input/**/{name}', recursive=True) or [None])[0]
        if not ckpt or not os.path.exists(ckpt):
            raise SystemExit(f'Checkpoint {name} not found; attach the DL_Track checkpoints dataset')
        data_root = args.data or find_data_root(target)
        found = pairs(data_root, target)
        if len(found) < 2:
            raise SystemExit(f'Only {len(found)} {target} pairs found in {data_root}')
        print(f'{target}: data {data_root}', flush=True)
        train, val = dataset_split(found)
        if args.smoke:
            train, val, args.epochs, args.batch = train[:2], val[:2], 1, 1
        model = keras.models.load_model(ckpt, compile=False)
        val_data = tf_data(found, val, args.batch, augment=False)
        baseline = mean_dice(model, val_data, thresholds[target])
        print(f'{target}: {len(train)} train / {len(val)} val, pretrained val Dice {baseline:.4f}', flush=True)
        model.compile(optimizer=keras.optimizers.Adam(args.lr), loss=dice_bce)
        best, history = baseline, []
        epochs = args.fasc_epochs if target == 'fasc' and args.fasc_epochs else args.epochs
        for epoch in range(epochs):
            model.fit(tf_data(found, train, args.batch, augment=True), epochs=1, verbose=2)
            score = mean_dice(model, val_data, thresholds[target]); history.append(score)
            print(f'{target} epoch {epoch + 1}: val Dice {score:.4f} (best {best:.4f})', flush=True)
            if score > best:
                best = score
                model.save_weights(str(out / f'{target}_finetuned.weights.h5'))
        report[target] = {'checkpoint': name, 'train': len(train), 'val': len(val), 'pretrained_val_dice': baseline,
                          'val_dice_by_epoch': history, 'best_val_dice': best, 'saved': best > baseline}
        (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
