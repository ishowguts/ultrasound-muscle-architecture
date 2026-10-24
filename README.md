# Ultrasound muscle architecture

A college project that runs pretrained DL_Track_US image models on ultrasound scans and saves aponeurosis/fascicle predictions, line overlays, muscle thickness, fascicle length and pennation angle. The contribution is integration and evaluation of an existing method; the networks are not trained from scratch here.

The image baseline runs locally on Apple Silicon. See [validation results](docs/VALIDATION.md) and [source attribution](ATTRIBUTION.md). Lengths are in **pixels** unless a verified physical calibration is supplied. This is an experimental project, not a clinically validated measurement tool.

## Setup

Use Python 3.10 on an Apple Silicon Mac. The validated runtime is Python 3.10.21, TensorFlow/Keras 2.10 on CPU. Other platforms have not been tested; `tensorflow-macos` is platform specific.

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
git clone https://github.com/PaulRitsche/DL_Track_US.git references/DL_Track_US
git -C references/DL_Track_US checkout b88df46320d822734fed2f1b711ccc2e8300efa1
mkdir -p models data/raw/dltrack_examples/images
```

Download the official checkpoints into `models/` with these exact filenames. The runner checks their SHA-256 hashes before loading:

- [Aponeurosis checkpoint](https://osf.io/download/6jzuy/): `model-apo-VGG16_DiceBCE_512_300.h5`
- [Fascicle checkpoint](https://osf.io/download/6fpkt/): `model-fasc-VGG16_DiceBCE_200_512.h5`
- [Video fascicle checkpoint (IFSS)](https://osf.io/download/n823q/): `model-fasc-IFSS_1Encoder_3d_a08_g2_noaug_nocrop_HL.h5` (video tool and `--ifss-fallback`)

Download the official test inputs into `data/raw/dltrack_examples/images/`: [1.tif](https://osf.io/download/frjzm/), [2.tif](https://osf.io/download/696d0e317e11a02234d8ea41/), [3.tif](https://osf.io/download/696d0e372ab52e112857b6b9/), [4.tif](https://osf.io/download/696d0e357e11a02234d8ea46/) and [flip_flags.txt](https://osf.io/download/zt5nd/).

Obtain UMUD separately from its [Kaggle Data page](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/data), following its access requirements and terms. Extract it into `data/raw/umud/`, so the test folder is `data/raw/umud/test_images_v2/test_set_v2/`. Data, weights and scan outputs are not stored in Git.

## Run

From the project directory:

```sh
.venv/bin/python test_analysis.py
.venv/bin/python reports/audit_umud.py --self-check
.venv/bin/python run_analysis.py --examples --output outputs/examples_run1
.venv/bin/python run_analysis.py --manifest sample_manifest.json --output outputs/umud_run1
```

Every run requires a **new output directory**.

### Kaggle submission (UMUD test set)

Best submission (v10, public score 0.44105), about 70 minutes for 309 images:

```sh
.venv/bin/python test_submission.py
.venv/bin/python run_analysis.py --manifest test_manifest_309.json --ifss-fallback --output outputs/test309_v10_reproduce
.venv/bin/python make_submission.py outputs/test309_v10_reproduce --recompute-thickness --output outputs/submission_v10_reproduce
```

Upload `submission.csv` from the last folder. What each step adds:

- Muscle thickness is the perpendicular distance between the two fitted aponeurosis lines at 25%, 50% and 75% of their overlap (`measure.py`), replacing the upstream loop that indexed the lower aponeurosis by x-coordinate values.
- When the default pair of aponeuroses yields no fascicles or a thickness under 10 mm, the runner retries with the adjacent pair of bands that encloses the most fascicle pixels (`apo_pair.py`); rows record `aponeurosis_pair`.
- `--ifss-fallback`: when both orientations still give no estimate, the runner retries with the video IFSS fascicle model on the single frame stacked three times; rows record `fascicle_model` (`image` or `ifss`).
- `--recompute-thickness` re-measures thickness from saved lines, so older runs can be exported with the same method.

Scores of every submission and what changed are in `docs/VALIDATION.md`.

`test_manifest_309.json` lists all 309 test images with automatic orientation, crop and ruler detection (`"calibration": {"method": "auto"}`). The ruler detector knows the layouts of the three test-set devices and reads the vertical scale (or the bottom 1 cm ticks on older scans), as the organisers specify. `make_submission.py` writes comma-separated `submission.csv` (`image_id,pa_deg,fl_mm,mt_mm`), an identical `submission_comma.csv` compatibility copy, and `details.csv` showing for every value whether it was measured, or filled with the median of the same scan layout (failed analysis, undetected ruler, or a value outside the label ranges published by the organisers, which signals a wrong aponeurosis or fascicle pick). The run is fully automatic; no test image is labelled by hand.

### Short video (offline)

```sh
.venv/bin/python test_video.py
.venv/bin/python run_video.py data/raw/dltrack_examples/video/video_calf_raise.mp4 --output outputs/video_run1
```

Uses the authors' video routine unchanged: the image aponeurosis model plus the temporal IFSS fascicle model on stacks of three frames. Orientation is chosen automatically on the first frame (`--flip 0/1` to override). Outputs `frames.csv` (per-frame status, thickness, fascicle length, pennation), `timeseries.png`, `annotated.mp4` and `run.json`. The first two analysed frames only fill the frame stack. `--step N` analyses every Nth frame, `--start`/`--frames` select a segment, and `--mm-per-pixel` applies a verified scale. Frames are not cropped.

### Web app

```sh
.venv/bin/python test_app.py
.venv/bin/python app.py
```

Opens `http://127.0.0.1:7860` (this computer only, no public link). Upload a scan, optionally tick calibration and click two marks on the scan's depth ruler with their real distance in millimetres, then **Analyse**. The result shows the overlay, status, measurements in pixels (and millimetres when calibrated), and CSV/JSON/overlay downloads. Each analysis is saved under `outputs/app_runs/`. Exported files contain the image file name, not local paths. `sample_manifest.json` contains the eight checked UMUD inputs and explicit horizontal orientation flags. For another image, create a JSON list:

```json
[{"path": "data/my_scan.tif"}]
```

- `flip`: `0`, `1` (horizontally reverse for analysis) or `"auto"` (default). Auto runs both orientations and keeps the better result: a usable status first, then more retained fascicles; ties keep `0`. It doubles inference time. Both candidates are recorded in `measurements.json`.
- `crop`: `"auto"` (default), `"none"` or `[x0, y0, x1, y1]` in original pixels. Auto finds the largest non-black region and crops to it only when that removes at least 25% of the frame, so letterboxed screenshots lose their black borders and UI text before resizing to 512 × 512. Grey interface panels are not removed; pass an explicit box for those. All exported coordinates and masks are mapped back to the full original image.
- `calibration` (optional), measured on the full original image before any crop:
  - `{"method": "manual", "points": [[x, y], [x, y]], "distance_mm": 10}`: two points on the vertical ruler and their known distance. Only the vertical pixel separation is used.
  - `{"method": "bar", "spacing_mm": 5, "confirmed": true}`: the pinned upstream scale-bar detector. Without `"confirmed": true` the detection is recorded as a candidate and no millimetres are produced. Confirm only after checking the detected marks and their spacing.
- `mm_per_pixel`: alternative to `calibration` for an already verified scale. Never derive it from TIFF DPI.

Millimetre values assume the same pixel spacing on both axes; raw pixel values are always kept.

## Outputs

- `measurements.csv` and `run.json`: results, status, runtime, settings, checkpoint hashes and source revision.
- Per image: `overlay.png` (when geometry is available), `lines.json`, `measurements.json`, two thresholded mask PNGs and `probabilities.npz`.
- Overlays, masks and line coordinates refer to the original image. Probability arrays retain the 512 × 512 analysis orientation; apply the recorded flip when using them elsewhere.
- `ok` means finite measurements were produced, not that their anatomical accuracy is established. `low_confidence` means finite measurements from fewer than 3 retained fascicles; numbers are exported but should not be trusted without review. Other statuses include `no_geometry`, `no_fascicles`, `invalid_geometry` and `error`. Missing results are null, not zero. Extrapolated line points may extend outside the image.

The default `reference` settings follow the upstream documented backup. `--profile current` uses the pinned source's current settings file, which differs from that reference. Model inference is performed once per image and model, followed by the unmodified upstream geometry routine. Physical calibration, manual ground-truth evaluation, a user interface and video integration remain separate work.

### Fine-tuning (tried, not adopted)

`fine_tune_kaggle.ipynb` (same code as `finetune.py`) fine-tunes both image checkpoints on the UMUD training masks on a Kaggle GPU and saves `<target>_finetuned.weights.h5` only when validation Dice beats the pretrained model. `run_analysis.py --weights-dir <folder>` loads such files. Validation Dice improved (aponeurosis 0.925 to 0.949, fascicle 0.410 to 0.512) but the public score got worse (0.478 combined with the v10 fallbacks, 0.528 alone, against 0.441), so submissions use the pretrained weights.

The Kaggle uploader requires commas: uploading a semicolon-separated file produces `ID column image_id not found in submission`. For the preserved v2 baseline, upload `outputs/submission_v2/submission_comma.csv`. New exports use commas by default. A successful upload and returned score are still required; local format checks do not measure accuracy.
