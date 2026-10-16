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

Every run requires a **new output directory**. `sample_manifest.json` contains the eight checked UMUD inputs and explicit horizontal orientation flags. For another image, create a JSON list:

```json
[{"path": "data/my_scan.tif", "flip": 0}]
```

`flip: 1` horizontally reverses the image for analysis. Select orientation by inspecting the scan; automatic orientation detection is not implemented. Optional `mm_per_pixel` must come from verified scan calibration, not TIFF DPI. The current conversion assumes the same pixel spacing on both axes.

## Outputs

- `measurements.csv` and `run.json`: results, status, runtime, settings, checkpoint hashes and source revision.
- Per image: `overlay.png` (when geometry is available), `lines.json`, `measurements.json`, two thresholded mask PNGs and `probabilities.npz`.
- Overlays, masks and line coordinates refer to the original image. Probability arrays retain the 512 × 512 analysis orientation; apply the recorded flip when using them elsewhere.
- `ok` means finite measurements were produced, not that their anatomical accuracy is established. Other statuses include `no_geometry`, `no_fascicles`, `invalid_geometry` and `error`. Missing results are null, not zero. Extrapolated line points may extend outside the image.

The default `reference` settings follow the upstream documented backup. `--profile current` uses the pinned source's current settings file, which differs from that reference. Model inference is performed once per image and model, followed by the unmodified upstream geometry routine. Physical calibration, manual ground-truth evaluation, a user interface and video integration remain separate work.
