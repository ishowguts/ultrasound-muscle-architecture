# Image baseline validation

Checked 16 October 2026 on an Apple Silicon Mac, CPython 3.10.21, TensorFlow/Keras 2.10, CPU. Dependencies are frozen in requirements-lock.txt. Both official checkpoints loaded and passed SHA-256 and input/output shape checks. Each network has 25,862,337 parameters. The upstream geometry source is pinned and checked for local modification before execution.

## Authors' examples

All four official automated-test inputs produced complete measurements. Their hashes match the OSF automated-test inventory. Reference values were transcribed from the pinned authors' [result image](https://github.com/PaulRitsche/DL_Track_US/blob/b88df46320d822734fed2f1b711ccc2e8300efa1/DL_Track_US/docs/md_graphics/aia_test/results.png). Each table cell shows **reference / reproduced**. Lengths are pixels, angles degrees.

| Image | Thickness | Fascicle length | Pennation |
|---|---:|---:|---:|
| 1.tif | 175.967 / 176.329 | 546.813 / 546.574 | 18.497 / 18.745 |
| 2.tif | 162.881 / 162.552 | 542.399 / 532.373 | 17.945 / 18.655 |
| 3.tif | 283.692 / 288.066 | 906.778 / 862.400 | 18.608 / 19.018 |
| 4.tif | 196.636 / 197.050 | 945.418 / 930.240 | 11.809 / 12.095 |

Maximum absolute deviations: 4.375 px thickness, 44.379 px fascicle length and 0.711 degrees pennation. These are similar results, not exact reproduction or independent anatomical accuracy. The historical output's full runtime/settings are not available; the cause of the differences has not been isolated. The default uses documented backup thresholds with the implemented linear extrapolation path, not the different current settings file.

## UMUD demonstration sample

Eight fixed images cover TIFF/PNG, several resolutions and both orientations. Six produced complete finite measurements; two retained explicit failure statuses. **6/8 is processing completion, not 75% accuracy.** Both models successfully decoded and inferred on all eight inputs.

| Image | Flip | Status | Retained fascicles | Seconds |
|---|---:|---|---:|---:|
| IMG_00001.tif | 0 | ok | 14 | 4.778 |
| IMG_00036.tif | 1 | ok | 21 | 4.493 |
| IMG_00040.tif | 1 | ok | 1 | 3.582 |
| IMG_00056.tif | 0 | ok | 15 | 4.023 |
| IMG_00104.tif | 0 | ok | 17 | 4.159 |
| IMG_00207.tif | 1 | ok | 11 | 4.421 |
| IMG_00252.png | 1 | invalid_geometry | 0 | 4.064 |
| IMG_00309.png | 1 | no_fascicles | 0 | 4.164 |

Runtime includes per-image inference, geometry and export, excluding initial imports/model loading. Orientation was explicitly adapted: the first pass succeeded on four images; trying the opposite orientation on the four failures recovered IMG_00040 and IMG_00207. Both orientations failed for IMG_00252 and IMG_00309. Thus this is a small adapted demonstration, not a blind held-out benchmark. The two PNG failures do not establish a PNG-format problem; decoding and model inference worked.

All eight saved overlays were visually inspected for rendering and restored orientation. This was not an expert anatomical assessment. IMG_00040 retains only one fascicle; extrapolation extends beyond the visible field in several other scans. Results require manual comparison before being treated as reliable measurements. Failure overlays clearly state that a complete measurement is unavailable. Upstream emits empty-mean warnings on no-fascicle cases; the adapter rejects invalid measurements rather than exporting NaN or the upstream 10000-distance sentinel.

## Runnable checks and limits

`test_analysis.py` checks BGR preprocessing/normalization, grayscale decoding, invalid inputs, horizontal coordinate restoration including extrapolated points, empty segmentation, and invalid geometry handling. `reports/audit_umud.py --self-check` checks image inspection and image/mask pairing. Both passed in the project environment. The final sample also exercised sentinel rejection end to end.

No checkpoint training or fine-tuning was performed. No physical scale was inferred from TIFF DPI. All demonstrated length values are pixels; calibrated millimetres are unvalidated. No Dice/IoU, manual measurement accuracy or clinical suitability is claimed. Full dataset evaluation, calibration, automatic orientation, an application interface and real video analysis are not completed by this baseline.

Local evidence directories are `outputs/examples_reference`, `outputs/umud_sample` (initial pre-guard run), `outputs/orientation_check` and `outputs/umud_sample_final` (final sample). Earlier runs are retained for comparison; use the final sample for demonstration. Scan images and raw outputs are excluded from the repository.
