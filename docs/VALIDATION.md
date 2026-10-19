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

## Scan-field crop, automatic orientation and low confidence

Added 16 October 2026. Verified with an equivalent Linux aarch64 environment (CPython 3.10.21, TensorFlow/Keras 2.10, same pinned image packages); it reproduced the baseline author-example numbers exactly before the changes.

- **Crop.** By default the runner crops to the largest non-black region when that removes at least 25% of the frame, then maps lines and masks back to original pixels. The author examples are not cropped and their results are unchanged. An earlier version that also trimmed thin slivers moved example fascicle lengths away from the reference (1.tif 570.18 px, 2.tif 516.30 px), so mild trims are skipped.
- **Orientation.** With `flip: "auto"` both orientations are analysed and the better result is kept (usable status, then more fascicles). On the eight-image sample, the automatic choice matched the manually verified orientation on all six images that previously produced complete measurements. It roughly doubles runtime.
- **Low confidence.** Finite results from fewer than three retained fascicles are labelled `low_confidence` instead of `ok`.

| Image | Auto flip | Crop | Status | Fascicles | MT px | FL px | PA deg |
|---|---:|---|---|---:|---:|---:|---:|
| IMG_00001.tif | 0 | [200,77,1000,700] | ok | 23 | 340.99 | 1201.42 | 17.17 |
| IMG_00036.tif | 1 | full | ok | 21 | 394.16 | 1315.16 | 21.43 |
| IMG_00040.tif | 1 | full | low_confidence | 1 | 137.19 | 339.41 | 27.95 |
| IMG_00056.tif | 0 | full | ok | 15 | 358.47 | 1539.00 | 16.50 |
| IMG_00104.tif | 0 | full | ok | 17 | 408.29 | 1561.36 | 19.48 |
| IMG_00207.tif | 1 | [149,76,1048,706] | ok | 12 | 238.14 | 996.24 | 16.33 |
| IMG_00252.png | 0 | [324,108,827,667] | low_confidence | 2 | 139.49 | 459.13 | 20.58 |
| IMG_00309.png | 1 | [250,107,899,717] | low_confidence | 1 | 84.67 | 209.37 | 22.18 |

Cropping changed results on the two letterboxed TIFFs (IMG_00207 fascicle length 1149.49 to 996.24 px). Without reference measurements it is unknown which is closer to the truth; manual evaluation will compare cropped and uncropped analysis. The two PNG scans now produce geometry, but the overlays show the aponeurosis model selecting superficial fascia layers, so those measurements are wrong and are not counted as recovered. Gray interface panels are not removed by the automatic crop. This remains processing completion, not accuracy.

## Calibration spot check

Manual two-point calibration is applied on the full original image before cropping. For two scans the vertical depth ruler was read from the image: IMG_00001 ticks span 609 px for the labelled 4.5 cm depth (0.07389 mm/px) and IMG_00002 609 px for 4 cm (0.06568 mm/px). The scale was not fitted to any target values.

| Image | Crop | MT mm | FL mm | PA deg | sample_submission MT / FL / PA |
|---|---|---:|---:|---:|---|
| IMG_00001 | auto | 25.20 | 88.77 | 17.17 | 21.778 / 79.423 / 17.334 |
| IMG_00001 | none | 26.53 | 88.58 | 17.52 | |
| IMG_00002 | auto | 18.26 | 81.90 | 16.20 | 15.478 / 69.424 / 12.876 |
| IMG_00002 | none | 18.42 | 85.05 | 16.30 | |

**Correction (16 October 2026):** the competition host has stated that the two sample_submission values are invented examples, not measurements. The comparison columns above therefore carry no accuracy information; only the ruler-derived scale and the resulting millimetre values stand.

## Wider completion run (50 images)

`eval_manifest_50.json` holds 50 test images drawn with seed 2026 from the 301 test images not used in the demo sample (37 TIFF, 13 PNG). All ran with automatic crop and orientation and no calibration.

| Status | Count |
|---|---:|
| ok | 34 |
| low_confidence | 1 |
| no_fascicles | 10 |
| invalid_geometry | 3 |
| no_geometry | 2 |
| error | 0 |

**34/50 (68%) completed with three or more fascicles; this is processing completion, not accuracy.** PNG scans completed 8/13. Automatic cropping applied to 26 images. Among completed images the median pennation angle was 16.7° (13.1–27.7°). A non-expert look at the 34 completed overlays found most aponeurosis and fascicle lines plausible and about seven questionable, mainly small-field scans where the upper boundary may be a superficial layer. Runtime with both orientations was 11.5–16.1 s per image on a 4-core Linux virtual machine.

The first pass exposed an upstream geometry exception in one orientation of IMG_00232 that aborted the whole image. The runner now records a crashing orientation as a failed candidate and uses the other one; the image then completed as `no_fascicles`.

## Short video (offline)

The authors' video routine (image aponeurosis model plus the temporal IFSS fascicle model on three-frame stacks) ran unchanged on the public calf-raise recording (168 frames, 660 × 556, about 2.06 s) on the Apple Silicon Mac: 166 frames produced complete measurements and the first two frames filled the frame stack. Runtime was 698 s (4.2 s per frame) on CPU.

| Quantity (completed frames) | Min | Median | Max |
|---|---:|---:|---:|
| Fascicles per frame | 9 | 14 | 19 |
| Thickness (px) | 87.7 | 98.1 | 111.8 |
| Fascicle length (px) | 303.3 | 367.5 | 445.7 |
| Pennation (deg) | 19.0 | 22.4 | 27.2 |

Over the movement fascicle length falls from about 445 to 305 px while pennation rises from about 20° to 27°, the inverse relationship expected during contraction. This is a qualitative plausibility check only: there is no reference tracking for this clip and its physical scale was not verified, so values remain in pixels. The upstream routine's default settings crashed on this clip (an indexing error when two aponeurosis contours are closer than its 200 px minimum width), so the documented reference settings used for images were applied.

## Submission format and bounded geometry experiment (16 October 2026)

Kaggle rejected the semicolon-delimited baseline with `ID column image_id not found in submission`. The comma-delimited copy has 309 unique test IDs and four expected columns; the exporter now uses commas by default. Both contain identical predictions. The user reported a completed v2 submission with public score **0.56478** on 16 October 2026. This was supplied in pasted Kaggle submission details; it has not been independently fetched from Kaggle. The score is an error metric, not percentage accuracy.

Thirty baseline thickness measurements were below the published 10 mm lower bound. A single predefined experiment reran the unchanged upstream geometry on cached model probabilities, changing the minimum layer separation from 60 pixels to `ceil(10 / mm_per_pixel)`. The original results were reproduced exactly for all 30 cases and eight valid controls before applying the alternative. Eleven rejected scans then produced at least three fascicles and all three measurements within published ranges; eight controls were unchanged. The other 19 cases were not adopted.

The experimental candidate replaces only those 11 rows (33 values), keeping the other 298 rows and their existing fallback medians identical. It uses no manual challenge annotations or model retraining.

| Submission | All three values model-derived | PA fallback | FL fallback | MT fallback | Kaggle score |
|---|---:|---:|---:|---:|---|
| v2 baseline | 238 / 309 | 64 | 66 | 40 | 0.56478 (user-reported) |
| v3 width candidate | 249 / 309 | 57 | 57 | 29 | 0.59315 (user-reported) |

These are measurement availability counts, not accuracy. In-range values and multiple fascicles can still be anatomically incorrect. The candidate has not been promoted to the app or default runner. A lower Kaggle score is needed to support a challenge-performance improvement claim. Retrieval of the separate public expert benchmark failed, so no independent labelled MAE is available from this experiment.

The user subsequently reported v3 score **0.59315**, worse than v2 **0.56478** by 0.02837 (about 5.02% relative). The candidate is rejected for promotion; v2 remains the best measured submission. Additional in-range measurements did not improve public scoring. These aggregate scores do not identify per-target or per-image errors.

The [official metric notebook](https://www.kaggle.com/code/paulritsche/umud-score), Version 9, was read directly: primary score = `(MAE_PA/6 + MAE_FL/12 + MAE_MT/3)/3`, with PA in degrees and FL/MT in mm. Tiny normalized MedAE (1e-6) and RMSE (1e-9) terms break ties. This is a dimensionless error measure, not accuracy percentage.

## Thickness measurement fix (scores 16 October 2026, user-reported)

The pinned upstream thickness routine searches the lower aponeurosis with x-coordinates used as array indices, which shifts the search window and overestimates thickness when that aponeurosis starts away from the image edge. `measure.py` instead averages the perpendicular distance between the inner aponeurosis edges at 25%, 50% and 75% of their overlap, as the challenge annotators measured. On the 309 test images the change lowered thickness by 0.73 mm on average and by more than 1 mm on 74 images; statuses and all angle and length values were unchanged.

| Submission | Change | Public score (lower is better) |
|---|---|---:|
| v2 | baseline | 0.56478 |
| v4 | v2 + perpendicular three-location thickness | **0.50424** |
| v5 | v4 + 25 missing fascicle lengths estimated from thickness | 0.51176 |
| v6 | v4 + pennation from exact line geometry | 0.53209 |
| v7 | v6 + fascicle length from exact line intersections | 0.56731 |
| v8 | v4 + fascicle-guided aponeurosis pair on failed scans | **0.45242** |
| v9 | v8 with medians instead of 1–2-fascicle estimates | 0.46209 |
| v10 | v8 + video (IFSS) fascicle model on scans the image model missed | **0.44105** |
| v11b | fine-tuned models where they measured, v10 elsewhere | 0.47787 |
| v11a | fine-tuned models only | 0.52788 |

The thickness fix alone reduced the public error score by 0.06054 (10.7%). Estimating missing fascicle lengths from thickness made the score slightly worse and was not adopted. Recomputing pennation and fascicle length from exact line geometry (removing upstream's integer-truncated, unsigned aponeurosis angle and its grid-limited intersections) also made the score worse, so the annotators' conventions appear closer to the upstream measurements; those changes were not adopted. Public scores use about a third of the test images; the final ranking uses the rest.

## Fascicle-guided aponeurosis fallback

On many failed scans the upstream geometry paired the skin or subcutaneous fat layer with the superficial aponeurosis, leaving a thin band without fascicles (thickness under 10 mm). When the upstream result fails or its thickness is below the 10 mm test-set minimum, the runner now groups the aponeurosis-probability bands by depth, keeps the adjacent pair that encloses the most fascicle signal (the muscle belly) and passes only that pair to the unchanged upstream geometry. Re-running the 71 scans that previously needed median values (both orientations) used the fallback on 44 and produced complete measurements on 42. Complete measurements rose from 226 to 268 of 309 and median-filled angles and lengths fell from 64 to 27. The public score improved from 0.50424 to **0.45242**, a total reduction of 0.11236 (19.9%) from the v2 baseline.
