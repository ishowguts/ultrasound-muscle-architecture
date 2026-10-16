# Attribution and source provenance

Image segmentation checkpoints and the architecture measurement algorithm are from [DL_Track_US](https://github.com/PaulRitsche/DL_Track_US), by Paul Ritsche, Olivier Seynnes, Neil Cronin and collaborators. This project integrates their existing work; it does not introduce a new neural architecture.

The local runner loads `DL_Track_US/gui_helpers/do_calculations.py` from revision `b88df46320d822734fed2f1b711ccc2e8300efa1` without modifying it. Image preprocessing follows `importAndReshapeImage` in the same revision's `calculate_architecture.py`. Segmentation settings are taken from `_backup_settings.py`; the implemented linear extrapolation route is used. The numeric example references are transcribed from the authors' `docs/md_graphics/aia_test/results.png`. The source checkout retains its original Apache 2.0 `LICENSE` and `CITATION.cff`.

The two image checkpoints and example inputs come from the official [OSF project](https://osf.io/7mjsc/), v0.3.1 collection. The checkpoint folder is named models_v0.3.0. Source SHA-256 values are verified before loading.

The evaluation sample comes from the [UMUD Challenge](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/data). The data is provided under its stated CC BY-NC-SA 4.0 licence and competition rules. Obtain it through the official access process. Data, checkpoints and derived scan images are not included in this repository.

Consult the upstream citation file and the authors' [documentation](https://paulritsche.github.io/DL_Track_US/) when citing the method. This college project contributes the command-line integration, explicit orientation/coordinate handling, input validation, saved outputs, dataset audit and reproducibility checks.
