# Model comparison

`pipeline.py` produces every number in Tables 1–2 and Supplementary Tables 1, 3 and 5–10 of the paper, from the data in
`Distributable Notebooks/data`. It trains Random Forest, XGBoost, a 1D CNN and a Transformer on both tasks under three
scenarios, then compares them on identical test spectra.

```bash
pip install -r ../requirements.txt
python pipeline.py --check-only --allow-cpu   # checks data, split and noisy test set (~2 min)
python pipeline.py --allow-cpu                # full run (many hours on CPU); resumable
python pipeline.py --save-models --allow-cpu  # retrain the 16 final models only and save them to ../models
```

Without `--allow-cpu` the script requires a GPU. Before training, it checks the data (shape, class counts, checksum,
no missing values), the train/test split and the noisy test set against fixed fingerprints, and stops if anything
differs. Finished models are skipped, so an interrupted run can be restarted with the same command.

## Protocol

* **Split:** 80/20 stratified, `random_state=42`.
* **Noise:** Gaussian white noise with standard deviation 5% of each spectrum's own standard deviation, added to the
  min–max-normalized spectra *after* the split and before any model-specific scaling (test seed 42, training
  augmentation seed 43). All four models see identical noisy test spectra.
* **Scenarios:** clean train / clean test; clean train / noisy test; noise-augmented train (training spectra plus a
  noisy copy of them) / noisy test.
* **Hyperparameters:** the tuned values from the notebooks' Optuna searches (Supplementary Tables 2 and 4), at full
  precision.
* **CNN and Transformer:** `StandardScaler` fitted on training spectra only; early stopping (patience 3, best weights
  restored) on the last 20% of the shuffled training split, held out together with its noisy copies.
* **Cross-validation:** 5-fold for every model. Random Forest and XGBoost use unshuffled stratified folds, as in the
  notebooks; the networks use shuffled folds (seed 42). Noise-augmented models are trained on each fold plus a noisy copy
  of it and evaluated on a noisy copy of the held-out fold.
* **Statistics (mixed-ratio task):** Cochran's Q overall and per mixing ratio; pairwise McNemar tests with Holm
  correction; and, because spectra from the same raster scan are not independent, a 95% confidence interval for each
  pairwise accuracy difference from a bootstrap that resamples the 18 acquisition files within each mixing ratio
  (2,000 resamples).
* **Robustness:** five extra random seeds for the single-species networks; the mixed-ratio networks re-evaluated with
  noise added after standardization.

## Hardware

The paper's numbers come from two CPU runs: Random Forest and XGBoost on x86-64 (scikit-learn 1.8, XGBoost 3.2), and the
CNN and Transformer on Apple silicon (TensorFlow 2.21, Keras 3.15). Random Forest reproduces the original notebook models
exactly. Tree models trained on Apple silicon differ by a few test spectra, because floating-point ties in split
selection resolve differently; neural networks differ slightly on any platform (see `results/seed_summary.json`).

## Files

| Path | Contents |
| --- | --- |
| `model_predictions_<task>_<scenario>.csv` | Per-spectrum predictions of all four models on identical test spectra, with the acquisition file (`source_file`) of each spectrum |
| `predictions/` | The same predictions as used by the script (`.npz`), with training accuracy and cross-validation per model (`.json`) |
| `results/accuracy_tables.csv` | Accuracy, macro precision/recall/F1, training accuracy and cross-validation for every model and scenario |
| `results/*_cochran.csv`, `results/*_mcnemar.csv` | Cochran's Q and pairwise McNemar tests with replicate-level confidence intervals |
| `results/seed_summary.json`, `results/noise_alt.json` | Seed variability and the noise-placement check |
| `logs/` | Logs of the two runs behind the paper (57 checks, 0 failed) |
