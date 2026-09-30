# Trained models

The 16 models reported in the paper: four classifiers × two tasks × two training sets. Each was trained once with random
seed 42 by `model_comparison/pipeline.py --save-models`, and each reproduces the paper's per-spectrum test predictions
exactly (`identical_to_paper_clean` / `identical_to_paper_noisy` in its `.json` file).

| File name part | Meaning |
| --- | --- |
| `single` / `mixed` | Six-species identification / Sa : Pa mixing-ratio classification |
| `rf`, `xg`, `cnn`, `tr` | Random Forest, XGBoost, 1D CNN, Transformer |
| `clean` / `retr` | Trained on clean spectra / on clean spectra plus a noise-augmented copy |

* **Random Forest and XGBoost:** `<name>.joblib`, the fitted scikit-learn / XGBoost estimator.
* **CNN and Transformer:** `<name>.weights.h5` (Keras 3 weights) and `<name>_scaler.joblib` (the `StandardScaler` fitted
  on the training spectra).
* **Every model:** `<name>.json`, with the class names in label order, the hyperparameters, test accuracy and the
  platform it was trained on.

## Loading a model

Use `load_model()` from `model_comparison/pipeline.py`. It rebuilds the network architecture, applies the scaler, and
returns a function that maps spectra to class indices:

```python
import sys, json
sys.path.insert(0, "model_comparison")
from pipeline import load_model

predict = load_model("mixed", "cnn", "clean")          # task, model, training set
labels = predict(X)                                    # X: n_spectra x n_features, min-max-normalized
classes = json.load(open("models/mixed_cnn_clean.json"))["classes"]
print([classes[i] for i in labels])
```

`X` must have the same features, in the same order, as the files in `Distributable Notebooks/data`
(388 wavenumbers for the mixed-ratio task, 389 for the single-species task), and each spectrum must be min–max-normalized
as described in `Raman_Preprocessing.ipynb`.
