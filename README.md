# ML Benchmark for Raman Identification of Wound-Associated Bacteria and Polymicrobial Mixture Quantification

This repository accompanies the paper *An open-source machine-learning benchmark for Raman species identification of wound-associated bacteria and polymicrobial mixture quantification* (Kunchur et al.). It benchmarks four classifiers (Random Forest, XGBoost, a 1D convolutional neural network and a 1D Transformer) on Raman spectra of wound-associated bacteria, for two tasks: identifying single species and classifying the mixing ratio of two-species cultures.

The repository contains:

- the Raman spectra at every processing stage;
- the preprocessing and exploratory (PCA, ICA, SVD, ZCA) notebooks;
- the classifier notebooks with their hyperparameter searches;
- `model_comparison/pipeline.py`, which produces every result reported in the paper;
- the 16 trained models reported in the paper.

## Tasks

| Task | Spectra | Classes |
| --- | --- | --- |
| **Single-species identification** | 2,376 | Six species: Sa, Cs, Pa, Ec, Ef, Sm |
| **Mixing-ratio classification** | 61,754 | Sa : Pa at 1:9, 3:7, 5:5, 7:3 and 9:1 |

| Species | Abbreviation | Label in file names |
| --- | --- | --- |
| *Staphylococcus aureus* | Sa | `STAPH`, `MSSA`, `S` |
| *Corynebacterium striatum* | Cs | `Corn` |
| *Pseudomonas aeruginosa* (PAO1) | Pa | `PAO1`, `PAO`, `P` |
| *Escherichia coli* | Ec | `E.coli`, `E. Coli` |
| *Enterococcus faecalis* | Ef | `FAE` |
| *Stenotrophomonas maltophilia* | Sm | `Malt` |

File names, code and plots use the labels in the right-hand column.

## Results

Test-set accuracy of each classifier (one training run per model, random seed 42). All models were evaluated on the same held-out spectra (80/20 stratified split, `random_state=42`). Noise is Gaussian white noise with a standard deviation of 5% of each spectrum's own standard deviation, added after the train/test split. Noise-augmented models were trained on the training spectra plus a noisy copy of them.

**Mixing-ratio classification** (12,351 test spectra)

| Scenario | Random Forest | XGBoost | CNN | Transformer |
| --- | --- | --- | --- | --- |
| Clean train / clean test | 94.7% | 96.3% | **97.0%** | 95.4% |
| Clean train / noisy test | 94.2% | 95.8% | **96.8%** | 95.4% |
| Noise-augmented train / noisy test | 95.2% | 96.8% | **96.9%** | 95.5% |

**Single-species identification** (476 test spectra)

| Scenario | Random Forest | XGBoost | CNN | Transformer |
| --- | --- | --- | --- | --- |
| Clean train / clean test | 94.3% | 95.2% | **98.5%** | 89.7% |
| Clean train / noisy test | 94.1% | 96.2% | **98.5%** | 90.3% |
| Noise-augmented train / noisy test | 95.4% | 96.2% | **98.5%** | 89.3% |

Without retraining, the added noise changed accuracy by −0.5 to +1.1 percentage points. Noise-augmented retraining changed noisy-test accuracy by −1.1 to +1.3 percentage points.

On the mixing-ratio task, Cochran's Q was significant in every scenario (p < 0.001), and most pairwise McNemar tests remained significant after Holm correction. Spectra from the same raster scan are not independent, however. When accuracy differences were bootstrapped over the 18 acquisition files, only 2–3 of the six model pairs per scenario had 95% confidence intervals that excluded zero.

Training accuracy, 5-fold cross-validation, precision, recall, F1 and all statistical tables are in `model_comparison/results/`.

## Repository structure

```
├── data/                          Raman spectra at every processing stage (see Data)
├── Raman_Preprocessing.ipynb      Cropping, outlier correction and normalization
├── general_info.ipynb             Dataset summary: spectrum counts and mean spectra
├── combine_csvs_by_prefix.py      Stacks CSV files of the same species into one file
│
├── Distributable Notebooks/
│   ├── data/                      Normalized spectra read by the models (one file per replicate)
│   ├── raw_data/                  Raw acquisition files (.mat) for the mixed-ratio data
│   ├── New Notebooks/             Classifier notebooks and the PCA, ICA, SVD and ZCA notebooks
│   └── train_gpu_headless.ps1     Runs a notebook on the GPU (see Notebooks)
│
├── model_comparison/
│   ├── pipeline.py                Produces every result in the paper; also loads the trained models
│   ├── predictions/               Per-spectrum test predictions of the 16 models
│   ├── model_predictions_*.csv    The same predictions as tables, with the source file of each spectrum
│   ├── results/                   Accuracy, cross-validation and statistical tables
│   └── logs/                      Run logs and checks
│
├── models/                        The 16 trained models (see models/README.md)
├── docker/                        GPU image for the CNN and Transformer notebooks
└── requirements.txt               Package versions used for the paper
```

## Reproducing the results

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Tissue-Engineering-BioImaging-Lab/ML-benchmark-for-Raman-ID-of-wound-associated-bacteria-and-polymicrobial-mixture-quantification.git
cd ML-benchmark-for-Raman-ID-of-wound-associated-bacteria-and-polymicrobial-mixture-quantification
pip install -r requirements.txt
cd model_comparison
python pipeline.py --check-only --allow-cpu   # checks the data, split and noisy test set (about 2 minutes)
python pipeline.py --allow-cpu                # full run
```

Before training, `pipeline.py` checks the data, the train/test split and the noisy test set against fixed fingerprints, and stops if anything differs. The full run trains every model with 5-fold cross-validation and takes many hours on a CPU. Without `--allow-cpu`, the script requires a GPU. An interrupted run can be restarted with the same command; finished models are skipped. Results are written to `model_comparison/output/`. The evaluation protocol and statistical tests are described in the Methods section of the paper.

## Trained models

`models/` holds the 16 models reported in the paper: four classifiers × two tasks × two training sets (clean, and clean plus noise-augmented). Each model reproduces the paper's per-spectrum test predictions exactly. To use one:

```python
import sys; sys.path.insert(0, "model_comparison")
from pipeline import load_model

predict = load_model("mixed", "cnn", "clean")   # task, model, training set
labels = predict(X)                             # X: min–max-normalized spectra, one per row
```

`models/README.md` lists the files and explains the class labels and input format. To retrain and save the models yourself, run `python pipeline.py --save-models --allow-cpu`.

## Data

Every CSV file has the same layout: **the first row is the Raman shift (wavenumber) axis**, and each following row is one spectrum. The class label comes from the file name; for example, `1-9_S-P_2.csv` becomes `1:9 | Staph : PAO`.

The models read `Distributable Notebooks/data/`, which contains 18 mixed-ratio files (one per mixing ratio and replicate) and six single-species files (one per species). `data/` holds the same normalized spectra in smaller files, together with the raw and cropped stages:

| Stage | Mixed ratios | Six single species |
| --- | --- | --- |
| **Raw** | `data/raw/mixed_ratio/` | `Raw <species> csv/` |
| **Cropped** to 400–2000 cm⁻¹ | `data/fingerprint_region/mixed_ratio/` | `<species> Filtered csv - fingerprint region/` |
| **Normalized** | `data/mixed_ratio/` | `data/six_single_species/` and `<species> Filtered, outlier corrected and normalized/` |

The single-species stages are in `data/Single Species Bacterial Colonies/`, with one folder per species (`Corn`, `E. Coli`, `FAE`, `MSSA`, `Malt`, `PAO`).

Two notes on the mixed-ratio data:

- The 20 `g_mixed_*` files in `data/mixed_ratio/` are available only in normalized form.
- The three `rep1_single_*` files in `data/raw/mixed_ratio/` are pure Sa and Pa controls recorded alongside the mixtures. The classifiers do not use them.

### Preprocessing

`Raman_Preprocessing.ipynb` converts raw spectra into normalized spectra:

1. Crops each spectrum to **400–2000 cm⁻¹**, reading `data/raw/mixed_ratio/` and writing `data/fingerprint_region/mixed_ratio/`.
2. Replaces outliers in each wavenumber channel using the interquartile-range rule (1.5 × IQR).
3. Applies min–max normalization to each spectrum.

Steps 2 and 3 write to `data/fingerprint_region/mixed_ratio/Outlier Removed and Normalized/`.

## Notebooks

`Distributable Notebooks/New Notebooks/` contains one notebook per classifier and task (`RF`, `XG`, `CNN` and `Transformer`, each for `mixed` and `six_single`), a `*_combined` variant of each for the noise-augmented training set, and the PCA, ICA, SVD and ZCA notebooks behind the exploratory analysis.

The classifier notebooks contain the Optuna hyperparameter searches. Their tuned values are listed in Supplementary Tables 2 and 4 of the paper and are used by `model_comparison/pipeline.py`, which implements the evaluation reported in the paper. The notebooks are distributed with their outputs cleared; use `pipeline.py` to reproduce the paper's results.

**GPU training.** TensorFlow no longer supports GPUs on native Windows, so the CNN and Transformer notebooks run inside an NVIDIA TensorFlow Docker image. Build the image once, then run a notebook from PowerShell:

```powershell
cd docker
docker build -t raman-cnn-gpu -f Dockerfile.gpu .
cd "..\Distributable Notebooks"
.\train_gpu_headless.ps1 CNN_mixed.ipynb
```

The Random Forest, XGBoost and decomposition notebooks run on a CPU without Docker.

## Citation

If you use this dataset or code, please cite:

> Kunchur, N. N. et al. An open-source machine-learning benchmark for Raman species identification of wound-associated bacteria and polymicrobial mixture quantification (in submission).

## License

Released under the [MIT License](LICENSE).
