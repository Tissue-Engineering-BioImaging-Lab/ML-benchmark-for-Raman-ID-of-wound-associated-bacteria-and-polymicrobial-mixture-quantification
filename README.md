# ML Benchmark for Species Identification and Polymicrobial Mixture Quantification

Machine-learning benchmark for identifying wound-associated bacterial species and classifying two-species mixtures from **Raman spectra**. This repository accompanies the paper *An open-source machine-learning benchmark for Raman species identification of wound-associated bacteria and polymicrobial mixture quantification* (Kunchur et al.). It contains the spectral data at every processing stage, the preprocessing and exploratory notebooks, the classifier notebooks with their hyperparameter searches, the script that produces every number in the paper, and the 16 trained models.

## Tasks

| Task | Data | Classes |
| --- | --- | --- |
| **Single-species identification** | 2,376 spectra | Six species: Sa, Cs, Pa, Ec, Ef, Sm |
| **Mixture classification** | 61,754 spectra | Sa : Pa at 1:9, 3:7, 5:5, 7:3 and 9:1 |

| Species | Abbreviation | Label in file names |
| --- | --- | --- |
| *Staphylococcus aureus* | Sa | `STAPH`, `MSSA`, `S` |
| *Corynebacterium striatum* | Cs | `Corn` |
| *Pseudomonas aeruginosa* (PAO1) | Pa | `PAO1`, `PAO`, `P` |
| *Escherichia coli* | Ec | `E.coli` |
| *Enterococcus faecalis* | Ef | `FAE` |
| *Stenotrophomonas maltophilia* | Sm | `Malt` |

The code and file names use the labels in the right-hand column, so those are what you'll see in outputs and plots.

## Results

Test-set accuracy of the four classifiers (one training run each, seed 42), all evaluated on the same held-out spectra (80/20 stratified split, `random_state=42`). Noise is Gaussian white noise at 5% of each spectrum's standard deviation, added after the train/test split. "Noise-augmented" models were trained on the training spectra plus a noisy copy of them.

**Mixture classification** (12,351 test spectra)

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

All four models are largely insensitive to the added noise (−0.5 to +1.1 percentage points), and noise-augmented retraining changes accuracy by −1.1 to +1.3 points. On the mixture task, Cochran's Q is significant in every scenario (p < 0.001) and most pairwise McNemar tests remain significant after Holm correction. Spectra from the same raster scan are not independent, however: when accuracy differences are bootstrapped over the 18 acquisition files, only 2–3 of the six model pairs per scenario have 95% confidence intervals that exclude zero. Full tables are in `model_comparison/results/`.

## Repository structure

```
├── data/                          All spectral data, at every processing stage (see Data)
├── Raman_Preprocessing.ipynb      Cropping, outlier removal and normalization
├── general_info.ipynb             Dataset summary: spectrum counts, mean spectra
├── combine_csvs_by_prefix.py      Stack CSVs of the same species into one file
│
├── Distributable Notebooks/       Self-contained notebooks + the data the models read
│   ├── data/                      Normalized spectra, one file per replicate (read by all models)
│   ├── raw_data/                  Original acquisitions for the mixed-ratio data
│   ├── New Notebooks/             Classifier notebooks (hyperparameter searches) and the PCA, ICA, SVD and ZCA notebooks
│   └── README.ipynb               Walk-through of every notebook
│
├── model_comparison/              pipeline.py: produces every number in the paper; predictions, results, logs
├── models/                        The 16 trained models reported in the paper
├── docker/                        GPU image for the CNN and Transformer notebooks
└── requirements.txt               Package versions used for the paper
```

## Reproducing the paper

```bash
git clone <this repository>
cd <repository folder>
pip install -r requirements.txt
cd model_comparison
python pipeline.py --check-only --allow-cpu   # checks data, split and noisy test set (~2 min)
python pipeline.py --allow-cpu                # full run; resumable
```

Requires Python 3.10+. Before training, `pipeline.py` checks the data, the train/test split and the noisy test set against fixed fingerprints and stops if anything differs. The full run with cross-validation takes many hours on CPU; without `--allow-cpu` it requires a GPU. See `model_comparison/README.md` for the protocol, the statistics and the output files.

To use the trained models without retraining, see `models/README.md`:

```python
import sys; sys.path.insert(0, "model_comparison")
from pipeline import load_model
predict = load_model("mixed", "cnn", "clean")   # returns class indices for min-max-normalized spectra
```

## Data

Every CSV uses the same layout: **the first row is the Raman shift (wavenumber) axis**, and each row after it is one spectrum. Class labels are taken from the file name (for example, `1-9_S-P_2.csv` becomes `1:9 | Staph : PAO`).

The models read `Distributable Notebooks/data/`: 18 mixed-ratio files (one per mixing ratio and replicate) and six single-species files (one per species). `data/` holds the same normalized spectra split into more files, plus the earlier processing stages:

| Stage | Mixed ratios | Six single species |
| --- | --- | --- |
| **Raw** | `data/raw/mixed_ratio/` | `Raw <species> csv/` |
| **Cropped** to 400–2000 cm⁻¹ | `data/fingerprint_region/mixed_ratio/` | `<species> Filtered csv - fingerprint region/` |
| **Normalized** | `data/mixed_ratio/` | `data/six_single_species/`, and `<species> Filtered, outlier corrected and normalized/` |

The single-species stages are in `data/Single Species Bacterial Colonies/`, one subfolder per species (`Corn`, `E. Coli`, `FAE`, `MSSA`, `Malt`, `PAO`).

Two caveats about the mixed-ratio data:

- The 20 `g_mixed_*` files in `data/mixed_ratio/` are available only in normalized form.
- The three `rep1_single_*` files in `data/raw/mixed_ratio/` are pure Sa and Pa controls recorded alongside the mixtures. They are not used by the models.

### Preprocessing

`Raman_Preprocessing.ipynb` turns raw spectra into normalized ones:

1. Crop each spectrum to **400–2000 cm⁻¹**. Reads `data/raw/mixed_ratio/` and writes `data/fingerprint_region/mixed_ratio/`.
2. Replace outliers per wavenumber using the IQR method (1.5 × IQR).
3. Apply min–max normalization to each spectrum.

Steps 2 and 3 write to `data/fingerprint_region/mixed_ratio/Outlier Removed and Normalized/`.

## Notebooks

`Distributable Notebooks/New Notebooks/` has one notebook per classifier and task (`RF`, `XG`, `CNN`, `Transformer` × `mixed`, `six_single`), plus `*_combined` variants for the noise-augmented models, and the four decomposition notebooks (PCA, ICA, SVD, ZCA) behind the exploratory analysis in the paper. `Distributable Notebooks/README.ipynb` walks through each one.

The classifier notebooks are the record of the original Optuna hyperparameter searches; the tuned values are listed in Supplementary Tables 2 and 4 of the paper and are reused by `model_comparison/pipeline.py`. Their outputs have been cleared, because their test-set results are superseded by the pipeline (see [Changes](#changes-from-the-original-analysis)).

**GPU training.** TensorFlow no longer supports GPUs on native Windows, so the CNN and Transformer notebooks run inside an NVIDIA TensorFlow Docker image. Build it once, then run a notebook headlessly from PowerShell:

```powershell
cd docker
docker build -t raman-cnn-gpu -f Dockerfile.gpu .
cd "..\Distributable Notebooks"
.\train_gpu_headless.ps1 CNN_mixed.ipynb
```

The Random Forest, XGBoost and decomposition notebooks run on CPU without Docker.

## Changes from the original analysis

The results above replace those of an earlier version of this analysis (repository `Tissue-Engineering-BioImaging-Lab/ML-Benchmark-for-Species-Identification-and-Polymicrobial-Mixture-Quantification`), after three problems were found:

1. **Missing values in the mixed-ratio data.** The earlier Random Forest and XGBoost notebooks combined the mixed-ratio CSVs without `join="inner"`. Files with slightly different wavenumber grids produced columns of missing values, which the noise function spread across whole spectra, so these models appeared to collapse to about 40% on noisy spectra. The notebooks in `Distributable Notebooks/` were not affected.
2. **Noisy copies of test spectra in training.** The `*_combined` (noise-augmented) notebooks add noise before the train/test split, so about 80% of noisy test spectra had their clean twin in the training set. The pipeline adds noise only after the split.
3. **Scaler fitted on all spectra.** The CNN and Transformer notebooks fit `StandardScaler` on all spectra before the split. The pipeline fits it on the training spectra only.

The pipeline also reports training accuracy and 5-fold cross-validation for every model, and replicate-level confidence intervals for the model comparisons.

## License

Released under the [MIT License](LICENSE).
