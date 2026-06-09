# Data — CVD Risk Predictor

## Dataset Sources

### Framingham Heart Study

- **URL:** <https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset>
- **File:** `data/raw/framingham.csv`
- **Records:** ~4,240 (before cleaning)
- **Target column:** `TenYearCHD` — binary flag indicating whether the patient developed coronary heart disease within 10 years of the examination (1 = developed CHD, 0 = did not).
- **Role:** Primary training dataset (AGENTS.md §3, Decision table).

### Cleveland Heart Disease (UCI)

- **URL:** <https://archive.ics.uci.edu/dataset/45/heart+disease>
- **File:** `data/raw/cleveland.csv`
- **Records:** ~303 (before cleaning)
- **Target column:** `target` — integer 0-4 indicating disease presence severity. **This is NOT a probability of heart attack.** It represents:
  - `0` = no disease
  - `1–4` = increasing presence of heart disease
- **Binarisation:** Values > 0 are mapped to 1 (disease present) during preprocessing.
- **Missing values:** The raw file uses `?` as the missing-value marker (columns `ca` and `thal` most affected). These are replaced with `NaN` and rows are dropped via listwise deletion.
- **Role:** External validation dataset — the trained model is evaluated on this dataset **without retraining** to measure cross-dataset generalisation (AGENTS.md §2, Contribution 1).

---

## Feature Alignment

During preprocessing, `align_features()` maps columns from both datasets to a shared schema. Only features present in **both** datasets are retained.

### Shared (Aligned) Features

| Aligned Name          | Framingham Column | Cleveland Column | Description                           |
|-----------------------|-------------------|------------------|---------------------------------------|
| `age`                 | `age`             | `age`            | Age in years                          |
| `sex`                 | `male`            | `sex`            | Biological sex (1 = male, 0 = female) |
| `cholesterol`         | `totChol`         | `chol`           | Total serum cholesterol (mg/dL)       |
| `systolic_bp`         | `sysBP`           | `trestbps`       | Resting systolic blood pressure       |
| `fasting_blood_sugar` | `diabetes`        | `fbs`            | Fasting blood sugar > 120 mg/dL       |

### Dropped Columns — Framingham

| Column           | Reason                                                                           |
|------------------|----------------------------------------------------------------------------------|
| `education`      | Socio-demographic variable, not a clinical risk factor; no Cleveland equivalent   |
| `currentSmoker`  | Binary smoking flag; Cleveland has no equivalent                                 |
| `cigsPerDay`     | Continuous smoking variable; no Cleveland equivalent. Captured by LRS instead     |
| `BPMeds`         | Blood-pressure medication flag; no Cleveland equivalent                          |
| `prevalentStroke`| Medical history flag; no Cleveland equivalent                                    |
| `prevalentHyp`   | Medical history flag; no Cleveland equivalent                                    |
| `diaBP`          | Diastolic BP; Cleveland only records resting (systolic-equivalent) `trestbps`    |
| `BMI`            | Body mass index; no Cleveland equivalent                                         |
| `glucose`        | Fasting glucose (continuous); Cleveland uses binary `fbs` instead                |

### Dropped Columns — Cleveland

| Column    | Reason                                                                       |
|-----------|------------------------------------------------------------------------------|
| `cp`      | Chest pain type; no Framingham equivalent                                    |
| `restecg` | Resting ECG result; no Framingham equivalent                                 |
| `thalach` | Maximum heart rate achieved; mapped to `max_heart_rate` but dropped — see ¹  |
| `exang`   | Exercise-induced angina; no Framingham equivalent                            |
| `oldpeak` | ST depression induced by exercise; no Framingham equivalent                  |
| `slope`   | Slope of peak exercise ST segment; no Framingham equivalent                  |
| `ca`      | Number of major vessels coloured by fluoroscopy; no Framingham equivalent     |
| `thal`    | Thalassemia type; no Framingham equivalent                                   |

> **¹ Note on `heartRate` / `thalach`:** Framingham's `heartRate` is resting heart rate, while Cleveland's `thalach` is *maximum* heart rate achieved during exercise. Despite both measuring heart rate, they are clinically distinct measurements and cannot be aligned. The `_FEATURE_MAP` in `preprocessing.py` maps them as a shared feature (`max_heart_rate`), but users should be aware of this semantic difference when interpreting cross-dataset results.

---

## Data Split

As mandated by AGENTS.md §4, the data split is **fixed and non-negotiable**:

| Partition    | Ratio | Purpose                                           |
|--------------|-------|---------------------------------------------------|
| Training     | 60%   | Model fitting only                                |
| Calibration  | 20%   | Platt Scaling / Isotonic Regression calibration    |
| Holdout      | 20%   | Final evaluation only — never touched until reporting |

- **Random seed:** `42` (set in `configs/model_config.yaml`, applied via `src/utils/seed.py`)
- **Stratified:** Split is stratified on the target variable to preserve class distribution
- **Holdout discipline:** The holdout set is **never** used for any intermediate decision. Conformal prediction coverage guarantees are only valid on data the calibrator has never seen.

---

## Scaler Policy

The `StandardScaler` is fitted on the **training partition only**.

**Why:** Fitting the scaler on the full dataset (including calibration and holdout) would leak distributional information (mean and variance) from evaluation partitions into the training pipeline. This is a subtle form of data leakage that artificially inflates performance metrics and invalidates the statistical guarantees of conformal prediction.

The fitted scaler is then applied (`.transform()` only) to the calibration and holdout sets, and to Cleveland data during cross-dataset evaluation.

---

## Directory Structure

```
data/
├── raw/
│   ├── framingham.csv      # Original Framingham dataset
│   ├── cleveland.csv        # Original Cleveland dataset
│   └── .env                 # Environment variables (not committed)
├── processed/
│   ├── framingham_clean.csv # After cleaning & alignment
│   └── cleveland_clean.csv  # After cleaning & alignment
└── README.md                # This file
```
