# AGENTS.md — CVD Risk Predictor: Project Context & Ground Truth

> This file is the single source of truth for this project.
> Every decision recorded here was made deliberately and should not be overridden without explicit instruction.
> If you are an AI assistant, read this entire file before generating any code, suggestions, or analysis.

---

## 1. PROJECT IDENTITY

**Project name:** Cardiovascular Risk Predictor — Fairness-Aware, Uncertainty-Quantified ML System

**Research question:**
> "How reliable, fair, and generalisable are ML-based cardiovascular risk models when evaluated beyond their training distribution?"

**Paper title (working):**
> "Fairness-Aware, Uncertainty-Quantified Cardiovascular Risk Prediction with Cross-Dataset Generalisation Analysis and Lifestyle Feature Integration"

**What this project is NOT:**
- Not a Kaggle tutorial
- Not a basic UCI Cleveland classifier
- Not a demo with accuracy claims and no analysis
- Not a project that adds features for impressiveness without research value

---

## 2. RESEARCH CONTRIBUTIONS (EXACTLY FOUR — NO MORE)

These are the four contributions the paper makes. Every line of code serves one of these.

### Contribution 1 — Cross-Dataset Generalisation Study (PRIMARY)
- Train on Framingham dataset
- Evaluate frozen (no retraining) on Cleveland dataset
- Measure AUC degradation, Precision/Recall drop, Brier score degradation
- Use SHAP to identify which features are responsible for performance drop
- **The finding:** a specific, quantified AUC drop with a named feature driver

### Contribution 2 — Conformal Prediction for Uncertainty Quantification (SECONDARY)
- Wrap calibrated model using MAPIE library
- Output prediction intervals instead of single probabilities
- Example output: "Risk: 72%, Confidence Interval: 61–83% at 90% confidence"
- Analyse interval width by demographic subgroup
- **The finding:** confidence intervals are systematically wider for underrepresented groups — a quantified form of algorithmic inequity

### Contribution 3 — Calibration Layer (METHODOLOGICAL RIGOUR)
- XGBoost is poorly calibrated out of the box — raw probabilities are not trustworthy
- Apply Platt Scaling or Isotonic Regression on calibration set
- Evaluate with calibration curve + Brier score
- Calibration must happen BEFORE conformal prediction — this order is non-negotiable
- **The finding:** calibrated vs uncalibrated probability comparison, Brier score improvement

### Contribution 4 — Lifestyle Risk Score (LRS) as Engineered Feature
- Composite feature built from: sleep regularity, physical activity, smoking pack-years, sedentary hours per day, alcohol units per week
- Each component normalised to 0–1 scale
- Weights either equal (state this explicitly) or derived from Framingham Risk Score literature (preferred)
- Added to model feature set; evaluate whether it improves AUC
- Include LRS in SHAP analysis
- **The finding:** does lifestyle data add predictive signal beyond clinical features?

---

## 3. DECISIONS MADE — DO NOT REVISIT UNLESS EXPLICITLY TOLD TO

| Decision | Chosen | Reason |
|---|---|---|
| Primary dataset | Framingham | Larger, better-labelled |
| Validation dataset | Cleveland UCI | Clean, well-documented |
| Third dataset (BRFSS) | DROPPED from core | Feature alignment cost too high; mention as future work only |
| Frontend | React + Tailwind (or Streamlit if time is short) | React for portfolio value; Streamlit if Week 7 is tight |
| Backend | FastAPI + Uvicorn | Industry standard, async, typed |
| Experiment tracking | Weights & Biases (MANDATORY, not optional) | Reproducibility, interview credibility |
| Federated learning | CONDITIONAL — only if cross-dataset results are non-obvious | Synthetic simulation on small data adds little unless findings are unexpected |
| Dual-risk architecture | DROPPED | Requires longitudinal labelled data not available |
| DiCE counterfactuals | DROPPED | UI feature without clinical validation = weak contribution |
| Physiological age feature | DROPPED | Rebranding of existing Framingham Risk Score logic |
| Model complexity | XGBoost + Logistic Regression only | Model is the instrument, not the contribution |

---

## 4. DATA SPLIT — FIXED, NON-NEGOTIABLE

```
60% → Training set      (model training only)
20% → Calibration set   (Platt/Isotonic calibration only)
20% → Holdout set       (final evaluation only — never touch until final reporting)
```

**Rules:**
- Fixed random seed (set once, log in W&B, never change)
- Stratified split on target variable
- The holdout set is NEVER used for any intermediate decision
- Conformal prediction coverage guarantees are only valid on data the calibrator has never seen
- Calibrating and evaluating on the same split silently invalidates uncertainty results — this is a data leakage trap

---

## 5. MODELING PIPELINE — CORRECT ORDER

```
Step 1: Train model on training set (XGBoost, Logistic Regression)
Step 2: Evaluate baseline — AUC, Precision, Recall on calibration set
Step 3: Calibrate model on calibration set (Platt Scaling or Isotonic Regression)
Step 4: Evaluate calibration quality — calibration curve + Brier score
Step 5: Apply Conformal Prediction (MAPIE) on calibrated model
Step 6: Final evaluation on holdout — AUC, Brier, calibration curve, interval coverage, interval width
Step 7: Cross-dataset evaluation — apply frozen trained model to Cleveland, measure degradation
Step 8: SHAP analysis — per dataset, per demographic group, SHAP stability across bootstrap resamples
```

**Why this order matters:** Applying conformal prediction before calibration produces mathematically inconsistent uncertainty intervals. The calibration step corrects the base probability estimates that MAPIE's interval construction depends on.

---

## 6. EVALUATION METRICS — ALL ARE MANDATORY

| Metric | Purpose |
|---|---|
| ROC-AUC | Discrimination ability |
| Precision / Recall | Class-level performance |
| Brier Score | Quantitative calibration quality |
| Calibration curve | Visual calibration quality |
| Conformal coverage | Whether intervals contain the true outcome at stated confidence |
| Interval width | Proxy for model uncertainty per patient / demographic group |
| SHAP stability | Rank variance of top-5 features across 10 bootstrap resamples of holdout |

**SHAP stability definition (use this exactly):**
Run SHAP analysis 10 times on bootstrapped samples of the holdout set. Report the rank variance of the top-5 features. High variance = unstable explanations = clinically untrustworthy. Low variance = robust explanations.

---

## 7. LIFESTYLE RISK SCORE (LRS) — SPECIFICATION

```python
LRS = weighted_sum(
    sleep_regularity_score,   # 0–1, derived from sleep duration variance
    activity_score,            # 0–1, derived from weekly MET hours
    smoking_score,             # 0–1, derived from pack-years
    sedentary_score,           # 0–1, derived from sedentary hours/day
    alcohol_score              # 0–1, derived from units/week
)
```

- Each component normalised independently to 0–1 before weighting
- Weights must be justified: use equal weights (0.2 each) OR cite Framingham Risk Score literature for smoking/activity weights
- Document weight choice in paper — arbitrary weights with no justification is a reviewer rejection
- LRS is one feature fed into the model alongside standard clinical features
- Evaluate LRS contribution via SHAP — does it appear in top features?

---

## 8. TECH STACK — LOCKED

### Frontend
- **Primary:** React.js + Tailwind CSS
- **Fallback (if Week 7 is short on time):** Streamlit
- Charts: Plotly.js or Recharts
- Decision rule: if you haven't built React + FastAPI before, use Streamlit and spend saved time on the paper

### Backend
- Python 3.10+
- FastAPI
- Uvicorn (ASGI server)
- Pydantic (request validation)

### ML Stack
- scikit-learn
- XGBoost
- NumPy, Pandas
- SHAP (explainability)
- MAPIE (conformal prediction)
- imbalanced-learn / SMOTE (data balancing if needed)

### Experiment Tracking
- Weights & Biases (W&B) — mandatory
- Log: dataset name, split seed, model version, all metrics, all plots

### Deployment
- Backend: Docker → Render / Railway / AWS EC2
- Frontend: Vercel / Netlify

---

## 9. PROJECT STRUCTURE

```
project/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── framingham_clean.csv
│       └── cleveland_clean.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_training.ipynb
│   ├── 04_cross_dataset_eval.ipynb
│   ├── 05_calibration.ipynb
│   ├── 06_conformal.ipynb
│   └── 07_shap_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── explainability/
│   │   └── shap_analysis.py
│   ├── uncertainty/
│   │   ├── calibration.py
│   │   └── conformal.py
│   └── utils/
│       └── metrics.py
│
├── api/
│   ├── main.py
│   ├── routes.py
│   └── schemas.py
│
├── frontend/
│   └── (React or Streamlit app)
│
├── models/
│   ├── model.pkl
│   ├── calibrator.pkl
│   └── shap_explainer.pkl
│
├── requirements.txt
├── README.md
└── AGENTS.md          ← this file
```

---

## 10. API SPECIFICATION

### POST /predict
```json
Input:
{
  "age": 55,
  "sex": 1,
  "cholesterol": 220,
  "blood_pressure": 140,
  "smoking": 1,
  "diabetes": 0,
  "lrs": 0.62
}

Output:
{
  "risk": 0.72,
  "lower_bound": 0.61,
  "upper_bound": 0.83,
  "confidence_level": 0.90
}
```

### POST /explain
```json
Output:
{
  "top_features": ["cholesterol", "age", "lrs", "blood_pressure", "smoking"],
  "shap_values": { "cholesterol": 0.18, "age": 0.14, ... }
}
```

### GET /health
```json
{ "status": "ok" }
```

---

## 11. FRONTEND — MINIMUM REQUIRED SCREENS

1. **Input form** — sliders + dropdowns for all features including LRS components
2. **Risk dashboard** — risk probability, confidence interval visualised as a range bar
3. **Explanation panel** — SHAP waterfall or bar chart, top contributing features in plain English
4. **Dataset comparison view** — shows AUC and interval width across Framingham vs Cleveland (research result visualised)

---

## 12. EXECUTION TIMELINE

| Week | Focus |
|---|---|
| 1–2 | Data cleaning, feature alignment, LRS engineering |
| 3 | Model training, baseline evaluation, W&B setup |
| 4 | Calibration (Platt/Isotonic), Brier score, calibration curve |
| 5 | Conformal prediction (MAPIE), interval analysis by demographic |
| 6 | Cross-dataset evaluation, SHAP analysis, SHAP stability |
| 7 | FastAPI backend |
| 8 | Frontend + polish + README |

**Note on Week 7–8:** Backend + frontend in two weeks is tight. Decide on React vs Streamlit by end of Week 6. A clean Streamlit UI is better than a broken React one.

---

## 13. PAPER STRUCTURE

1. **Introduction** — why generalisation failure in CVD models is a clinical deployment problem, not just an academic one
2. **Related Work** — existing UCI-based models, their limitations (small data, single dataset, no calibration, no uncertainty)
3. **Methodology** — ensemble baseline, LRS feature engineering, three-way data split, calibration pipeline, conformal prediction, cross-dataset evaluation protocol, SHAP stability analysis
4. **Results** — four tables/figures: (a) AUC across datasets, (b) calibration curves + Brier scores, (c) confidence interval width by demographic group, (d) SHAP feature attribution comparison across datasets
5. **Discussion** — what your numbers mean for real-world deployment; what a clinician should trust and what they shouldn't
6. **Conclusion + Future Work** — mention BRFSS as future external validation, federated learning as future privacy-preserving extension

---

## 14. KNOWN FAILURE POINTS — WATCH THESE

| Risk | Mitigation |
|---|---|
| Data leakage via calibration/holdout overlap | Three-way split, fixed seed, holdout never touched until final eval |
| Uncalibrated probabilities fed into MAPIE | Always calibrate before applying conformal prediction |
| SHAP instability misread as model instability | Run bootstrap stability check, report rank variance explicitly |
| LRS weights unjustified | Cite Framingham literature or explicitly state equal weights with rationale |
| Feature misalignment between Framingham and Cleveland | Align features in preprocessing, document dropped features |
| UI built before science is done | Frontend is Week 7. Not before. |
| Overclaiming causality | The model finds associations. It does not identify causes. Every result claim must use associative language. |
| W&B skipped to save time | Non-negotiable. Set it up in Week 3 alongside first training run. |

---

## 15. WHAT THIS PROJECT IS NOT ALLOWED TO BECOME

- A project that reports 98% accuracy on UCI Cleveland and stops there
- A project with seven shallow features instead of four deep ones
- A project where the frontend is more polished than the analysis
- A project that adds federated learning without a motivated finding
- A project where calibration is an afterthought bolted on at the end
- A project where SHAP plots are included but never interpreted

---

## 16. ONE-SENTENCE SUMMARY FOR INTERVIEWS

> "I built a cardiovascular risk prediction system that quantifies not just risk but the model's uncertainty and fairness gaps — specifically, how confidence intervals widen for underrepresented demographic groups, and which features drive performance degradation when the model is tested outside its training distribution."

---

*Last updated: based on full planning conversation prior to execution start.*
*Do not modify this file without updating the relevant section and noting the reason.*
