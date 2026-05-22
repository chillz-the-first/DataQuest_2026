# DataQuest 2026 — Building Interpretable Credit Models

A credit risk modelling project built around a single constraint: the final predictive model must be logistic regression. No black boxes. Every decision must be explainable.

## Project Summary

| Stage | AUC |
|-------|-----|
| Spec reference (old model) | 0.680 |
| Post-cleaning baseline | 0.789 |
| Final model (with feature engineering) | 0.792 |
| LightGBM benchmark (not used) | 0.820 |

Most of the improvement came from data quality, not modelling complexity. The gap between the cleaned baseline and the non-linear ceiling is 0.028 — small enough to justify the interpretability trade-off.

---

## Project Structure

```
DataQuest_2026/
├── app/
│   └── streamlit_app.py        # Interactive EDA, model evaluation, business dashboard
├── data/
│   └── loan_book.csv           # Raw dataset (120K applicants, 26 features)
├── notebooks/
│   ├── 01_data_quality.ipynb   # Initial data inspection
│   ├── 02_eda_findings.ipynb   # IV ranking, univariate/bivariate analysis
│   └── 03_modeling.ipynb       # Baseline, feature engineering, final model
├── outputs/
│   ├── loan_book_clean.csv     # Cleaned dataset
│   └── model_objects.pkl       # Saved models and scalers
├── src/
│   ├── cleaning.py             # Cleaning functions (normalise, cap, impute)
│   ├── features.py             # WoE/IV computation and plotting
│   └── modeling.py             # WoE encoding function
├── report/
│   ├── Interpretable_Credit_Modeling.docx   # Research section
│   └── credit_modelling_summary.docx        # Modelling summary report
├── ai_log.md                   # AI usage reflection
└── requirements.txt
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DataQuest_2026.git
cd DataQuest_2026
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app opens automatically in your browser. It has three tabs:

- **EDA** — Data quality report, univariate explorer (distribution + WoE/IV per feature), bivariate heatmap explorer
- **Model** — AUC comparison, ROC curves, feature coefficients, model equation
- **Dashboard** — Threshold slider, approval/default trade-off chart, precision and recall with business interpretation

---

## Reproducing the Modelling Results

Run the notebooks in order:

1. `notebooks/01_data_quality.ipynb` — initial inspection (optional, for reference)
2. `notebooks/02_eda_findings.ipynb` — IV scoring and EDA analysis
3. `notebooks/03_modeling.ipynb` — baseline model, feature engineering, final model, saves `outputs/model_objects.pkl`

If `outputs/model_objects.pkl` already exists in the repo, you can skip straight to running the app.

---

## Data Cleaning Summary

The raw dataset was intentionally messy. The following steps were applied before any modelling:

- **Categorical normalisation** — `home_ownership` and `loan_purpose` had inconsistent capitalisations and synonyms, normalised to consistent lowercase values
- **Date parsing** — `application_date` contained three mixed formats, parsed with `pd.to_datetime(format="mixed")`
- **Missing value strategy** — `months_since_last_delinquency` filled with 999 (missing = never delinquent, not zero); numeric columns filled with median
- **Outlier capping** — Winsorisation via IQR method; rows kept, extreme values clipped
- **Deduplication** — run twice: once before capping, once after (capping can create new duplicates)

---

## Feature Engineering

Five ratio features were tested individually against the baseline. Four improved AUC and were kept:

| Feature | Description | Result |
|---------|-------------|--------|
| `loan_income_x_dti` | Loan/income × DTI ratio — double debt burden | Kept |
| `delinquency_severity` | Delinquency count / months since last delinquency | Kept |
| `income_to_loan_ratio` | Income relative to loan size | Kept |
| `revolving_to_income` | Revolving balance relative to income | Kept |
| `income_to_rate_ratio` | Income / interest rate interaction | Dropped (hurt AUC) |

WoE encoding was explored but not used in the final model — applying it across all features reduced AUC from 0.789 to 0.703.

---

## Final Model

```
P(default) = 1 / (1 + e^(-η))

η = −2.1613
  − 0.4934 × months_since_last_delinquency
  − 0.4538 × age
  − 0.2834 × loan_amount
  − 0.2119 × employment_length_years
  − 0.1835 × income_to_loan_ratio
  − 0.1665 × annual_income
  + 0.2781 × num_hard_inquiries_6mo
  + 0.2716 × loan_income_x_dti
  + 0.2554 × credit_utilisation_pct
  + 0.1524 × num_open_accounts
  + 0.1334 × num_delinquencies_2yr
  + 0.1309 × delinquency_severity
  ... (full equation in app Model tab)
```

---

## Dependencies

```
pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn
plotly
```

Full list in `requirements.txt`.

---

## Deliverables

| Item | Location |
|------|----------|
| Interactive app | `app/streamlit_app.py` |
| Research section | `report/Interpretable_Credit_Modeling.docx` |
| Modelling summary | `report/credit_modelling_summary.docx` |
| EDA notebook | `notebooks/02_eda_findings.ipynb` |
| Modelling notebook | `notebooks/03_modeling.ipynb` |
| AI usage reflection | `ai_log.md` |
