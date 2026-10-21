# Payment Fraud Detection

Identify synthetic payment-card transactions that should be prioritized for human review. This learning project covers data preparation, EDA, chronological evaluation, class imbalance, model comparison, and reusable predictions.

## Data and source

Generated with Jeff Allen's Federal Reserve CardSim simulator: https://www.federalreserve.gov/econres/feds/cardsim-a-bayesian-simulator-for-payment-card-fraud-detection-research.htm

Configuration: seed 212, 1,000 payers, 100 payees, 180 days starting January 1, 2023; bundled 2022–2023 DCPC survey inputs and default fraud settings. Output: 156,023 transactions, 13 columns, 1,561 fraud cases (approximately 1%). Dates span January 1–June 29, 2023. Inspection found no missing values or fully duplicated rows.

These are simulated transactions, not real customer records. Distance uses grid units. Amounts are left in the simulator's original units.

## Main findings

Fraud tends to have larger amounts and distances, with overlapping distributions. Remote and credit payments have higher fraud rates; remote credit payments reach 3.74%. Hour and payment location together reveal a strong overnight pattern. Daily volume and weekly fraud rates show no obvious sustained trend. These associations partly reflect the simulator's label-generation rules.

## Evaluation and decisions

Fit: January 1–April 25. Validation: April 26–May 24. Final test: May 25–June 29. The final model is refitted using all transactions before May 25.

Inputs, in order: amount, distance, credit_card, remote, hour. The target is fraud. Run identifiers and raw customer/merchant IDs are excluded; date_time is retained for splitting. Scaling is fitted within logistic regression pipelines. Trees require no scaling.

| Candidate | Validation average precision |
|---|---:|
| Logistic regression | 0.3314 |
| Logistic regression with circular hour | 0.3985 |
| Gradient-boosted trees | 0.7279 |
| Class-weighted gradient-boosted trees | 0.7564 |

The dummy reference catches no fraud; its training AP was 0.0099. AP measures ranking and is not accuracy.

Selected model: HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, class_weight="balanced", random_state=42). Weighting gave rare fraud cases more influence without resampling validation or test data. It improved AP but does not establish fairness or remove every kind of bias.

Threshold 0.70 was selected on validation as an educational high-recall tradeoff, without an established business cost or review budget. Validation precision was 38.8%, recall 97.1%, accuracy 98.50%, with 606 alerts. Class-weighted scores should not be treated as calibrated fraud probabilities.

## Final test results

| Metric | Result |
|---|---:|
| Accuracy | 98.28% |
| Precision | 37.20% |
| Recall | 94.72% |
| Average precision | 0.7686 |
| Fraud caught / missed | 305 / 17 |
| False alerts | 515 |
| Legitimate correctly cleared | 30,009 |
| Total alerts | 820 |

Accuracy alone is misleading: an all-legitimate model approaches 99% while catching no fraud. Our model catches most fraud with approximately 1.7 false alerts per fraud caught. The threshold and model were not changed using final test results.

## Files and use

- data/notebook1.ipynb: cleaned notebook with original saved outputs and explanatory Markdown.
- data/transaction-data-S212P1000D180.csv: generated dataset.
- data/fraud_model.joblib: final fitted model, feature order, and threshold.
- requirements.txt: package versions from the original Python 3.14.6 environment.
- LICENSE: original simulator code public-domain notice; not a newly assigned license for this entire project.

Open this folder in PyCharm. Select a Python 3.14 environment, then run in its terminal:

```powershell
python -m pip install -r requirements.txt
python -m notebook
```

Open data/notebook1.ipynb and run it from top to bottom. Its data/model paths are relative to the notebook folder. The save cell overwrites the bundled model when rerun. The cleaned notebook retains recorded outputs and has not been rerun end to end; a fresh run may vary with platform or environment. Running model comparison is for reproducing the documented workflow, not reopening test-guided tuning.

For predictions without retraining, run this from a notebook in the data folder:

```python
import joblib
import pandas as pd

bundle = joblib.load("fraud_model.joblib")
transactions = pd.DataFrame([{
    "amount": 50.0, "distance": 100.0,
    "credit_card": 1, "remote": 1, "hour": 23
}])
scores = bundle["model"].predict_proba(transactions[bundle["features"]])[:, 1]
transactions["fraud_score"] = scores
transactions["flag_for_review"] = scores >= bundle["threshold"]
```

Load only model files you trust. A review flag is not proof of fraud or an instruction to block payment.

## Limitations

EDA examined the full dataset before splitting, limiting test independence. Multiple decisions used a single validation period; robustness across additional periods remains untested. Model performance reflects synthetic label-generation rules using similar inputs, not independently observed real fraud. Probability calibration, real-world data, business costs, and live monitoring have not been evaluated. This is a learning project, not a deployed fraud service.

## Simulator source

The existing cardsim/ package, vignette.py, and pyproject.toml generate the synthetic dataset. They are preserved. The root LICENSE contains the original simulator notice. The dataset and fitted model already exist in data/; regeneration is not required to review the analysis.


## Streamlit app: Payment Review

The local demo provides two workflows:

- **Check a transaction:** enter a naira amount, an explicit exchange rate in naira per US dollar, distance in simulator grid units, card type, payment location, and hour. The amount is converted to the model's dollar scale before scoring.
- **Review a CSV:** choose USD or NGN for the entire amount column. Original CardSim CSVs use USD and require no conversion. NGN files require a positive exchange rate. No live rates are fetched.

### Launch from PyCharm

Open the project folder and use PyCharm's Terminal, rather than running app.py as an ordinary Python script:

```powershell
cd C:\Users\USER\Desktop\cardsim
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Keep the terminal running and open http://localhost:8501. Use Ctrl+C to stop the server. The navy/light/teal theme is configured in `.streamlit/config.toml`.

### CSV format and results

Required columns: `amount`, `distance`, `credit_card`, `remote`, `hour`. Amount must be positive, distance nonnegative, flags 0/1, and hour an integer from 0 to 23. Missing/nonnumeric/infinite model inputs are rejected. The app accepts up to 250,000 rows. It preserves extra columns but uses only the five model features; a supplied `fraud` label is never an input.

An example USD CSV is downloadable in the app. The existing example dataset is `data/transaction-data-S212P1000D180.csv`.

The table shows the review flag and model score first, with the highest scores first. A toggle filters to flagged payments. **Download all scored transactions** exports every uploaded row in its original order, even if the table is filtered.

The export retains the input amount and records `input_row` (one-based data row, excluding the header), `input_currency`, `ngn_per_usd`, `amount_usd`, `outside_amount_range`, `fraud_score`, and `flag_for_review`. Upload original transactions rather than previously scored exports. Warnings identify dollar-equivalent amounts outside the generated dataset's observed range (approximately $0.21–$744.50); these are observed dataset limits, not guarantees that every in-range combination is reliable.

Model: `data/fraud_model.joblib`. Review threshold: **0.70**. Conversion changes the input currency scale; it does not validate US-trained synthetic patterns for Nigerian payments. Scores are not calibrated probabilities. Flags are intended for review, not automatic blocking. Full-dataset upload totals include training data and must not be presented as an independent evaluation. Recorded final test metrics are shown separately in **About the model**.

### Development assistance

This project was developed with AI-assisted mentorship and coding support for explanations, modeling suggestions, implementation, and debugging. Experiments and outputs were run and reviewed during the learning workflow.
