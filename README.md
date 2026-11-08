# Fraud Detection API

Offline research on rare-event classification, alongside an earlier Flask API
prototype. Uses the public [ULB/Worldline credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Monthly fraud-alert thresholds

The [companion technical report and reproduction command](research/monthly_thresholds/README.md) compare fixed and monthly cutoffs under a shared review allowance on two years of simulated transactions. The [monthly data audit](research/monthly_thresholds/results/data_audit.csv) and [held-out results](research/monthly_thresholds/results/monthly.csv) accompany the paper.

## Research extension: reliability under distribution shift

The November 2026 extension compares Logistic Regression and Random Forest with
no imbalance treatment, balanced weights, or training-only SMOTE. It separates
training, probability calibration, validation and chronological test data;
freezes model selection and thresholds before test evaluation; and studies
controlled label-prior shifts without retuning.

- [Experimental protocol and limitations](research/PROTOCOL.md)
- [Results and interpretation](research/REPORT.md)
- [Aggregate run artifacts](research/results/2026-11-05)
- [Implementation](research/experiment.py) and [safeguard tests](tests/test_research.py)

The earlier score table has been withdrawn: its pipeline fitted preprocessing
before splitting and selected a model using test recall. It also listed XGBoost
without a matching implementation. Those scores are not validated evidence.
The new study reports all comparisons, including negative findings.

### Reproduce the study

Python 3.12 was used for the recorded run. Create a virtual environment, then:

```bash
pip install -r research/requirements-lock.txt
python -m pytest tests/ -q
python -m research.experiment --data data/creditcard.csv --output research/results/new-run
python -m research.report --run research/results/new-run
```

Download `creditcard.csv` from Kaggle or the TensorFlow mirror documented in the
protocol. The CSV and personal application documents are not committed.
The runner refuses to overwrite a result directory. Default seeds are 42, 43,
44; Random Forest has 100 trees and uses four workers. Results are offline
benchmark measurements, not deployed fraud-system performance.

## Tech Stack

- Python 3.11
- scikit-learn (models, preprocessing, evaluation)
- imbalanced-learn (SMOTE for oversampling)
- Flask (REST API)
- pandas / numpy (data pipeline)
- pytest + pytest-cov (automated tests)
- flake8 (linting)
- GitHub Actions CI

## Project Structure

```
fraud-detection/
├── src/
│   ├── api.py           # Flask REST API
│   ├── config.py        # All settings in one place
│   ├── data_loader.py   # Data loading, preprocessing, feature engineering
│   └── model.py         # Training, evaluation, save/load
├── tests/
│   ├── conftest.py      # Shared fixtures
│   ├── test_api.py      # API endpoint tests
│   ├── test_data_loader.py  # Data pipeline tests
│   └── test_model.py    # Model training/eval tests
├── notebooks/
│   └── 01_eda.py        # Exploratory data analysis
├── data/                # Dataset goes here (not in git)
├── models/              # Trained models (not in git)
├── train.py             # Training entry point
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml
```

## Setup

```bash
# clone and install
git clone https://github.com/ErayGulerler/fraud-detection-api.git
cd fraud-detection-api
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Legacy API demo training

`train.py` and `src/` are retained for the original demonstration. They are not
the research evaluation path. Known limitations include preprocessing before
the random split, full-hour counts that include future events, test-based model
selection, and serving-time Amount scaling that differs from training. Do not
use this path for performance claims or production decisions. The research
pipeline does not export a compatible API model.

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and put `creditcard.csv` in the `data/` folder.

```bash
# train with default settings (Random Forest)
python train.py

# compare the two legacy models (exploratory only)
python train.py --compare
```

## Running the API

```bash
# after training (need model files in models/)
python -m src.api
```

### Endpoints

**POST /predict** — classify a single transaction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1":-1.3,"V2":1.2,"V3":0.5,"V4":-0.8,"V5":0.3,"V6":-0.1,"V7":0.2,"V8":0.1,"V9":-0.5,"V10":-0.3,"V11":1.1,"V12":0.2,"V13":-0.7,"V14":-0.9,"V15":0.1,"V16":-0.2,"V17":0.3,"V18":0.1,"V19":-0.1,"V20":0.05,"V21":0.1,"V22":-0.3,"V23":0.05,"V24":0.2,"V25":-0.1,"V26":0.1,"V27":0.05,"V28":0.02,"Amount":125.50}'
```

Response:
```json
{"fraud": false, "fraud_probability": 0.0832, "threshold": 0.35}
```

**POST /predict/batch** — classify up to 1000 transactions

**GET /health** — health check

## Tests

```bash
# run all tests
pytest tests/ -v

# with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests use generated sample data so you don't need the Kaggle dataset to run them.

## Things I'd improve

- [ ] Add XGBoost as an option (currently only RF)
- [ ] Feature importance visualization endpoint
- [ ] Model versioning and A/B testing
- [ ] Swagger/OpenAPI docs
- [ ] Proper logging instead of print statements
- [ ] Monitoring/alerting for model drift

## License

MIT
