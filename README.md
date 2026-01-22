# Fraud Detection API

Machine learning system that classifies credit card transactions as fraudulent or legitimate. Built around the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284k transactions where only 0.17% are fraud.

The interesting challenge here is the extreme class imbalance. A dumb model that always predicts "not fraud" gets 99.8% accuracy, so you need SMOTE + proper metrics (precision/recall, not accuracy) to actually learn anything useful.

## Results

After trying Logistic Regression, Random Forest, and XGBoost:

| Model | Precision | Recall | F1 | AUC-ROC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.87 | 0.62 | 0.72 | 0.95 |
| **Random Forest** | **0.95** | **0.91** | **0.93** | **0.98** |
| XGBoost | 0.94 | 0.92 | 0.93 | 0.98 |

RF and XGBoost are very close. Went with Random Forest since it's easier to interpret and doesn't need an extra dependency.

## Tech Stack

- Python 3.11
- scikit-learn (models, preprocessing, evaluation)
- imbalanced-learn (SMOTE for oversampling)
- Flask (REST API)
- pandas / numpy (data pipeline)
- pytest + pytest-cov (testing, 87% coverage)
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

## Training

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and put `creditcard.csv` in the `data/` folder.

```bash
# train with default settings (Random Forest)
python train.py

# compare all models
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
