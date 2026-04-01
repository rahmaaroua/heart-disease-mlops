# Heart Disease MLOps Pipeline

End-to-end MLOps project: UCI Heart Disease dataset → 3 trained models → FastAPI backend → Flask web UI → Docker → Kubernetes → GitHub Actions CI/CD.

## Quick Start

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
dvc init

# 2. Generate / place dataset
python data/raw/generate.py          # or drop heart_disease_uci.csv into data/raw/

# 3. Run full pipeline
python src/preprocess.py
python src/train.py                  # trains LR + RF + XGBoost, picks champion

# 4. Start FastAPI backend (port 8000)
uvicorn api.main:app --reload --port 8000

# 5. Start Flask frontend (port 5000) — new terminal
cd frontend && python app.py
# Open http://localhost:5000
```

## Docker

```bash
docker build -t heart-disease-app .
docker run -p 8000:8000 -p 5000:5000 heart-disease-app
# UI  → http://localhost:5000
# API → http://localhost:8000/docs
```

## Kubernetes (Minikube)

```bash
minikube start
# Edit k8s/deployment.yaml → replace YOUR_DOCKERHUB_USERNAME
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods                          # wait for Running
minikube service heart-disease-app        # opens UI in browser
```

## GitHub Actions Secrets

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

## Models

| Model | F1 | ROC-AUC | Notes |
|---|---|---|---|
| Logistic Regression | 0.838 | 0.924 | Baseline, interpretable |
| Random Forest | 0.800 | 0.893 | Ensemble |
| XGBoost ★ | 0.843 | 0.906 | Champion — best F1 |

## Project Structure

```
heart-disease-mlops/
├── data/
│   ├── raw/heart_disease_uci.csv     ← DVC tracked
│   └── processed/                    ← DVC outputs
├── src/
│   ├── preprocess.py                 ← DVC stage 1
│   └── train.py                      ← DVC stage 2
├── api/
│   └── main.py                       ← FastAPI (port 8000)
├── frontend/
│   ├── app.py                        ← Flask (port 5000)
│   └── templates/
│       ├── upload.html               ← Screen 1: upload + model select
│       └── results.html              ← Screen 2: predictions table
├── models/                           ← trained model artifacts
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/
│   └── pipeline.yaml                 ← CI/CD
├── dvc.yaml
├── params.yaml
├── Dockerfile
├── start.sh
└── requirements.txt
```

## API Endpoints

| Method | URL | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/models` | List models + metrics |
| POST | `/predict/csv` | Upload CSV → JSON predictions |

## Web App Flow

1. Open `http://localhost:5000`
2. Upload a CSV (same schema as `heart_disease_uci.csv`)
3. Choose one of 3 models
4. Click **Run predictions** → redirected to results page
5. Filter by disease / no disease / high risk
6. Sort any column
7. Export results as CSV
