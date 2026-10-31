# Predictive Maintenance Command Center

Streamlit dashboard for predictive and prescriptive maintenance on **NASA C-MAPSS FD004**: deep learning models (LSTM, GRU + Attention mechanism) for engine condition classification and RUL prediction, explainability (XAI), and AI-generated (Gemini LLM) maintenance advisories.

## Features

- **Fleet Overview**: full-fleet inference straight from the raw C-MAPSS test files, predicted-vs-true RUL scatter, condition-state distribution, and fleet-level health metrics.
- **Engine Deep-Dive**: per-engine condition probabilities, RUL estimate, sensor trends, and XAI sensor-impact analysis (perturbation-based impacts, with Integrated Gradients compared in the notebook).
- **AI Prescriptive Advisory**: Gemini generates a maintenance recommendation per engine, cached per engine ID so switching engines never shows stale advice. Failed API calls are never logged as advisories.
- **Model Governance & Impact**: model card, model-comparison tables, sensitivity analyses, an editable maintenance cost-avoidance estimator, and an operator decision audit trail (persisted to CSV, displayed in-app, downloadable).
- **Flexible artifact loading**: load models from a local folder or upload a zip from the sidebar. If artifacts are missing, the app falls back to **demo mode** with simulated data instead of crashing.

## Condition classes

| Class | Meaning |
|---|---|
| Healthy | No fault |
| Fan | Fan degradation |
| HPC | High-pressure compressor degradation |
| HPC+Fan | Compound fault |

## Models

- **Condition classifier**: four-class sequence model (attention-based recurrent/convolutional variants compared).
- **RUL regressor**: LSTM / GRU / GRU + Attention variants compared. In `rul_model_comparison.csv`, GRU + Attention is the strongest on the test split (RMSE 15.46, MAE 10.15, R² 0.871).
- Full experiments (class-imbalance strategies, hyperparameter tuning, window-size, RUL-cap and label-threshold ablations, bootstrap confidence intervals, XAI comparison) are in the notebook and CSV files.

## Repository structure

```
├── Dashboard.py
├── requirements.txt
├── runtime.txt
├── test_FD004.txt
├── RUL_FD004.txt
├── NOTEBOOK_Predictive_Maintenance_Prescriptive.ipynb
├── selected_deployment_models/
│   ├── condition_classifier.keras
│   ├── rul_regressor.keras
│   ├── preprocessing_and_label_rules.joblib
│   ├── deployment_manifest.json
│   └── *.csv   (comparison tables, optional)
└── *.csv       (experiment results)
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run Dashboard.py
```

Python 3.11 is recommended (TensorFlow 2.15).

### Configuration

| Setting | How |
|---|---|
| Model artifacts folder | Sidebar, or env var `PDM_ARTIFACT_DIR` (default: `./selected_deployment_models`) |
| FD004 data folder | Sidebar, or env var `PDM_DATA_DIR` (default: script folder) |
| Gemini API key | Entered in the sidebar (get one at https://aistudio.google.com/apikey) |

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (include the `selected_deployment_models/` folder).
2. Go to https://share.streamlit.io and click **Create app**.
3. Select the repo, set the main file to `Dashboard.py`, and under **Advanced settings** choose **Python 3.11**.
4. Deploy.

Notes: Community Cloud has about 1 GB RAM; if TensorFlow runs out of memory, use Hugging Face Spaces or Render. The audit log is written to local disk, which resets on redeploy; the in-app CSV download still works.

## Dataset

[NASA C-MAPSS](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) turbofan engine degradation simulation, subset **FD004** (6 operating conditions, 2 fault modes).

## Disclaimer

This is a research/educational project. The cost-avoidance figures are illustrative, and AI-generated advisories must be reviewed by a qualified engineer before any maintenance action.
