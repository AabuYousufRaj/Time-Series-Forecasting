# PGCB Power Demand Forecasting

An end-to-end machine-learning project for short-term electricity-demand forecasting using historical Power Grid Company of Bangladesh (PGCB) data. The repository includes the research notebook, prepared model outputs, explainability plots, and a Streamlit decision-support dashboard.

![Dashboard overview](PGCB_Forecasting/Dashboard%201.png)

## Highlights

- Compares LightGBM quantile regression, Random Forest, PyTorch LSTM, and Prophet forecasts.
- Presents actual demand, forecasts, peak variance, and model error in an interactive dashboard.
- Includes LightGBM confidence bands and SHAP-based global and local explanations.
- Includes EDA charts, LSTM training history, cross-validation results, and evaluation metrics.
- Keeps the dashboard runnable from any working directory by resolving assets relative to `app.py`.

## Results

The prepared evaluation outputs report the following performance on the project evaluation split:

| Model | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| LightGBM (Quantile) | 196.55 MW | 267.43 MW | 1.59% |
| Random Forest | 199.57 MW | 275.08 MW | 1.60% |
| LSTM | 347.39 MW | 469.78 MW | 2.80% |
| Prophet | 1,136.13 MW | 1,474.43 MW | 9.72% |

The dashboard also includes the prepared LSTM results. See `PGCB_Forecasting/evaluation_metrics.csv` for the complete table.

## Repository Layout

```text
.
├── Codes.ipynb                 # Data preparation, EDA, training, evaluation, and plots
├── Dataset/
│   └── PGCB_date_power_demand.xlsx
├── PGCB_Forecasting/
│   ├── app.py                  # Streamlit dashboard
│   ├── *.csv                   # Dashboard, forecast, metric, and training outputs
│   ├── *.pkl / *.pt            # Serialized trained models and scalers
│   └── *.png                   # Diagnostics, EDA, and explainability visuals
├── Report.pdf                  # Project report
└── requirements.txt
```

## Quick Start

### 1. Create an environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the dashboard

Run this command from the repository root:

```bash
streamlit run PGCB_Forecasting/app.py
```

The dashboard loads the prepared CSV and image artifacts from `PGCB_Forecasting/`. The serialized model files are included for reproducibility and future inference work; the current dashboard visualizes the prepared forecast outputs.

## Reproducing the Analysis

`Codes.ipynb` contains the analysis workflow, including data inspection, hourly resampling, time-based interpolation, exploratory analysis, feature preparation, model training, evaluation, and artifact generation. The notebook was authored for a Kaggle-style environment and currently searches `/kaggle/input` for the source workbook. To reproduce it locally, update that input path to `Dataset/PGCB_date_power_demand.xlsx` or run the notebook in the matching environment.

## Data

The source workbook is included for research and demonstration purposes. Confirm that you have the right to redistribute any underlying data before publishing a public mirror. Forecast and evaluation CSVs are derived artifacts, not a replacement for the original data documentation.

## Screenshots

| Forecasting | Explainable AI |
| --- | --- |
| ![Forecasting view](PGCB_Forecasting/Dashboard%202.png) | ![Explainable AI view](PGCB_Forecasting/Dashboard%203.png) |

## Limitations

- This is a research and decision-support prototype, not an operational grid-control system.
- Forecast quality depends on the historical data and evaluation design in the notebook.
- The dashboard uses precomputed artifacts; retraining is performed through the notebook workflow.
- External image loading for the PGCB logo requires network access. The dashboard remains usable if that image is unavailable.

## Citation

If you use this project in academic or professional work, cite the accompanying `Report.pdf` and describe the dataset version, evaluation split, and model artifacts used.

## License

No license has been selected for this repository yet. Until a license is added, standard copyright restrictions apply.
