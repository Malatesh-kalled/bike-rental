# Bike Rental Demand Prediction

A machine learning app that predicts daily bike rental demand based on weather conditions, seasons, and calendar features.

Trains an XGBoost regression model on historical bike sharing data and serves predictions through a Streamlit interface.


---

## What it does

- Loads and preprocesses a bike sharing dataset automatically on startup
- Engineers features like weekend flag, quarter, and seasonal indicators
- Trains and caches an XGBoost regression model
- Predicts total daily bike rental demand (`cnt`) based on user inputs
- Visualizes which factors (temperature, rain, season) have the biggest impact on demand

---

## Demo

> Live app: [Bike rental](https://bike-rental-vbzqfgw7k2gw9oqiu2kntz.streamlit.app/)

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| ML Model | XGBoost Regression |
| Data processing | Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Language | Python 3.8+ |
| Deployment | Streamlit Cloud |

---

## How it works

1. App loads `Dataset.csv` and preprocesses it on startup
2. Invalid values are replaced, leakage columns (`casual`, `registered`) are dropped
3. New features are engineered: `is_weekend`, `quarter`, `is_summer`, `is_winter`
4. Categorical columns are one-hot encoded; `temp`, `hum`, `windspeed` are scaled
5. XGBoost model is trained and cached so it only runs once per session
6. User inputs weather and calendar conditions via the UI
7. Model returns predicted bike rental demand for that day

---

## Project structure

```
bike-rental-demand/
├── app.py               # Main Streamlit app
├── Dataset.csv          # Bike sharing dataset
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Local setup

**1. Clone the repo**
```bash
git clone https://github.com/Malatesh-Kalled/bike-rental-demand.git
cd bike-rental-demand
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo → set main file to `app.py`
4. Click **Deploy**

No API keys or secrets needed — this app runs entirely on local data and a trained model.

---

## Feature engineering

| Feature | Description |
|---|---|
| `is_weekend` | 1 if Saturday or Sunday |
| `quarter` | Calendar quarter (1–4) |
| `is_summer` | 1 if summer season |
| `is_winter` | 1 if winter season |
| Dropped `atemp` | Highly correlated with `temp` — removed to avoid multicollinearity |
| Dropped `casual`, `registered` | Target leakage — these sum to `cnt` |

---

## Requirements

```
streamlit
pandas
scikit-learn
xgboost
matplotlib
seaborn
```

---

## Author

**Malatesh M Kalled**
[linkedin.com/in/malatesh-kalled](https://www.linkedin.com/in/malatesh-kalled) · [github.com/malateshkalled](https://github.com/malateshkalled)
