# Bike Rental Demand Prediction - Streamlit App

This folder is ready to upload to GitHub and deploy on Streamlit Community Cloud.

## Files needed in GitHub

- `app.py`
- `Dataset.csv`
- `requirements.txt`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy using GitHub and Streamlit Cloud

1. Create a new GitHub repository.
2. Upload these files to the repository:
   - `app.py`
   - `Dataset.csv`
   - `requirements.txt`
3. Go to Streamlit Community Cloud: `https://share.streamlit.io`
4. Click `New app`.
5. Select your GitHub repository.
6. Set main file path as:

```text
app.py
```

7. Click `Deploy`.

## Project explanation

The app trains an XGBoost regression model from `Dataset.csv` and predicts total bike rental demand, `cnt`.

The preprocessing follows the project notebook:

- Replace invalid values such as `?`
- Drop `instant`
- Remove leakage columns `casual` and `registered`
- Drop `atemp` because it is highly correlated with `temp`
- Create `is_weekend`, `quarter`, `is_summer`, and `is_winter`
- One-hot encode categorical columns
- Scale `temp`, `hum`, and `windspeed`

The app trains and caches the model automatically when Streamlit starts.
