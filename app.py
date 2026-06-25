from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


DATA_PATH = Path(__file__).resolve().parent / "Dataset.csv"
TARGET_COLUMN = "cnt"
SCALE_COLUMNS = ["temp", "hum", "windspeed"]
CATEGORICAL_COLUMNS = ["season", "holiday", "workingday", "weathersit"]
INVALID_VALUES = ["?", "#", "N/A", "NA", "null", "None", "-", ""]


st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="B",
    layout="wide",
)


def preprocess_data(raw_df, scaler=None, fit_scaler=True, feature_columns=None):
    df = raw_df.copy()
    df.columns = df.columns.str.strip()
    df.replace(INVALID_VALUES, np.nan, inplace=True)

    if "instant" in df.columns:
        df.drop(columns=["instant"], inplace=True)

    numeric_columns = ["temp", "atemp", "hum", "windspeed", "yr", "mnth", "casual", "registered"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(df[column].mode()[0])

    numeric_only = df.select_dtypes(include="number").columns
    df[numeric_only] = df[numeric_only].fillna(df[numeric_only].median())

    if "hum" in df.columns:
        hum_median = df["hum"].median()
        df["hum"] = df["hum"].replace(0, hum_median)

    df.drop(columns=[column for column in ["casual", "registered", "atemp"] if column in df.columns], inplace=True)

    df["dteday"] = pd.to_datetime(df["dteday"], dayfirst=True, errors="coerce")
    df["is_weekend"] = df["dteday"].dt.dayofweek.isin([5, 6]).astype(int)
    df["quarter"] = df["dteday"].dt.quarter
    df.drop(columns=["dteday"], inplace=True)

    df["is_summer"] = df["mnth"].apply(lambda month: 1 if month in [5, 6, 7, 8] else 0)
    df["is_winter"] = df["mnth"].apply(lambda month: 1 if month in [11, 12, 1, 2] else 0)

    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=int)

    if fit_scaler:
        scaler = StandardScaler()
        df[SCALE_COLUMNS] = scaler.fit_transform(df[SCALE_COLUMNS])
    else:
        df = df.reindex(columns=feature_columns, fill_value=0)
        df[SCALE_COLUMNS] = scaler.transform(df[SCALE_COLUMNS])

    return df, scaler


def prepare_prediction_input(input_row, scaler, feature_columns):
    input_df = pd.DataFrame([input_row])
    processed_df, _ = preprocess_data(
        input_df,
        scaler=scaler,
        fit_scaler=False,
        feature_columns=feature_columns,
    )
    return processed_df


@st.cache_resource
def train_model():
    if not DATA_PATH.exists():
        st.error("Dataset.csv not found. Add Dataset.csv to the same GitHub folder as app.py.")
        st.stop()

    raw_df = pd.read_csv(DATA_PATH)
    processed_df, scaler = preprocess_data(raw_df, fit_scaler=True)

    X = processed_df.drop(TARGET_COLUMN, axis=1)
    y = processed_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions),
    }

    return model, scaler, list(X.columns), metrics


model, scaler, feature_columns, metrics = train_model()

st.title("Bike Rental Demand Prediction")
st.caption("Streamlit deployment for the Bike Rental XGBoost regression project")

input_area, model_area = st.columns([2, 1])

with input_area:
    st.subheader("Enter Details")

    with st.form("bike_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_date = st.date_input("Date", value=date(2012, 6, 15))
            hr = st.slider("Hour", 0, 23, 8)
            season = st.selectbox("Season", ["springer", "summer", "fall", "winter"])
            yr = st.selectbox("Year", [2011, 2012], index=1)

        with col2:
            mnth = st.slider("Month", 1, 12, 6)
            weekday = st.slider("Weekday", 0, 6, 5)
            holiday = st.selectbox("Holiday", ["No", "Yes"])
            workingday = st.selectbox("Working Day", ["Working Day", "No work"])

        with col3:
            weathersit = st.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
            temp = st.slider("Temperature", 0.0, 1.0, 0.50, 0.01)
            hum = st.slider("Humidity", 0.0, 1.0, 0.60, 0.01)
            windspeed = st.slider("Windspeed", 0.0, 1.0, 0.20, 0.01)

        submitted = st.form_submit_button("Predict Demand")

    if submitted:
        input_row = {
            "dteday": selected_date.strftime("%d-%m-%Y"),
            "season": season,
            "yr": yr,
            "mnth": mnth,
            "hr": hr,
            "holiday": holiday,
            "weekday": weekday,
            "workingday": workingday,
            "weathersit": weathersit,
            "temp": temp,
            "hum": hum,
            "windspeed": windspeed,
        }

        model_input = prepare_prediction_input(input_row, scaler, feature_columns)
        predicted_demand = int(round(max(0, model.predict(model_input)[0])))

        st.metric("Predicted Total Bike Rentals", predicted_demand)

        if predicted_demand < 100:
            st.info("Low demand expected. Normal or reduced bike availability may be enough.")
        elif predicted_demand < 300:
            st.success("Moderate demand expected. Maintain regular bike availability.")
        else:
            st.warning("High demand expected. Increase bike availability before this time slot.")

with model_area:
    st.subheader("Model Summary")
    st.write("Model: XGBoost Regressor")
    st.write("Target: `cnt`")
    st.write("Data leakage columns removed: `casual`, `registered`")
    st.write("Multicollinear column removed: `atemp`")

    st.metric("MAE", f"{metrics['MAE']:.2f}")
    st.metric("RMSE", f"{metrics['RMSE']:.2f}")
    st.metric("R2 Score", f"{metrics['R2 Score']:.3f}")
