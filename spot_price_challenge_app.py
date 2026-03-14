import glob
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb

st.set_page_config(page_title="Spot Price Challenge", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_country_data():
    files = glob.glob("datasets/*-spot-price.csv")
    data = {}
    for file_path in files:
        country = os.path.basename(file_path).split("-")[0]
        df = pd.read_csv(file_path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"value (EUR/MWh)": "price"})
        df = df.sort_values("time").reset_index(drop=True)
        data[country] = df
    return data


def create_features(dataframe):
    df_feat = dataframe.copy()
    df_feat["hour_of_day"] = df_feat["time"].dt.hour
    df_feat["day_of_week"] = df_feat["time"].dt.dayofweek
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour_of_day"] / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour_of_day"] / 24)
    df_feat["price_last_hour"] = df_feat["price"].shift(1)
    df_feat["price_same_hour_yesterday"] = df_feat["price"].shift(24)
    df_feat["price_same_hour_last_week"] = df_feat["price"].shift(168)
    df_feat["average_price_last_24_hours"] = df_feat["price"].shift(1).rolling(24).mean()
    df_feat["average_price_last_week"] = df_feat["price"].shift(1).rolling(168).mean()
    return df_feat


FEATURES = [
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "price_last_hour",
    "price_same_hour_yesterday",
    "price_same_hour_last_week",
    "average_price_last_24_hours",
    "average_price_last_week",
]


@st.cache_resource
def train_model(country, df):
    df_feat = create_features(df).dropna().reset_index(drop=True)
    X = df_feat[FEATURES]
    y = df_feat["price"]

    split = int(len(df_feat) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    test_time = df_feat["time"].iloc[split:]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f"lightgbm_{country}_model.pkl"))

    return model, df_feat, test_time, y_test.reset_index(drop=True), pd.Series(preds), mae, rmse


def recursive_forecast(model, df, steps=24):
    history = df[["time", "price"]].copy()
    future_rows = []

    for step in range(steps):
        next_time = history["time"].max() + pd.Timedelta(hours=1)

        temp = pd.concat(
            [history, pd.DataFrame({"time": [next_time], "price": [np.nan]})],
            ignore_index=True,
        )

        temp_feat = create_features(temp)
        row = temp_feat.iloc[[-1]][FEATURES]
        pred = float(model.predict(row)[0])

        history = pd.concat(
            [history, pd.DataFrame({"time": [next_time], "price": [pred]})],
            ignore_index=True,
        )
        future_rows.append({"time": next_time, "predicted_price": pred})

    return pd.DataFrame(future_rows)


def closest_option(true_value, options):
    return min(options, key=lambda x: abs(x - true_value))


# -----------------------------
# App
# -----------------------------
st.title("⚡ Spot Price Challenge")
st.caption("Train a model, explore spot prices, then play a prediction game against the model.")

data = load_country_data()
if not data:
    st.error("No files found in datasets/. Add your CSV files first.")
    st.stop()

countries = sorted(data.keys())

with st.sidebar:
    st.header("Settings")
    selected_country = st.selectbox("Country", countries, index=0)
    forecast_hours = st.slider("Forecast horizon (hours)", 24, 168, 48, step=24)
    show_history_days = st.slider("History to show (days)", 7, 90, 30, step=7)

df = data[selected_country]
model, df_feat, test_time, y_test, preds, mae, rmse = train_model(selected_country, df)
future_df = recursive_forecast(model, df, steps=forecast_hours)

last_known = df["time"].max()
challenge_row = future_df.iloc[0]
challenge_time = challenge_row["time"]
true_prediction = float(challenge_row["predicted_price"])

# Build multiple choice options around the model prediction
rng = np.random.default_rng(42)
offsets = np.array([-15, -7, 0, 8])
options = np.round(true_prediction + offsets, 1).tolist()
rng.shuffle(options)

col1, col2, col3 = st.columns([1.2, 1, 1])
with col1:
    st.metric("Country", selected_country)
with col2:
    st.metric("MAE", f"{mae:.2f}")
with col3:
    st.metric("RMSE", f"{rmse:.2f}")

st.subheader("Market Dashboard")
history_cutoff = df["time"].max() - pd.Timedelta(days=show_history_days)
hist_plot = df[df["time"] >= history_cutoff].copy()

chart_df = pd.DataFrame({
    "time": pd.concat([hist_plot["time"], future_df["time"]], ignore_index=True),
    "Historical price": pd.concat([hist_plot["price"], pd.Series([np.nan] * len(future_df))], ignore_index=True),
    "Model forecast": pd.concat([pd.Series([np.nan] * len(hist_plot)), future_df["predicted_price"]], ignore_index=True),
}).set_index("time")

# Hidden forecast toggle
show_model_now = st.toggle("Reveal the model forecast", value=False)
if not show_model_now:
    chart_df["Model forecast"] = np.nan

st.line_chart(chart_df, height=360)

st.subheader("🎯 Prediction Game")
st.write(
    f"Predict the spot price for **{selected_country}** at **{challenge_time:%Y-%m-%d %H:%M}**. "
    "Pick the option closest to the model forecast."
)

selected_option = st.radio("Choose your price (EUR/MWh)", options, horizontal=True)

if "score" not in st.session_state:
    st.session_state.score = 0
if "submitted" not in st.session_state:
    st.session_state.submitted = False

if st.button("Submit prediction"):
    st.session_state.submitted = True
    if float(selected_option) == closest_option(true_prediction, options):
        st.session_state.score += 10

if st.session_state.submitted:
    st.success(f"Your current score: {st.session_state.score} points")

    compare_df = pd.DataFrame({
        "time": pd.concat([hist_plot["time"], pd.Series([challenge_time, challenge_time])], ignore_index=True),
        "Historical price": pd.concat([hist_plot["price"], pd.Series([np.nan, np.nan])], ignore_index=True),
        "Your prediction": pd.concat([pd.Series([np.nan] * len(hist_plot)), pd.Series([selected_option, np.nan])], ignore_index=True),
        "Model forecast": pd.concat([pd.Series([np.nan] * len(hist_plot)), pd.Series([np.nan, true_prediction])], ignore_index=True),
    }).set_index("time")

    st.write("After submitting, your choice and the model forecast are shown together:")
    st.line_chart(compare_df, height=320)

    st.write(
        f"**Your choice:** {float(selected_option):.1f} EUR/MWh  |  "
        f"**Model forecast:** {true_prediction:.1f} EUR/MWh"
    )

with st.expander("How this works"):
    st.markdown(
        '''
- The app loads spot price data for each country.
- A LightGBM model is trained on historical data.
- MAE and RMSE show how well the model performs on held-out historical data.
- The forecast is hidden at first so the user can play a guessing game.
- After the user submits an answer, the app reveals both the user's choice and the model forecast.
        '''
    )