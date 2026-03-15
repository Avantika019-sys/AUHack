import glob
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================================================
# SETTINGS
# =========================================================

os.makedirs("models", exist_ok=True)


csv_files = glob.glob("../data/spot-price/*-spot-price.csv")

# =========================================================
# LOOP THROUGH EACH COUNTRY
# =========================================================

for file_path in csv_files:
    file_name = os.path.basename(file_path)
    country = file_name.split("-")[0]

    print("\n==============================")
    print(f"Country: {country}")
    print("==============================")

    # =====================================================
    # LOAD DATA
    # =====================================================

    df = pd.read_csv(file_path)

    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"value (EUR/MWh)": "price"})
    df = df.sort_values("time").reset_index(drop=True)

    # =====================================================
    # FEATURE FUNCTION
    # =====================================================

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

    features = [
        "day_of_week",
        "hour_sin",
        "hour_cos",
        "price_last_hour",
        "price_same_hour_yesterday",
        "price_same_hour_last_week",
        "average_price_last_24_hours",
        "average_price_last_week"
    ]

    # =====================================================
    # 1) EVALUATE MODEL ON KNOWN DATA
    # =====================================================

    df_eval = create_features(df).dropna().reset_index(drop=True)

    X = df_eval[features]
    y = df_eval["price"]

    split = int(len(df_eval) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    eval_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42
    )

    eval_model.fit(X_train, y_train)

    test_predictions = eval_model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = mean_squared_error(y_test, test_predictions) ** 0.5

    print(f"Evaluation MAE  : {mae:.3f}")
    print(f"Evaluation RMSE : {rmse:.3f}")

    # =====================================================
    # 2) TRAIN FINAL MODEL ON ALL KNOWN DATA
    # =====================================================

    final_df = create_features(df).dropna().reset_index(drop=True)

    X_all = final_df[features]
    y_all = final_df["price"]

    final_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42
    )

    final_model.fit(X_all, y_all)

    joblib.dump(final_model, f"models/lightgbm_{country}_model.pkl")

    # =====================================================
    # 3) FORECAST FUTURE RECURSIVELY
    # =====================================================

    history = df[["time", "price"]].copy()

    last_time = history["time"].max()
    #forecast_end_date = last_time + pd.Timedelta(days=7)
    forecast_end_date = pd.Timestamp("2026-12-31 23:45:00")

    if last_time >= forecast_end_date:
        print(f"{country}: data already reaches or passes forecast end date.")
        continue

    future_times = pd.date_range(
        start=last_time + pd.Timedelta(hours=1),
        end=forecast_end_date,
        freq="h"
    )

    future_predictions = []

    for future_time in future_times:
        temp_history = history.copy()

        temp_row = pd.DataFrame({
            "time": [future_time],
            "price": [np.nan]
        })

        temp_history = pd.concat([temp_history, temp_row], ignore_index=True)

        temp_feat = create_features(temp_history)

        latest_row = temp_feat.iloc[[-1]][features]

        predicted_price = final_model.predict(latest_row)[0]

        future_predictions.append(predicted_price)

        history = pd.concat([
            history,
            pd.DataFrame({
                "time": [future_time],
                "price": [predicted_price]
            })
        ], ignore_index=True)

    future_df = pd.DataFrame({
        "time": future_times,
        "predicted_price": future_predictions
    })

    print(f"Forecasted from {last_time} to {forecast_end_date}")
    print(f"Number of future hours predicted: {len(future_df)}")

    # =====================================================
    # 4) SHOW GRAPH
    # =====================================================

    # historical_plot = df.tail(200).copy()
    historical_plot = df.copy()

    plt.figure(figsize=(12,5))

    # historical data (solid line)
    plt.plot(
        historical_plot["time"],
        historical_plot["price"],
        label="Historical Price",
        linewidth=2
    )

    # future prediction (dashed red line)
    plt.plot(
        future_df["time"],
        future_df["predicted_price"],
        linestyle="--",
        color="red",
        linewidth=2,
        label="Forecast"
    )

    # vertical line where forecast starts
    plt.axvline(
        last_time,
        color="gray",
        linestyle=":",
        linewidth=2,
        label="Forecast Start"
    )

    plt.title(f"{country} Electricity Price Forecast")
    plt.xlabel("Time")
    plt.ylabel("Price (EUR/MWh)")
    plt.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()