import glob
import math
import os
import random
import sqlite3

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

COUNTRY_FILES = [
    "AT-spot-price.csv",
    "BE-spot-price.csv",
    "CH-spot-price.csv",
    "CZ-spot-price.csv",
    "DE-spot-price.csv",
    "DK1-spot-price.csv",
    "FR-spot-price.csv",
    "NL-spot-price.csv",
    "PL-spot-price.csv",
]
COUNTRIES = [f.split("-spot-price.csv")[0] for f in COUNTRY_FILES]

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

UNIT_PRICES = {
    "Solar Unit": 40,
    "Wind Unit": 60,
    "Battery Unit": 80,
    "Hydro Unit": 100,
}

QUESTION_BANK = [
    "What is the model forecast for {country} at {time}?",
    "Choose the closest predicted spot price for {country} at {time}.",
    "Which option best matches the forecasted price in {country} at {time}?",
    "At {time}, where will {country}'s spot price most likely land?",
    "Select the forecasted price level for {country} at {time}.",
    "Which value is closest to the machine-learning forecast for {country} at {time}?",
    "Predict the day-ahead spot price for {country} at {time}.",
    "Pick the best forecast for {country} at {time}.",
    "Which price matches the model continuation for {country} at {time}?",
    "Guess the most likely spot price for {country} at {time}.",
    "Choose the forecasted market price for {country} at {time}.",
    "What should the model predict for {country} at {time}?",
    "Select the closest future price for {country} at {time}.",
    "Which answer is nearest the forecast for {country} at {time}?",
    "At {time}, the model suggests what price for {country}?",
    "What is the expected spot price for {country} at {time}?",
    "Choose the machine-learning estimate for {country} at {time}.",
    "Find the best forecasted value for {country} at {time}.",
    "Which option is the model's predicted price for {country} at {time}?",
    "Identify the closest forecast point for {country} at {time}.",
]

def db_path():
    return "game_data.db"

def init_db():
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS plays (username TEXT PRIMARY KEY, last_play_date TEXT)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS balances (username TEXT PRIMARY KEY, coins INTEGER NOT NULL DEFAULT 0)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS portfolio (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, country TEXT NOT NULL, unit TEXT NOT NULL, cost INTEGER NOT NULL, bought_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.commit()
    conn.close()

def get_today_str():
    return pd.Timestamp.today().strftime("%Y-%m-%d")

def get_last_play_date(username):
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()
    cur.execute("SELECT last_play_date FROM plays WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def set_last_play_date(username, play_date):
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO plays(username, last_play_date) VALUES(?, ?) "
        "ON CONFLICT(username) DO UPDATE SET last_play_date=excluded.last_play_date",
        (username, play_date),
    )
    conn.commit()
    conn.close()

def get_coins(username):
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()
    cur.execute("SELECT coins FROM balances WHERE username = ?", (username,))
    row = cur.fetchone()
    if row is None:
        cur.execute("INSERT INTO balances(username, coins) VALUES(?, 0)", (username,))
        conn.commit()
        coins = 0
    else:
        coins = int(row[0])
    conn.close()
    return coins

def add_coins(username, amount):
    current = get_coins(username)
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO balances(username, coins) VALUES(?, ?) "
        "ON CONFLICT(username) DO UPDATE SET coins=?",
        (username, current + amount, current + amount),
    )
    conn.commit()
    conn.close()

def spend_coins(username, amount):
    current = get_coins(username)
    if current < amount:
        return False
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()
    cur.execute("UPDATE balances SET coins = ? WHERE username = ?", (current - amount, username))
    conn.commit()
    conn.close()
    return True

def add_portfolio_item(username, country, unit, cost):
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO portfolio(username, country, unit, cost) VALUES(?, ?, ?, ?)",
        (username, country, unit, cost),
    )
    conn.commit()
    conn.close()

def get_portfolio(username):
    conn = sqlite3.connect(db_path())
    df = pd.read_sql_query(
        "SELECT country, unit, cost, bought_at FROM portfolio WHERE username = ? ORDER BY id DESC",
        conn,
        params=(username,),
    )
    conn.close()
    return df

@st.cache_data
def load_country_data():
    files = glob.glob("../data/spot-price/*-spot-price.csv")
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

@st.cache_resource
def train_model(country):
    data = load_country_data()
    df = data[country]
    df_feat = create_features(df).dropna().reset_index(drop=True)
    X = df_feat[FEATURES]
    y = df_feat["price"]

    split = int(len(df_feat) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

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

    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, os.path.join("../models", f"lightgbm_{country}_model.pkl"))

    return model, mae, rmse

def recursive_forecast(model, df, steps=72):
    history = df[["time", "price"]].copy()
    future_rows = []
    for _ in range(steps):
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

def build_question_options(true_value):
    offsets = np.array([-18, -9, 0, 11])
    options = np.round(true_value + offsets, 1).tolist()
    random.shuffle(options)
    return options

def build_daily_questions(country, future_df, count=5):
    step = max(1, len(future_df) // (count + 1))
    templates = random.sample(QUESTION_BANK, count)
    questions = []
    for i in range(count):
        row = future_df.iloc[min((i + 1) * step - 1, len(future_df) - 1)]
        ts = pd.Timestamp(row["time"])
        questions.append({
            "country": country,
            "time_label": ts.strftime("%Y-%m-%d %H:%M"),
            "correct": float(row["predicted_price"]),
            "options": build_question_options(float(row["predicted_price"])),
            "prompt": templates[i].format(country=country, time=ts.strftime("%Y-%m-%d %H:%M")),
        })
    return questions

def draw_wheel_svg(countries, chosen=None):
    n = len(countries)
    colors = ["#ff6b6b", "#ffd166", "#06d6a0", "#4cc9f0", "#f72585", "#7209b7", "#4361ee", "#90be6d", "#f9844a"]
    cx, cy, r = 210, 210, 180
    parts = [
        '<svg width="420" height="450" viewBox="0 0 420 450" xmlns="http://www.w3.org/2000/svg">',
        '<polygon points="210,10 190,45 230,45" fill="#222"/>',
        '<g transform="translate(0,20)">'
    ]
    for i, country in enumerate(countries):
        start = 2 * math.pi * i / n - math.pi / 2
        end = 2 * math.pi * (i + 1) / n - math.pi / 2
        x1, y1 = cx + r * math.cos(start), cy + r * math.sin(start)
        x2, y2 = cx + r * math.cos(end), cy + r * math.sin(end)
        large = 1 if end - start > math.pi else 0
        path = f"M {cx} {cy} L {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f} Z"
        parts.append(f'<path d="{path}" fill="{colors[i % len(colors)]}" stroke="white" stroke-width="2"/>')
        mid = (start + end) / 2
        tx, ty = cx + (r * 0.62) * math.cos(mid), cy + (r * 0.62) * math.sin(mid)
        rotate = math.degrees(mid) + 90
        parts.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" fill="white" font-size="16" font-weight="700" text-anchor="middle" dominant-baseline="middle" transform="rotate({rotate:.1f} {tx:.1f} {ty:.1f})">{country}</text>'
        )
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="34" fill="#222"/><text x="{cx}" y="{cy}" fill="white" font-size="18" text-anchor="middle" dominant-baseline="middle">SPIN</text>')
    parts.append('</g>')
    if chosen:
        parts.append(f'<text x="210" y="432" fill="#222" font-size="26" font-weight="700" text-anchor="middle">Country: {chosen}</text>')
    parts.append('</svg>')
    return "".join(parts)