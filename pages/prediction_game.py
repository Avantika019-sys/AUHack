import glob
import os
import random
import sqlite3

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb

st.set_page_config(page_title="Spot Price Game", layout="wide")

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "data", "spot-price")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DB_PATH = os.path.join(BASE_DIR, "game_data.db")

# =========================================================
# CONSTANTS
# =========================================================

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
    "1 MWh Electricity": 10,
    "5 MWh Electricity": 30,
    "10 MWh Electricity": 150,
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
]

# =========================================================
# DATABASE
# =========================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS plays (
            username TEXT PRIMARY KEY,
            last_play_date TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS balances (
            username TEXT PRIMARY KEY,
            coins INTEGER NOT NULL DEFAULT 0
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            country TEXT NOT NULL,
            unit TEXT NOT NULL,
            cost INTEGER NOT NULL,
            bought_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def get_today_str():
    return pd.Timestamp.today().strftime("%Y-%m-%d")


def get_last_play_date(username):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT last_play_date FROM plays WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def set_last_play_date(username, play_date):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO plays(username, last_play_date) VALUES(?, ?) "
        "ON CONFLICT(username) DO UPDATE SET last_play_date=excluded.last_play_date",
        (username, play_date),
    )
    conn.commit()
    conn.close()


def get_coins(username):
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE balances SET coins = ? WHERE username = ?", (current - amount, username))
    conn.commit()
    conn.close()
    return True


def add_portfolio_item(username, country, unit, cost):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO portfolio(username, country, unit, cost) VALUES(?, ?, ?, ?)",
        (username, country, unit, cost),
    )
    conn.commit()
    conn.close()


def get_portfolio(username):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT country, unit, cost, bought_at FROM portfolio WHERE username = ? ORDER BY id DESC",
        conn,
        params=(username,),
    )
    conn.close()
    return df

# =========================================================
# DATA + MODEL
# =========================================================

#@st.cache_data
def load_country_data():
    files = glob.glob(os.path.join(DATASETS_DIR, "*-spot-price.csv"))
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
def train_model(country, df):
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

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, f"lightgbm_{country}_model.pkl"))

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


def build_questions(country, future_df, count=3):
    step = max(1, len(future_df) // (count + 1))
    templates = random.sample(QUESTION_BANK, count)

    questions = []
    for i in range(count):
        row = future_df.iloc[min((i + 1) * step - 1, len(future_df) - 1)]
        ts = pd.Timestamp(row["time"])
        questions.append({
            "country": country,
            "time": ts,
            "time_label": ts.strftime("%Y-%m-%d %H:%M"),
            "correct": float(row["predicted_price"]),
            "options": build_question_options(float(row["predicted_price"])),
            "prompt": templates[i].format(
                country=country,
                time=ts.strftime("%Y-%m-%d %H:%M")
            ),
        })
    return questions

# =========================================================
# APP STATE
# =========================================================

init_db()

defaults = {
    "username": "",
    "questions": [],
    "quiz_submitted": False,
    "quiz_score": 0,
    "challenge_country": None,
}

for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

data = load_country_data()
if not data:
    st.error(f"No files found in {DATASETS_DIR}")
    st.stop()

countries = sorted(data.keys())

# =========================================================
# HEADER + DASHBOARD
# =========================================================

st.title("Spot Price Game")

top1, top2 = st.columns([1.5, 1])
with top1:
    selected_country = st.selectbox("Dashboard country", countries, index=0)
with top2:
    history_days = st.selectbox("Dashboard history window", [7, 14, 30, 60, 90], index=2)

dashboard_df = data[selected_country]
dashboard_model, dashboard_mae, dashboard_rmse = train_model(selected_country, dashboard_df)

m1, m2, m3 = st.columns(3)
m1.metric("Selected country", selected_country)
m2.metric("MAE", f"{dashboard_mae:.2f}")
m3.metric("RMSE", f"{dashboard_rmse:.2f}")

st.subheader("Spot Price Dashboard")

history_cutoff = dashboard_df["time"].max() - pd.Timedelta(days=history_days)
hist_plot = dashboard_df[dashboard_df["time"] >= history_cutoff].copy()

dashboard_chart = pd.DataFrame({
    "time": hist_plot["time"],
    "Spot price": hist_plot["price"].astype(float),
}).set_index("time")

st.line_chart(dashboard_chart, height=360)

# =========================================================
# TABS
# =========================================================

challenge_tab, reward_tab = st.tabs(["Daily Challenge", "Reward Shop"])

# =========================================================
# DAILY CHALLENGE
# =========================================================

with challenge_tab:
    st.subheader("Daily Spot Price Challenge")
    st.write("Answer today's 3 questions about predicted spot prices.")

    username = st.text_input("Enter your name to continue", value=st.session_state.username)
    st.session_state.username = username.strip()

    if st.session_state.username:
        today = get_today_str()
        last_play_date = get_last_play_date(st.session_state.username)
        already_played = last_play_date == today

        st.metric("Your coins", get_coins(st.session_state.username))

        challenge_country = st.selectbox(
            "Challenge country",
            countries,
            index=0,
            key="daily_challenge_country_select"
        )

        if st.session_state.challenge_country != challenge_country:
            st.session_state.challenge_country = challenge_country
            st.session_state.questions = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = 0

        if already_played:
            st.info("You already completed today's challenge. Come back tomorrow.")
        else:
            quiz_df = data[challenge_country]
            quiz_model, _, _ = train_model(challenge_country, quiz_df)
            quiz_future = recursive_forecast(quiz_model, quiz_df, steps=48)

            if not st.session_state.questions:
                st.session_state.questions = build_questions(
                    challenge_country,
                    quiz_future,
                    count=3
                )

            answers = []
            for i, q in enumerate(st.session_state.questions):
                st.markdown(f"**Question {i + 1}**")
                st.write(q["prompt"])
                answer = st.radio(
                    f"Answer {i + 1}",
                    q["options"],
                    horizontal=True,
                    key=f"question_{i}"
                )
                answers.append(float(answer))

            if st.button("Submit all answers", type="primary"):
                score = 0
                for answer, q in zip(answers, st.session_state.questions):
                    best = min(q["options"], key=lambda x: abs(x - q["correct"]))
                    if float(answer) == float(best):
                        score += 10

                st.session_state.quiz_score = score
                st.session_state.quiz_submitted = True
                add_coins(st.session_state.username, score)
                set_last_play_date(st.session_state.username, today)
                st.balloons()

        if st.session_state.quiz_submitted:
            reveal_df = data[challenge_country]
            reveal_model, _, _ = train_model(challenge_country, reveal_df)
            reveal_future = recursive_forecast(reveal_model, reveal_df, steps=24)

            c1, c2 = st.columns(2)
            c1.metric("Points earned today", st.session_state.quiz_score)
            c2.metric("Total coins", get_coins(st.session_state.username))

            st.markdown("### Forecast Reveal Dashboard")

            reveal_cutoff = reveal_df["time"].max() - pd.Timedelta(days=30)
            reveal_hist = reveal_df[reveal_df["time"] >= reveal_cutoff].copy()

            reveal_chart = pd.DataFrame({
                "time": pd.concat([reveal_hist["time"], reveal_future["time"]], ignore_index=True),
                "Current / historical price": pd.concat([
                    reveal_hist["price"].astype(float),
                    pd.Series([np.nan] * len(reveal_future), dtype=float)
                ], ignore_index=True),
                "Prediction continuation": pd.concat([
                    pd.Series([np.nan] * len(reveal_hist), dtype=float),
                    reveal_future["predicted_price"].astype(float)
                ], ignore_index=True),
            }).set_index("time")

            st.line_chart(reveal_chart, height=380)

            with st.expander("Show correct answers"):
                for i, q in enumerate(st.session_state.questions):
                    st.write(f"Q{i + 1}: {q['time_label']} → {q['correct']:.1f} EUR/MWh")
    else:
        st.info("Enter your name to start today's challenge.")

# =========================================================
# REWARD SHOP
# =========================================================

with reward_tab:
    st.subheader("Reward Shop")
    st.write("Spend your points on units in the countries you believe will increase.")

    shop_username = st.text_input(
        "Player name",
        value=st.session_state.username,
        key="shop_name"
    )

    if shop_username.strip():
        clean_name = shop_username.strip()
        coins = get_coins(clean_name)
        st.metric("Available coins", coins)

        c1, c2, c3 = st.columns(3)
        with c1:
            buy_country = st.selectbox("Country", countries, key="shop_country")
        with c2:
            buy_unit = st.selectbox("Unit type", list(UNIT_PRICES.keys()))
        with c3:
            unit_cost = UNIT_PRICES[buy_unit]
            st.metric("Unit cost", f"{unit_cost} coins")

        if st.button("Buy unit"):
            if spend_coins(clean_name, unit_cost):
                add_portfolio_item(clean_name, buy_country, buy_unit, unit_cost)
                st.success(f"Bought 1 {buy_unit} for {buy_country}")
            else:
                st.error("Not enough coins")

        st.markdown("### Owned units")
        portfolio = get_portfolio(clean_name)
        if len(portfolio):
            st.dataframe(portfolio, use_container_width=True)
        else:
            st.info("You have not bought any units yet.")
    else:
        st.info("Enter your player name to open the shop.")
