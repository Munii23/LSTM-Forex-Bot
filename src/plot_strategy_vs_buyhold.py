import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# === Einstellungen ===
SEQ_LEN = 50
FEATURES = ["close", "ema_50", "ema_200", "rsi", "macd_hist", "atr"]
TARGET = "target"
START_CAPITAL = 10000
RISK_PERCENT = 1
RR = 2
PIP_DIVISOR = {"EURUSD": 10000, "XAUUSD": 1, "BTCUSD": 1}

# === Pfade ===
project_root = os.path.dirname(os.path.dirname(__file__))
features_dir = os.path.join(project_root, "data", "processed")
models_dir = os.path.join(project_root, "models")
output_dir = os.path.join(project_root, "comparison_plots")
os.makedirs(output_dir, exist_ok=True)

# === Farbschema
import matplotlib.cm as cm
color_map = plt.get_cmap("tab20")

# === Plot Funktion ===
def plot_comparison(df, equity_curve, buy_hold_curve, name):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[SEQ_LEN + 1:], equity_curve, label="LSTM Strategy", color="green")
    plt.plot(df.index, buy_hold_curve, label="Buy & Hold", color="blue")
    plt.title(f"Vergleich: {name}")
    plt.xlabel("Zeit")
    plt.ylabel("Kontostand ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}.png"))
    plt.close()

# === Vorbereitung für gesamtvergleich
file_list = [f for f in os.listdir(features_dir) if f.endswith("_features.csv")]
buy_hold_lines = {}
lstm_lines = {}

# === Schleife über alle Dateien ===
for idx, fname in enumerate(file_list):
    raw_name = fname.replace("_features.csv", "")
    model_path = os.path.join(models_dir, raw_name + "_model.h5")
    file_path = os.path.join(features_dir, fname)
    if not os.path.exists(model_path):
        continue

    symbol = raw_name.split("_")[0]
    pip_div = PIP_DIVISOR.get(symbol, 1)

    df = pd.read_csv(file_path, index_col="time", parse_dates=True).dropna()
    data = df[FEATURES].values
    target = df[TARGET].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    model = load_model(model_path)

    capital = START_CAPITAL
    equity_curve = []

    for i in range(SEQ_LEN, len(data_scaled) - 1):
        seq = np.expand_dims(data_scaled[i - SEQ_LEN:i], axis=0)
        pred = model.predict(seq, verbose=0)[0][0]
        signal = "SELL" if pred > 0.5 else "BUY"
        entry = df.iloc[i]["close"]

        if signal == "BUY":
            sl = entry - (20 / pip_div)
            tp = entry + (20 * RR / pip_div)
            stop_hit = df.iloc[i + 1]["low"] <= sl
            target_hit = df.iloc[i + 1]["high"] >= tp
        else:
            sl = entry + (20 / pip_div)
            tp = entry - (20 * RR / pip_div)
            stop_hit = df.iloc[i + 1]["high"] >= sl
            target_hit = df.iloc[i + 1]["low"] <= tp

        risk_amount = capital * RISK_PERCENT / 100
        reward = risk_amount * RR

        if target_hit:
            capital += reward
        elif stop_hit:
            capital -= risk_amount
        equity_curve.append(capital)

    # Buy-and-Hold-Kurve
    start_price = df.iloc[SEQ_LEN]["close"]
    buy_hold_curve = df["close"] / start_price * START_CAPITAL

    # Einzelplot
    plot_comparison(df, equity_curve, buy_hold_curve.values, raw_name)

    # Für Sammelplot
    buy_hold_lines[raw_name] = (df.index, buy_hold_curve.values, color_map(idx % 20))
    lstm_lines[raw_name] = (df.index[SEQ_LEN:-1], equity_curve, color_map(idx % 20))

# === Vergleich aller Buy & Hold Strategien
plt.figure(figsize=(14, 7))
for name, (index, values, color) in buy_hold_lines.items():
    plt.plot(index, values, label=name, color=color)
plt.title("Buy & Hold Performance – Alle Timeframes & Symbole")
plt.xlabel("Zeit")
plt.ylabel("Kontostand ($)")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ALL_BUY_AND_HOLD_COMPARISON.png"))
plt.close()

# === Vergleich aller LSTM Strategien
plt.figure(figsize=(14, 7))
for name, (index, values, color) in lstm_lines.items():
    plt.plot(index, values, label=name, color=color)
plt.title("LSTM Strategie Performance – Alle Timeframes & Symbole")
plt.xlabel("Zeit")
plt.ylabel("Kontostand ($)")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ALL_LSTM_STRATEGY_COMPARISON.png"))
plt.close()

print("✅ Alle Diagramme wurden gespeichert unter: comparison_plots/")
