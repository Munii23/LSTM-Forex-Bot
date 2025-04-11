import os
import glob
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# === Konfiguration ===
SEQ_LEN = 50
FEATURES = ["close", "ema_50", "ema_200", "rsi", "macd_hist", "atr"]
TARGET = "target"
CAPITAL = 10000
RISK_PERCENT = 1
STOP_LOSS_PIPS = 20
RR = 2
pip_value_map = {"EURUSD": 10, "XAUUSD": 1, "BTCUSD": 1}

results = []

# === Dateipfade ===
data_dir = os.path.join("data", "processed")
model_dir = os.path.join("models")

#  Nur eine Test-Datei laden für schnelleren Debug
# files = sorted(glob.glob(os.path.join(data_dir, "*_to_today_features.csv")))
files = [os.path.join(data_dir, "BTCUSD_M30_2020_to_today_features.csv")]

for file in files:
    symbol_tf = os.path.basename(file).replace("_2020_to_today_features.csv", "")
    symbol = symbol_tf.split("_")[0]
    pip_value = pip_value_map.get(symbol, 1)

    model_path = os.path.join(model_dir, f"{symbol_tf}_2020_to_today_model.h5")
    print(f"📂 Suche Modell unter: {model_path}")
    print(f"📄 Datei: {file}")
    print(f"→ symbol_tf: {symbol_tf}")
    print(f"→ Modellpfad: {model_path}")
    print(f"→ Modell vorhanden? {os.path.exists(model_path)}")

    try:
        if not os.path.exists(model_path):
            print("❌ Modell nicht gefunden. Datei wird übersprungen.")
            continue

        df = pd.read_csv(file, index_col="time", parse_dates=True).dropna()
        data = df[FEATURES].values
        target = df[TARGET].values

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        model = load_model(model_path)

        capital = CAPITAL
        wins, losses, trades = 0, 0, 0

        print(f"🔁 Starte Backtest für {symbol_tf} mit {len(data_scaled)} Zeilen...")

        for i in range(SEQ_LEN, len(data_scaled) - 1):
            if i % 500 == 0:
                print(f"⏳ Fortschritt: {i}/{len(data_scaled)}")

            seq = np.expand_dims(data_scaled[i - SEQ_LEN:i], axis=0)
            pred = model.predict(seq, verbose=0)[0][0]
            signal = "SELL" if pred > 0.5 else "BUY"
            entry = df.iloc[i]["close"]

            pip_divisor = 10000 if symbol == "EURUSD" else 1

            if signal == "BUY":
                sl = entry - (STOP_LOSS_PIPS / pip_divisor)
                tp = entry + ((STOP_LOSS_PIPS * RR) / pip_divisor)
                stop_hit = df.iloc[i + 1]["low"] <= sl
                target_hit = df.iloc[i + 1]["high"] >= tp
            else:
                sl = entry + (STOP_LOSS_PIPS / pip_divisor)
                tp = entry - ((STOP_LOSS_PIPS * RR) / pip_divisor)
                stop_hit = df.iloc[i + 1]["high"] >= sl
                target_hit = df.iloc[i + 1]["low"] <= tp

            risk_amount = capital * RISK_PERCENT / 100
            reward_amount = risk_amount * RR

            if target_hit:
                capital += reward_amount
                wins += 1
                trades += 1
            elif stop_hit:
                capital -= risk_amount
                losses += 1
                trades += 1

        if trades > 0:
            accuracy = (wins / trades) * 100
            net_profit = capital - CAPITAL
            results.append({
                "Symbol/TF": symbol_tf,
                "Trades": trades,
                "Wins": wins,
                "Losses": losses,
                "Accuracy (%)": round(accuracy, 2),
                "Net Profit ($)": round(net_profit, 2),
                "Final Capital ($)": round(capital, 2)
            })
        else:
            print("⚠️ Keine Trades ausgeführt.")

    except Exception as e:
        print(f"❌ Fehler bei {symbol_tf}: {e}")

# === Ergebnisse anzeigen ===
df_results = pd.DataFrame(results)
print("\n📊 Backtest Ergebnisse:")
print(df_results)
