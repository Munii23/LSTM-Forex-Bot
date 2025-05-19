import os
import pandas as pd
import ta  # technischer Indikatoren-Toolkit

# === Eingabe- & Ausgabeverzeichnis ===
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
output_dir = os.path.join(data_dir, "processed")
os.makedirs(output_dir, exist_ok=True)


# === Technische Indikatoren hinzufügen ===
def add_indicators(df):
    df = df.copy()

    # Berechnungen
    df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd = ta.trend.macd_diff(df["close"])
    df["macd_hist"] = macd
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    # Zielwert: Wenn nächste Kerze höher schließt → 0 (BUY), sonst → 1 (SELL)
    df["target"] = df["close"].shift(-1) < df["close"]
    df["target"] = df["target"].astype(int)

    # Zeit als Index (optional)
    df.set_index("time", inplace=True)

    return df.dropna()


# === Alle CSV-Dateien im Ordner verarbeiten ===
for file in os.listdir(data_dir):
    if not file.endswith(".csv"):
        continue

    print(f"Verarbeite Datei: {file}")
    filepath = os.path.join(data_dir, file)
    df = pd.read_csv(filepath, parse_dates=["time"])

    df_features = add_indicators(df)

    out_name = file.replace(".csv", "_features.csv")
    out_path = os.path.join(output_dir, out_name)
    df_features.to_csv(out_path)
    print(f"Gespeichert unter: {out_path}")

print(" Preprocessing abgeschlossen!")
