import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

SEQ_LEN = 50
FEATURES = ["close", "ema_50", "ema_200", "rsi", "macd_hist", "atr"]

# Pfade
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
features_dir = os.path.join(project_root, "data", "processed")
models_dir = os.path.join(project_root, "models")
signals_dir = os.path.join(project_root, "signals")
os.makedirs(signals_dir, exist_ok=True)

# Alle passenden Feature-Dateien durchgehen
for fname in os.listdir(features_dir):
    if not fname.endswith("_features.csv"):
        continue

    # Name ohne "_features.csv"
    raw_name = fname.replace("_features.csv", "")

    # Kürze "_2020_to_today" raus, um WÄHRUNG_TIMEFRAME zu bekommen
    short_name = raw_name.split("_2020")[0]

    # Pfade
    input_csv = os.path.join(features_dir, fname)
    model_path = os.path.join(models_dir, raw_name + "_model.h5")
    output_csv = os.path.join(signals_dir, f"{short_name}_signals.csv")

    # Modellprüfung
    if not os.path.exists(model_path):
        print(f" Modell nicht gefunden für: {fname}")
        continue

    print(f"\n {short_name}: Starte Verarbeitung...")

    # Daten laden
    df = pd.read_csv(input_csv, index_col="time", parse_dates=True).dropna()
    data = df[FEATURES].values

    # Normalisieren
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Modell laden
    model = load_model(model_path)

    # Signale generieren
    signals = []
    for i in range(SEQ_LEN, len(data_scaled)):
        seq = np.expand_dims(data_scaled[i - SEQ_LEN:i], axis=0)
        pred = model.predict(seq, verbose=0)[0][0]
        signal = 1 if pred > 0.5 else -1
        time = df.index[i].strftime("%Y.%m.%d %H:%M:%S")
        close = df.iloc[i]["close"]
        signals.append({"time": time, "close": close, "signal": signal})

    # Speichern
    pd.DataFrame(signals).to_csv(output_csv, index=False)
    print(f"Gespeichert unter: {output_csv}")
