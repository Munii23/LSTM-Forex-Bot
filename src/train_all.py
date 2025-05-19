import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# === Konfiguration ===
SEQ_LEN = 50
FEATURES = ["close", "ema_50", "ema_200", "rsi", "macd_hist", "atr"]
TARGET = "target"

# Datenpfade vorbereiten
base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(model_dir, exist_ok=True)  # Ordner erstellen falls nicht vorhanden

files = sorted(glob.glob(os.path.join(base_dir, "*.csv")))

for file in files:
    filename = os.path.basename(file)
    print(f"\n Trainiere Modell für: {filename}")

    try:
        # Daten einlesen
        df = pd.read_csv(file, index_col="time", parse_dates=True).dropna()
        data = df[FEATURES].values
        target = df[TARGET].values

        # Skalieren
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Sequenzen vorbereiten
        X, y = [], []
        for i in range(SEQ_LEN, len(data_scaled)):
            X.append(data_scaled[i - SEQ_LEN:i])
            y.append(target[i])
        X, y = np.array(X), np.array(y)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Modell
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(FEATURES))),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1, activation="sigmoid")
        ])

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)

        # Auswertung
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Genauigkeit: {acc * 100:.2f}%")

        #  Modell speichern
        model_name = filename.replace("_features.csv", "_model.h5")
        model_path = os.path.join(model_dir, model_name)
        model.save(model_path)
        print(f" Modell gespeichert unter: {model_path}")

    except Exception as e:
        print(f" Fehler bei Datei {filename}: {e}")
