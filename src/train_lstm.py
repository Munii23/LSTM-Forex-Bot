import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# === Parameter ===
SEQ_LEN = 50
FEATURES = ["close", "ema_50", "ema_200", "rsi", "macd_hist", "atr"]
TARGET = "target"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "EURUSD_H1_2020_to_today_features.csv")

# === Daten laden ===
df = pd.read_csv(DATA_PATH, index_col="time", parse_dates=True)
df = df.dropna()
data = df[FEATURES].values
target = df[TARGET].values

# === Normalisieren ===
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# === Sequenzen bauen ===
X, y = [], []
for i in range(SEQ_LEN, len(data_scaled)):
    X.append(data_scaled[i - SEQ_LEN:i])
    y.append(target[i])
X, y = np.array(X), np.array(y)

# === Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === LSTM-Modell ===
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(FEATURES))))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))  # Binär: 0 = Buy, 1 = Sell

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# === Training ===
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# === Auswertung ===
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTestgenauigkeit: {accuracy * 100:.2f}%")
