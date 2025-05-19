import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os

# === Einstellungen ===
symbols = ["EURUSD", "XAUUSD", "BTCUSD"]
timeframes = {
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

from_date = datetime(2020, 1, 1)
to_date = datetime.now()

# === MT5 initialisieren ===
if not mt5.initialize():
    print("MetaTrader5 konnte nicht gestartet werden:", mt5.last_error())
    quit()
print("MetaTrader5 Verbindung erfolgreich!")

# === Speicherordner vorbereiten ===
output_folder = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(output_folder, exist_ok=True)

# === Daten abrufen ===
for symbol in symbols:
    for tf_name, tf_value in timeframes.items():
        print(f"Lade Daten für {symbol} - {tf_name} ...")
        rates = mt5.copy_rates_range(symbol, tf_value, from_date, to_date)
        if rates is None or len(rates) == 0:
            print(f"Keine Daten gefunden für {symbol} - {tf_name}")
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        filename = f"{symbol}_{tf_name}_2020_to_today.csv"
        filepath = os.path.join(output_folder, filename)
        df.to_csv(filepath, index=False)
        print(f" Gespeichert unter: {filepath}")

# === Verbindung beenden ===
mt5.shutdown()
print(" Alle Daten erfolgreich gespeichert!")
