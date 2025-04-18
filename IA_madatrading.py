import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Télécharger les données BTC/USD (ou un autre actif)
ticker = "BTC-USD"
data = yf.download(ticker, period="30d", interval="1d")

# 2. Calcul de la variation globale sur 30 jours
data["Variation"] = data["Close"].pct_change()
variation_totale = float(data["Close"].iloc[-1] - data["Close"].iloc[0])
variation_pct = float((data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0] * 100)

# 3. Déterminer la tendance
if variation_totale > 0:
    tendance = "Haussière 📈"
elif variation_totale < 0:
    tendance = "Baissière 📉"
else:
    tendance = "Neutre ➖"

print(f"Tendance sur les 30 derniers jours pour {ticker} : {tendance} ({variation_pct:.2f}%)")

# 4. Tracer le graphique
plt.figure(figsize=(10, 5))
plt.plot(data["Close"], label="Prix de clôture")

x = np.arange(len(data["Close"]))
y = data["Close"].to_numpy().flatten()
coeffs = np.polyfit(x, y, 1)
trend_line = np.poly1d(coeffs)
plt.plot(data.index, trend_line(x), color='red', linestyle='--', label="Tendance linéaire")

# 5. Prédiction du prix au jour suivant (J+1)
next_day_index = len(data["Close"])
predicted_price = trend_line(next_day_index)
print(f"Prix prédit pour demain ({ticker}) : {predicted_price:.2f} USD")

# Ajouter la prédiction sur le graphique
plt.scatter(data.index[-1] + pd.Timedelta(days=1), predicted_price, color='orange', label='Prévision J+1')

plt.title(f"{ticker} - 30 derniers jours")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================== ETHEREUM ======================
ticker = "ETH-USD"
data = yf.download(ticker, period="30d", interval="1d")

data["Variation"] = data["Close"].pct_change()
variation_totale = float(data["Close"].iloc[-1] - data["Close"].iloc[0])
variation_pct = float((data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0] * 100)

if variation_totale > 0:
    tendance = "Haussière 📈"
elif variation_totale < 0:
    tendance = "Baissière 📉"
else:
    tendance = "Neutre ➖"

print(f"\nTendance sur les 30 derniers jours pour {ticker} : {tendance} ({variation_pct:.2f}%)")

plt.figure(figsize=(10, 5))
plt.plot(data["Close"], label="Prix de clôture ETH")

x = np.arange(len(data["Close"]))
y = data["Close"].to_numpy().flatten()
coeffs = np.polyfit(x, y, 1)
trend_line = np.poly1d(coeffs)
plt.plot(data.index, trend_line(x), color='red', linestyle='--', label="Tendance linéaire")

# Prédiction J+1
next_day_index = len(data["Close"])
predicted_price = trend_line(next_day_index)
print(f"Prix prédit pour demain ({ticker}) : {predicted_price:.2f} USD")
plt.scatter(data.index[-1] + pd.Timedelta(days=1), predicted_price, color='orange', label='Prévision J+1')

# Prédiction J+3
future_day_index = len(data["Close"]) + 2
predicted_price_j3 = trend_line(future_day_index)
print(f"Prix prédit dans 3 jours ({ticker}) : {predicted_price_j3:.2f} USD")
plt.scatter(data.index[-1] + pd.Timedelta(days=3), predicted_price_j3, color='purple', label='Prévision J+3')

plt.title(f"{ticker} - 30 derniers jours")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()