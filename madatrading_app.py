import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Madatrading AI", layout="wide")

# Fonction RSI
def compute_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Titre principal
st.title("📈 Analyse IA – Madatrading")

# Choix de l’actif
ticker = st.selectbox("Choisir un actif :", ["BTC-USD", "ETH-USD"])

# Récupération des données
data = yf.download(ticker, period="730d", interval="1d")
data["Variation"] = data["Close"].pct_change()
variation_totale = float(data["Close"].iloc[-1] - data["Close"].iloc[0])
variation_pct = float((data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0] * 100)

# RSI et SMA
data["RSI"] = compute_rsi(data)
data["SMA_10"] = data["Close"].rolling(window=10).mean()

# Momentum (10 jours)
data["Momentum_10"] = data["Close"] - data["Close"].shift(10)

# Variation sur 7 jours (%)
data["Var_7d"] = data["Close"].pct_change(periods=7) * 100

# Stochastic RSI (14 jours)
min_rsi = data["RSI"].rolling(window=14).min()
max_rsi = data["RSI"].rolling(window=14).max()
data["StochRSI"] = (data["RSI"] - min_rsi) / (max_rsi - min_rsi)

# EMA
data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

# MACD
exp1 = data["Close"].ewm(span=12, adjust=False).mean()
exp2 = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = exp1 - exp2
data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

# Bandes de Bollinger (20 jours)
data["BB_MIDDLE"] = data["Close"].rolling(window=20).mean()
data["BB_STD"] = data["Close"].rolling(window=20).std()
data["BB_UPPER"] = data["BB_MIDDLE"] + 2 * data["BB_STD"]
data["BB_LOWER"] = data["BB_MIDDLE"] - 2 * data["BB_STD"]

# ATR (Average True Range - 14 jours)
data["H-L"] = data["High"] - data["Low"]
data["H-PC"] = abs(data["High"] - data["Close"].shift())
data["L-PC"] = abs(data["Low"] - data["Close"].shift())
data["TR"] = data[["H-L", "H-PC", "L-PC"]].max(axis=1)
data["ATR"] = data["TR"].rolling(window=14).mean()

# Dataset pour prédiction supervisée (Target = prix J+1)
feature_columns = [
    "Close", "RSI", "SMA_10", "EMA_10", "EMA_20",
    "MACD", "Signal", "BB_UPPER", "BB_LOWER", "ATR",
    "Momentum_10", "Var_7d", "StochRSI"
]

# Supprimer les lignes avec NaN dans les features
df_model = data[feature_columns].copy()
horizon = st.selectbox("Horizon de prédiction :", ["J+1", "J+3", "J+7", "J+10"])
horizon_offset = int(horizon.replace("J+", ""))
target_col = f"Target_{horizon}"
df_model[target_col] = data["Close"].shift(-horizon_offset)
df_model = df_model.dropna()
X = df_model[feature_columns]
y = df_model[target_col]

# Prédiction J+7 avec Linear Regression
test_size_ratio = 0.5 if len(X) >= 10 else 0.25 if len(X) >= 4 else 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, shuffle=False)
X_train.index = y_train.index
X_test.index = y_test.index

# Entraînement du modèle
# Sélection du modèle IA
model_type = st.selectbox("Modèle IA :", ["Régression Linéaire", "Random Forest", "Ridge", "Gradient Boosting", "XGBoost"])
if model_type == "Régression Linéaire":
    model = LinearRegression()
elif model_type == "Random Forest":
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
elif model_type == "Ridge":
    model = Ridge(alpha=1.0)
elif model_type == "Gradient Boosting":
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
elif model_type == "XGBoost":
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Prédiction sur le dernier jour connu
last_input = X.tail(1)
predicted_j7 = model.predict(last_input)[0]

# Prédiction sur tous les X_test
y_pred = model.predict(X_test)

# Diagnostics de debug
st.write("🧪 Taille de X_test :", len(X_test))
st.write("y_test (fin) :", y_test.tail())
st.write("y_pred (fin) :", y_pred[-5:])

# Calcule les erreurs et scores
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"===== Évaluation du modèle IA ({horizon}) =====")
print(f"🔹 Modèle sélectionné : {model_type}")
print(f"🔹 MAE (Erreur absolue moyenne)  : {mae:.2f} USD")
print(f"🔹 RMSE (Erreur quadratique moy.) : {rmse:.2f} USD")
print(f"🔹 R² (détermination) : {r2:.4f}")
print("========================================\n")

# Affiche les résultats dans Streamlit
st.markdown("### 📊 Évaluation du modèle IA (J+7)")
st.write(f"🔹 Erreur absolue moyenne (MAE) : {mae:.2f} USD")
st.write(f"🔹 Racine de l'erreur quadratique moyenne (RMSE) : {rmse:.2f} USD")
st.write(f"🔹 Coefficient de détermination (R²) : {r2:.2f}")

# Tendance
if variation_totale > 0:
    tendance = "Haussière 📈"
elif variation_totale < 0:
    tendance = "Baissière 📉"
else:
    tendance = "Neutre ➖"

st.markdown(f"### Tendance sur 30 jours : **{tendance} ({variation_pct:.2f}%)**")

# Affichage du dataset d'entraînement
if st.checkbox("Afficher le dataset d'entraînement"):
    st.dataframe(df_model.tail(10))

# Courbe + tendance linéaire
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data["Close"], label="Prix de clôture")

x = np.arange(len(data["Close"]))
y = data["Close"].to_numpy().flatten()
coeffs = np.polyfit(x, y, 1)
trend_line = np.poly1d(coeffs)
ax.plot(data.index, trend_line(x), linestyle='--', color='red', label="Tendance linéaire")

# Prédictions
next_day_index = len(data["Close"])
predicted_price = trend_line(next_day_index)
future_day_index = next_day_index + 2
predicted_price_j3 = trend_line(future_day_index)

ax.scatter(data.index[-1] + pd.Timedelta(days=1), predicted_price, color='orange', label='Prévision J+1')
ax.scatter(data.index[-1] + pd.Timedelta(days=3), predicted_price_j3, color='purple', label='Prévision J+3')

# Ajout de la prédiction IA J+7 au graphique
j7_date = data.index[-1] + pd.Timedelta(days=7)
ax.scatter(j7_date, predicted_j7, color='deepskyblue', label='Prédiction IA J+7')

ax.set_title(f"{ticker} - 30 derniers jours")
ax.set_xlabel("Date")
ax.set_ylabel("Prix")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# Affichage prédictions
st.write(f"📍 Prix prédit J+1 : **{predicted_price:.2f} USD**")
st.write(f"📍 Prix prédit J+3 : **{predicted_price_j3:.2f} USD**")

# Graphique de comparaison entre réel et prédit
if len(y_test) < 2:
    st.warning("Pas assez de données pour afficher la comparaison.")
else:
    fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
    test_dates = X_test.index
    ax_comp.plot(test_dates, y_test, label="Prix réel", color="blue", marker='o')
    ax_comp.plot(test_dates, y_pred, label="Prédiction IA J+7", color="orange", linestyle="--", marker='x')
    ax_comp.set_title("Comparaison : IA J+7 vs Réel (test set)")
    ax_comp.set_xlabel("Date")
    ax_comp.set_ylabel("Prix")
    ax_comp.legend()
    ax_comp.grid(True)
    fig_comp.autofmt_xdate()
    st.pyplot(fig_comp)

st.markdown("## 📊 Indicateurs techniques")

show_sma = st.checkbox("Afficher SMA 10", value=True)
show_rsi = st.checkbox("Afficher RSI", value=True)
show_ema = st.checkbox("Afficher EMA 10 & 20", value=True)
show_macd = st.checkbox("Afficher MACD", value=True)
show_bollinger = st.checkbox("Afficher Bandes de Bollinger", value=True)
show_atr = st.checkbox("Afficher ATR", value=True)

# Graphique SMA
if show_sma:
    fig_sma, ax_sma = plt.subplots(figsize=(10, 4))
    ax_sma.plot(data.index, data["Close"], label="Prix de clôture")
    ax_sma.plot(data.index, data["SMA_10"], label="SMA 10 jours", linestyle="--")
    ax_sma.set_title("Prix + Moyenne mobile (SMA 10 jours)")
    ax_sma.set_xlabel("Date")
    ax_sma.set_ylabel("Prix")
    ax_sma.legend()
    ax_sma.grid(True)
    st.pyplot(fig_sma)

# Graphique RSI
if show_rsi:
    fig_rsi, ax_rsi = plt.subplots(figsize=(10, 2))
    ax_rsi.plot(data.index, data["RSI"], label="RSI", color="green")
    ax_rsi.axhline(70, color='red', linestyle='--')
    ax_rsi.axhline(30, color='blue', linestyle='--')
    ax_rsi.set_title("Indice de force relative (RSI)")
    ax_rsi.set_xlabel("Date")
    ax_rsi.set_ylabel("RSI")
    ax_rsi.grid(True)
    st.pyplot(fig_rsi)

# Graphique EMA
if show_ema:
    fig_ema, ax_ema = plt.subplots(figsize=(10, 4))
    ax_ema.plot(data.index, data["Close"], label="Prix de clôture")
    ax_ema.plot(data.index, data["EMA_10"], label="EMA 10", linestyle="--")
    ax_ema.plot(data.index, data["EMA_20"], label="EMA 20", linestyle="--")
    ax_ema.set_title("Prix + Moyennes mobiles exponentielles (EMA 10 & 20)")
    ax_ema.set_xlabel("Date")
    ax_ema.set_ylabel("Prix")
    ax_ema.legend()
    ax_ema.grid(True)
    st.pyplot(fig_ema)

# Graphique MACD
if show_macd:
    fig_macd, ax_macd = plt.subplots(figsize=(10, 3))
    ax_macd.plot(data.index, data["MACD"], label="MACD", color="purple")
    ax_macd.plot(data.index, data["Signal"], label="Signal", color="orange", linestyle="--")
    ax_macd.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax_macd.set_title("MACD (Moving Average Convergence Divergence)")
    ax_macd.set_xlabel("Date")
    ax_macd.set_ylabel("Valeur")
    ax_macd.legend()
    ax_macd.grid(True)
    st.pyplot(fig_macd)

# Graphique Bandes de Bollinger
if show_bollinger:
    fig_bb, ax_bb = plt.subplots(figsize=(10, 4))
    ax_bb.plot(data.index, data["Close"], label="Prix de clôture", color='blue')
    ax_bb.plot(data.index, data["BB_MIDDLE"], label="Moyenne 20j", linestyle="--", color='black')
    ax_bb.plot(data.index, data["BB_UPPER"], label="Bande Supérieure", linestyle="--", color='green')
    ax_bb.plot(data.index, data["BB_LOWER"], label="Bande Inférieure", linestyle="--", color='red')
    ax_bb.fill_between(data.index, data["BB_LOWER"], data["BB_UPPER"], color='grey', alpha=0.1)
    ax_bb.set_title("Bandes de Bollinger (20 jours)")
    ax_bb.set_xlabel("Date")
    ax_bb.set_ylabel("Prix")
    ax_bb.legend()
    ax_bb.grid(True)
    st.pyplot(fig_bb)

# Graphique ATR
if show_atr:
    fig_atr, ax_atr = plt.subplots(figsize=(10, 2))
    ax_atr.plot(data.index, data["ATR"], label="ATR (14 jours)", color="darkred")
    ax_atr.set_title("ATR – Indicateur de Volatilité (14 jours)")
    ax_atr.set_xlabel("Date")
    ax_atr.set_ylabel("ATR")
    ax_atr.grid(True)
    ax_atr.legend()
    st.pyplot(fig_atr)

# (Optionnel) Prévoir J+10 également :
df_model_j10 = data[feature_columns].copy()
df_model_j10["Target_J10"] = data["Close"].shift(-10)
df_model_j10 = df_model_j10.dropna()
X_full = df_model_j10[feature_columns]
y_j10 = df_model_j10["Target_J10"]
model_j10 = LinearRegression()
model_j10.fit(X_full, y_j10)
predicted_j10 = model_j10.predict(X_full.tail(1))[0]
st.write(f"📍 Prix prédit J+10 (modèle IA) : **{predicted_j10:.2f} USD**")
