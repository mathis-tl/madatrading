"""
Configuration globale pour l'application d'analyse de crypto-monnaies.
"""

# Configuration API
API_ENDPOINTS = {
    "coingecko": "https://api.coingecko.com/api/v3",
    "binance": "https://api.binance.com"
}

# Choix de l'API à utiliser (options: 'coingecko', 'binance')
DEFAULT_API = "coingecko"

# Configuration base de données
DATABASE_URL = "sqlite:///crypto_data.db"

# Paramètres du modèle de prédiction
MODEL_CONFIG = {
    "window_size": 60,  # Nombre de jours d'historique à utiliser
    "features": ["price", "volume", "market_cap", "rsi", "ma_7", "ma_30"],
    "target": "price",
    "split_ratio": 0.8,  # Ratio d'entraînement/test
    "epochs": 50,
    "batch_size": 32,
    "neurons": [64, 32, 16]
}

# Cryptocurrencies à analyser
CRYPTO_SYMBOLS = ["BTC", "ETH", "XRP", "ADA", "SOL"]

# Période d'analyse
TIMEFRAME = "daily"  # options: 'hourly', 'daily', 'weekly'

# Chemins des fichiers de données
DATA_DIR = "data"
MODEL_SAVE_PATH = "models/saved_models"