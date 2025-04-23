"""
Module pour collecter les données depuis différentes API de crypto-monnaies.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta

class ApiCollector:
    """Classe pour collecter des données depuis des API de crypto."""
    
    def __init__(self, api_url):
        """
        Initialise le collecteur d'API.
        
        Args:
            api_url (str): URL de base de l'API.
        """
        self.api_url = api_url
        self.session = requests.Session()
    
    def fetch_data(self, symbol, days=30):
        """
        Récupère les données historiques pour une crypto-monnaie.
        
        Args:
            symbol (str): Symbole de la crypto-monnaie (ex: BTC, ETH).
            days (int): Nombre de jours d'historique à récupérer.
            
        Returns:
            pandas.DataFrame: DataFrame contenant l'historique des prix.
        """
        # Détecte si c'est l'API CoinGecko ou Binance
        if "coingecko" in self.api_url:
            return self._fetch_from_coingecko(symbol, days)
        elif "binance" in self.api_url:
            return self._fetch_from_binance(symbol, days)
        else:
            raise ValueError(f"API non supportée: {self.api_url}")
    
    def _fetch_from_coingecko(self, symbol, days):
        """Récupère les données depuis CoinGecko."""
        # Convertir le symbole au format attendu par CoinGecko
        mapping = {"BTC": "bitcoin", "ETH": "ethereum", "XRP": "ripple", 
                   "ADA": "cardano", "SOL": "solana"}
        coin_id = mapping.get(symbol.upper(), symbol.lower())
        
        endpoint = f"/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily"
        }
        
        url = f"{self.api_url}{endpoint}"
        response = self.session.get(url, params=params)
        
        if response.status_code != 200:
            raise ConnectionError(f"Erreur API: {response.status_code} - {response.text}")
        
        data = response.json()
        
        # Transformer les données en DataFrame
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
        market_caps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
        
        # Convertir les timestamps en dates lisibles
        prices["date"] = pd.to_datetime(prices["timestamp"], unit="ms")
        
        # Fusionner les données
        merged = prices.merge(volumes, on="timestamp").merge(market_caps, on="timestamp")
        merged["symbol"] = symbol.upper()
        merged.set_index("date", inplace=True)
        merged.drop("timestamp", axis=1, inplace=True)
        
        return merged
    
    def _fetch_from_binance(self, symbol, days):
        """Récupère les données depuis Binance."""
        # Format attendu par Binance
        symbol_pair = f"{symbol.upper()}USDT"
        
        # Calculer l'horodatage de début
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        endpoint = "/api/v3/klines"
        params = {
            "symbol": symbol_pair,
            "interval": "1d",
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }
        
        url = f"{self.api_url}{endpoint}"
        response = self.session.get(url, params=params)
        
        if response.status_code != 200:
            raise ConnectionError(f"Erreur API: {response.status_code} - {response.text}")
        
        data = response.json()
        
        # Transformer les données en DataFrame
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        
        # Nettoyer les données
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["symbol"] = symbol.upper()
        df["price"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["market_cap"] = df["price"] * df["volume"]  # Approximation
        
        # Sélectionner et renommer les colonnes
        result = df[["date", "price", "volume", "market_cap", "symbol"]]
        result.set_index("date", inplace=True)
        
        return result