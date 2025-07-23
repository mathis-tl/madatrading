"""
Module pour collecter les données depuis Yahoo Finance (yfinance).
Alternative fiable pour tester l'application.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class YFinanceCollector:
    """Classe pour collecter des données depuis Yahoo Finance."""
    
    def __init__(self):
        """Initialise le collecteur Yahoo Finance."""
        self.symbol_mapping = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD", 
            "XRP": "XRP-USD",
            "ADA": "ADA-USD",
            "SOL": "SOL-USD"
        }
    
    def fetch_data(self, symbol, days=365):
        """
        Récupère les données historiques pour une crypto-monnaie.
        
        Args:
            symbol (str): Symbole de la crypto-monnaie (ex: BTC, ETH).
            days (int): Nombre de jours d'historique à récupérer.
            
        Returns:
            pandas.DataFrame: DataFrame contenant l'historique des prix.
        """
        try:
            # Convertir le symbole au format Yahoo Finance
            yf_symbol = self.symbol_mapping.get(symbol.upper(), f"{symbol.upper()}-USD")
            
            # Calculer la période
            period = f"{days}d"
            
            # Télécharger les données
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return None
            
            # Reformater les données pour être compatible avec notre système
            result = pd.DataFrame()
            result['price'] = data['Close']
            result['volume'] = data['Volume']
            result['market_cap'] = result['price'] * result['volume']  # Approximation
            result['symbol'] = symbol.upper()
            
            # Renommer l'index
            result.index.name = 'date'
            
            return result
            
        except Exception as e:
            raise ConnectionError(f"Erreur lors du téléchargement depuis Yahoo Finance: {e}")
