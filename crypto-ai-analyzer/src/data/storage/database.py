"""
Module pour gérer le stockage et la récupération des données dans une base de données.
"""
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class CryptoData(Base):
    """Modèle SQLAlchemy pour les données de crypto-monnaies."""
    __tablename__ = 'crypto_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    date = Column(DateTime, nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<CryptoData(symbol='{self.symbol}', date='{self.date}', price={self.price})>"

class Database:
    """Classe pour gérer les opérations de base de données."""
    
    def __init__(self, db_url):
        """
        Initialise la connexion à la base de données.
        
        Args:
            db_url (str): URL de connexion à la base de données.
        """
        self.db_url = db_url
        self.engine = None
        self.session = None
        self.metadata = MetaData()
        
    def connect(self):
        """Établit la connexion à la base de données."""
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
    def insert_data(self, table, data):
        """
        Insère des données dans la base de données.
        
        Args:
            table (str): Nom de la table.
            data (dict ou DataFrame): Données à insérer.
            
        Returns:
            bool: True si l'insertion a réussi, False sinon.
        """
        if self.session is None:
            self.connect()
            
        try:
            if isinstance(data, pd.DataFrame):
                # Convertir DataFrame en liste de dictionnaires
                records = data.reset_index().to_dict(orient='records')
                for record in records:
                    crypto_data = CryptoData(
                        symbol=record['symbol'],
                        date=record['date'],
                        price=record['price'],
                        volume=record.get('volume', 0),
                        market_cap=record.get('market_cap', 0)
                    )
                    self.session.add(crypto_data)
            else:
                # Si c'est un dictionnaire simple
                crypto_data = CryptoData(
                    symbol=data['symbol'],
                    date=data.get('date', datetime.now()),
                    price=data['price'],
                    volume=data.get('volume', 0),
                    market_cap=data.get('market_cap', 0)
                )
                self.session.add(crypto_data)
                
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            print(f"Erreur lors de l'insertion des données: {e}")
            return False
            
    def fetch_data(self, symbol, start_date=None, end_date=None):
        """
        Récupère des données de la base de données.
        
        Args:
            symbol (str): Symbole de la crypto-monnaie.
            start_date (datetime, optional): Date de début.
            end_date (datetime, optional): Date de fin.
            
        Returns:
            pandas.DataFrame: Les données récupérées.
        """
        if self.session is None:
            self.connect()
            
        query = self.session.query(CryptoData).filter(CryptoData.symbol == symbol)
        
        if start_date:
            query = query.filter(CryptoData.date >= start_date)
        if end_date:
            query = query.filter(CryptoData.date <= end_date)
            
        # Exécuter la requête et convertir en DataFrame
        result = query.all()
        data = [{
            'symbol': r.symbol,
            'date': r.date,
            'price': r.price,
            'volume': r.volume,
            'market_cap': r.market_cap
        } for r in result]
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('date', inplace=True)
            
        return df
    
    def close(self):
        """Ferme la connexion à la base de données."""
        if self.session:
            self.session.close()
            self.session = None