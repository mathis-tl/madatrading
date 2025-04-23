"""
Module pour le prétraitement des données avant analyse et modélisation.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean_data(df):
    """
    Nettoie les données en supprimant les valeurs manquantes et dupliquées.
    
    Args:
        df (pandas.DataFrame): DataFrame à nettoyer.
        
    Returns:
        pandas.DataFrame: DataFrame nettoyé.
    """
    # Supprimer les valeurs manquantes
    df_clean = df.dropna()
    
    # Supprimer les doublons
    df_clean = df_clean.drop_duplicates()
    
    # Réindexer si nécessaire
    if df_clean.index.name == 'date':
        df_clean = df_clean.sort_index()
        
    return df_clean

def normalize_data(df, columns=None):
    """
    Normalise les données entre 0 et 1.
    
    Args:
        df (pandas.DataFrame): DataFrame à normaliser.
        columns (list, optional): Liste des colonnes à normaliser.
            Par défaut, normalise toutes les colonnes numériques.
            
    Returns:
        pandas.DataFrame: DataFrame normalisé.
    """
    df_norm = df.copy()
    
    if columns is None:
        # Sélectionner uniquement les colonnes numériques
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler = MinMaxScaler()
    df_norm[columns] = scaler.fit_transform(df[columns])
    
    return df_norm

def prepare_time_series(df, target_col, window_size=60, forecast_horizon=1):
    """
    Prépare les données de séries temporelles pour l'entraînement de modèles.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données.
        target_col (str): Nom de la colonne cible à prédire.
        window_size (int): Taille de la fenêtre pour les features (nombre de jours d'historique).
        forecast_horizon (int): Horizon de prévision (nombre de jours à prédire).
        
    Returns:
        tuple: (X, y) où X est un numpy array 3D de shape (n_samples, window_size, n_features)
               et y est un numpy array de shape (n_samples, forecast_horizon).
    """
    # S'assurer que l'index est bien trié
    df = df.sort_index()
    
    # Extraire la colonne cible
    target = df[target_col].values
    
    # Préparer les features et targets
    X, y = [], []
    
    for i in range(len(df) - window_size - forecast_horizon + 1):
        # Extraire la fenêtre de données pour les features
        features = df.iloc[i:i+window_size].values
        
        # Extraire la fenêtre de données pour la cible
        target_window = target[i+window_size:i+window_size+forecast_horizon]
        
        X.append(features)
        y.append(target_window)
    
    return np.array(X), np.array(y)

def split_data(X, y, train_ratio=0.8, shuffle=False):
    """
    Divise les données en ensembles d'entraînement et de test.
    
    Args:
        X (numpy.ndarray): Features.
        y (numpy.ndarray): Cibles.
        train_ratio (float): Ratio de données pour l'entraînement.
        shuffle (bool): Si True, mélange les données avant de les diviser.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Nombre d'échantillons
    n_samples = len(X)
    
    if shuffle:
        # Créer des indices aléatoires
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
    
    # Calculer l'indice de séparation
    split_idx = int(n_samples * train_ratio)
    
    # Diviser les données
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test