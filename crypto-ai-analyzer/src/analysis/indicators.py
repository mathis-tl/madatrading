"""
Module pour calculer différents indicateurs techniques utilisés dans l'analyse.
"""
import numpy as np
import pandas as pd

def moving_average(series, window_size=7):
    """
    Calcule la moyenne mobile d'une série temporelle.
    
    Args:
        series (pandas.Series): La série temporelle à analyser.
        window_size (int): La taille de la fenêtre pour la moyenne mobile.
        
    Returns:
        pandas.Series: La moyenne mobile calculée.
    """
    return series.rolling(window=window_size).mean()

def exponential_moving_average(series, window_size=7):
    """
    Calcule la moyenne mobile exponentielle d'une série temporelle.
    
    Args:
        series (pandas.Series): La série temporelle à analyser.
        window_size (int): La taille de la fenêtre pour la moyenne mobile.
        
    Returns:
        pandas.Series: La moyenne mobile exponentielle calculée.
    """
    return series.ewm(span=window_size, adjust=False).mean()

def relative_strength_index(series, window_size=14):
    """
    Calcule l'indice de force relative (RSI) d'une série temporelle.
    
    Args:
        series (pandas.Series): La série temporelle à analyser.
        window_size (int): La taille de la fenêtre pour le calcul du RSI.
        
    Returns:
        pandas.Series: Le RSI calculé.
    """
    # Calculer les différences de prix
    delta = series.diff()
    
    # Séparer les gains et les pertes
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculer la moyenne des gains et des pertes
    avg_gain = gain.rolling(window=window_size).mean()
    avg_loss = loss.rolling(window=window_size).mean()
    
    # Calculer le RS (Relative Strength) et le RSI
    rs = avg_gain / avg_loss.where(avg_loss != 0, 1)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def bollinger_bands(series, window_size=20, num_std=2):
    """
    Calcule les bandes de Bollinger d'une série temporelle.
    
    Args:
        series (pandas.Series): La série temporelle à analyser.
        window_size (int): La taille de la fenêtre pour le calcul.
        num_std (int): Le nombre d'écarts-types pour les bandes.
        
    Returns:
        tuple: (middle_band, upper_band, lower_band)
    """
    # Calculer la moyenne mobile (bande du milieu)
    middle_band = moving_average(series, window_size)
    
    # Calculer l'écart-type sur la période
    rolling_std = series.rolling(window=window_size).std()
    
    # Calculer les bandes supérieure et inférieure
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    return middle_band, upper_band, lower_band

def macd(series, fast_period=12, slow_period=26, signal_period=9):
    """
    Calcule l'indicateur MACD (Moving Average Convergence Divergence).
    
    Args:
        series (pandas.Series): La série temporelle à analyser.
        fast_period (int): Période pour la moyenne mobile rapide.
        slow_period (int): Période pour la moyenne mobile lente.
        signal_period (int): Période pour la ligne de signal.
        
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    # Calculer les moyennes mobiles exponentielles
    ema_fast = exponential_moving_average(series, fast_period)
    ema_slow = exponential_moving_average(series, slow_period)
    
    # Calculer la ligne MACD
    macd_line = ema_fast - ema_slow
    
    # Calculer la ligne de signal
    signal_line = exponential_moving_average(macd_line, signal_period)
    
    # Calculer l'histogramme
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def add_all_indicators(df, price_col='price'):
    """
    Ajoute tous les indicateurs techniques au DataFrame.
    
    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données.
        price_col (str): Le nom de la colonne contenant les prix.
        
    Returns:
        pandas.DataFrame: Le DataFrame avec les indicateurs ajoutés.
    """
    result = df.copy()
    
    # Ajouter les moyennes mobiles
    result['ma_7'] = moving_average(df[price_col], 7)
    result['ma_30'] = moving_average(df[price_col], 30)
    result['ma_90'] = moving_average(df[price_col], 90)
    
    # Ajouter les moyennes mobiles exponentielles
    result['ema_7'] = exponential_moving_average(df[price_col], 7)
    result['ema_30'] = exponential_moving_average(df[price_col], 30)
    
    # Ajouter le RSI
    result['rsi'] = relative_strength_index(df[price_col])
    
    # Ajouter les bandes de Bollinger
    mb, ub, lb = bollinger_bands(df[price_col])
    result['bollinger_mid'] = mb
    result['bollinger_upper'] = ub
    result['bollinger_lower'] = lb
    
    # Ajouter le MACD
    macd_line, signal_line, histogram = macd(df[price_col])
    result['macd'] = macd_line
    result['macd_signal'] = signal_line
    result['macd_hist'] = histogram
    
    return result