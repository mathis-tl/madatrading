"""
Point d'entrée principal de l'application d'analyse de crypto-monnaies.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.data.collectors.api_collector import ApiCollector
from src.data.storage.database import Database
from src.analysis.preprocessing import clean_data, normalize_data, prepare_time_series, split_data
from src.analysis.indicators import add_all_indicators
from src.models.training import Training
from src.models.prediction import Prediction
import src.config as config
from src.models.hyperparameter_tuning import grid_search, data_augmentation

def collect_data(symbols=None, days=730, use_api=True, save_to_db=True):
    """
    Collecte des données historiques pour les crypto-monnaies spécifiées.
    
    Args:
        symbols (list): Liste des symboles des crypto-monnaies.
        days (int): Nombre de jours d'historique à récupérer (augmenté à 2 ans).
        use_api (bool): Si True, utilise l'API, sinon essaie de récupérer de la DB.
        save_to_db (bool): Si True, sauvegarde les données dans la base de données.
        
    Returns:
        dict: Dictionnaire de DataFrames contenant les données par symbole.
    """
    if symbols is None:
        symbols = config.CRYPTO_SYMBOLS
    
    data_dict = {}
    
    # Initialiser le collecteur d'API
    api_url = config.API_ENDPOINTS[config.DEFAULT_API]
    api_collector = ApiCollector(api_url=api_url)
    
    # Initialiser la base de données
    db = Database(db_url=config.DATABASE_URL)
    
    for symbol in symbols:
        print(f"Collecte des données pour {symbol}...")
        
        if use_api:
            try:
                # Récupérer les données depuis l'API
                df = api_collector.fetch_data(symbol, days=days)
                print(f"  ✓ Données récupérées de l'API: {len(df)} entrées")
                
                # Sauvegarder dans la base de données si nécessaire
                if save_to_db:
                    db.insert_data(f"{symbol.lower()}_data", df)
                    print(f"  ✓ Données sauvegardées dans la base de données")
                
                data_dict[symbol] = df
                
            except Exception as e:
                print(f"  ✗ Erreur lors de la récupération depuis l'API: {e}")
                
                # Essayer de récupérer depuis la base de données
                print(f"  ⟳ Tentative de récupération depuis la base de données...")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                df = db.fetch_data(symbol, start_date, end_date)
                
                if not df.empty:
                    print(f"  ✓ Données récupérées de la base de données: {len(df)} entrées")
                    data_dict[symbol] = df
                else:
                    print(f"  ✗ Aucune donnée disponible pour {symbol}")
        else:
            # Récupérer directement depuis la base de données
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = db.fetch_data(symbol, start_date, end_date)
            
            if not df.empty:
                print(f"  ✓ Données récupérées de la base de données: {len(df)} entrées")
                data_dict[symbol] = df
            else:
                print(f"  ✗ Aucune donnée disponible pour {symbol} dans la base de données")
    
    return data_dict

def preprocess_data(data_dict):
    """
    Prétraite les données pour l'analyse.
    
    Args:
        data_dict (dict): Dictionnaire de DataFrames contenant les données par symbole.
        
    Returns:
        dict: Dictionnaire de DataFrames prétraités.
    """
    processed_data = {}
    
    for symbol, df in data_dict.items():
        print(f"Prétraitement des données pour {symbol}...")
        
        # Nettoyer les données
        df_clean = clean_data(df)
        print(f"  ✓ Données nettoyées: {len(df_clean)} entrées")
        
        # Ajouter les indicateurs techniques
        df_indicators = add_all_indicators(df_clean)
        print(f"  ✓ Indicateurs techniques ajoutés")
        
        # Supprimer les lignes avec des NaN (les premiers jours n'ont pas tous les indicateurs)
        df_indicators = df_indicators.dropna()
        print(f"  ✓ Lignes avec valeurs manquantes supprimées: {len(df_indicators)} entrées")
        
        processed_data[symbol] = df_indicators
    
    return processed_data

def train_models(data_dict, window_size=60, forecast_horizon=1):
    """
    Entraîne des modèles de prédiction pour chaque symbole.
    """
    models = {}
    
    for symbol, df in data_dict.items():
        print(f"Entraînement du modèle pour {symbol}...")
        
        # Étape 1: S'assurer que toutes les colonnes sont numériques
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        df_numeric = df[numeric_columns].copy()
        
        # Étape 2: Vérifier qu'il ne reste pas de NaN
        df_numeric = df_numeric.fillna(0)
        print(f"  ✓ Données nettoyées: {len(df_numeric)} entrées avec {len(numeric_columns)} caractéristiques numériques")
        
        # Étape 3: Normaliser les données
        df_norm = normalize_data(df_numeric)
        print(f"  ✓ Données normalisées")
        
        # Étape 4: Préparer les données pour l'entraînement
        X, y = prepare_time_series(df_norm, 'price', window_size, forecast_horizon)
        
        # Étape 5: Convertir explicitement en float32
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        print(f"  ✓ Données de séries temporelles préparées: {len(X)} échantillons")
        
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)
        print(f"  ✓ Données divisées: {len(X_train)} échantillons d'entraînement, {len(X_test)} échantillons de test")
        
        # Créer et entraîner le modèle
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_shape = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        # Créer un modèle
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(output_shape))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Entraîner le modèle
        trainer = Training(model=model, data=(X_train, y_train))
        history = trainer.train(epochs=50, batch_size=32, validation_split=0.2)
        print(f"  ✓ Modèle entraîné")
        
        # Sauvegarder le modèle et les données
        models[symbol] = {
            'model': trainer.get_model(),
            'data': {
                'X_test': X_test,
                'y_test': y_test,
                'dates': df.index[-len(X_test):]
            }
        }
    
    return models

def evaluate_models(models):
    """
    Évalue les performances des modèles.
    
    Args:
        models (dict): Dictionnaire contenant les modèles et les données de test.
    """
    for symbol, model_data in models.items():
        print(f"Évaluation du modèle pour {symbol}...")
        
        model = model_data['model']
        X_test = model_data['data']['X_test']
        y_test = model_data['data']['y_test']
        dates = model_data['data']['dates']
        
        # Créer un objet de prédiction
        predictor = Prediction(model=model)
        
        # Évaluer le modèle
        metrics = predictor.evaluate(X_test, y_test)
        print(f"  ✓ Métriques:")
        print(f"    - RMSE: {metrics['rmse']:.2f}")
        print(f"    - MAE: {metrics['mae']:.2f}")
        print(f"    - MAPE: {metrics['mape']:.2f}%")
        print(f"    - R²: {metrics['r2']:.4f}")
        
        # Tracer les prédictions
        plt_obj = predictor.plot_predictions(X_test, y_test, dates, 
                                           title=f"Prédictions vs Réalité pour {symbol}")
        
        # Sauvegarder le graphique
        os.makedirs("outputs", exist_ok=True)
        plt_obj.savefig(f"outputs/{symbol}_predictions.png")
        print(f"  ✓ Graphique sauvegardé dans outputs/{symbol}_predictions.png")
        plt_obj.close()

def main():
    """
    Fonction principale de l'application.
    """
    print("=== Démarrage de l'analyse des crypto-monnaies ===")
    
    # Collecter les données (augmenté à 2 ans d'historique)
    data_dict = collect_data(days=730)
    
    # Prétraiter les données
    processed_data = preprocess_data(data_dict)
    
    # Sélectionner un symbole pour l'optimisation des hyperparamètres (pour gagner du temps)
    symbol_for_tuning = "BTC"
    df = processed_data[symbol_for_tuning]
    
    # Nettoyer et normaliser
    df_numeric = df.select_dtypes(include=['number']).copy()
    df_numeric = df_numeric.fillna(0)
    df_norm = normalize_data(df_numeric)
    
    # Préparer les données pour différentes fenêtres
    window_size = 30  # Pour la préparation initiale
    X, y = prepare_time_series(df_norm, 'price', window_size, 1)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Augmenter les données
    X_augmented, y_augmented = data_augmentation(X, y)
    
    # Diviser les données
    split_idx = int(len(X_augmented) * 0.8)
    X_train, X_val = X_augmented[:split_idx], X_augmented[split_idx:]
    y_train, y_val = y_augmented[:split_idx], y_augmented[split_idx:]
    
    print("Optimisation des hyperparamètres...")
    tuning_results = grid_search(X_train, y_train, X_val, y_val)
    
    print("\nMeilleurs hyperparamètres trouvés:")
    for param, value in tuning_results['best_params'].items():
        print(f"  - {param}: {value}")
    print(f"  - RMSE: {tuning_results['best_score']:.4f}")
    
    # Sauvegarder l'analyse des hyperparamètres
    tuning_results['results_df'].to_csv('outputs/hyperparameter_results.csv')
    
    print("\nEntraînement des modèles avec les meilleurs hyperparamètres...")
    # Récupérer les meilleurs paramètres
    best_window = tuning_results['best_params']['window_size']
    best_lstm_units = tuning_results['best_params']['lstm_units']
    best_dropout = tuning_results['best_params']['dropout'] 
    best_lr = tuning_results['best_params']['learning_rate']
    best_batch = tuning_results['best_params']['batch_size']
    
    # Entraîner les modèles
    models = train_models_with_params(processed_data, best_window, best_lstm_units, 
                                      best_dropout, best_lr, best_batch)
    
    # Évaluer les modèles
    evaluate_models(models)
    
    print("=== Analyse terminée ===")

def train_models_with_params(data_dict, window_size, lstm_units, dropout, learning_rate, batch_size):
    """
    Entraîne des modèles avec les hyperparamètres optimisés.
    """
    models = {}
    
    for symbol, df in data_dict.items():
        print(f"Entraînement du modèle pour {symbol} avec paramètres optimisés...")
        
        # Étape 1: S'assurer que toutes les colonnes sont numériques
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        df_numeric = df[numeric_columns].copy()
        df_numeric = df_numeric.fillna(0)
        
        # Étape 2: Normaliser les données
        df_norm = normalize_data(df_numeric)
        print(f"  ✓ Données normalisées")
        
        # Étape 3: Préparer les données pour l'entraînement
        X, y = prepare_time_series(df_norm, 'price', window_size, 1)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        print(f"  ✓ Données de séries temporelles préparées: {len(X)} échantillons")
        
        # Étape 4: Augmenter les données
        X_augmented, y_augmented = data_augmentation(X, y)
        
        # Étape 5: Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = split_data(X_augmented, y_augmented, train_ratio=0.8)
        print(f"  ✓ Données divisées: {len(X_train)} échantillons d'entraînement, {len(X_test)} échantillons de test")
        
        # Étape 6: Créer un modèle avec les hyperparamètres optimisés
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential()
        
        # Première couche LSTM
        model.add(LSTM(lstm_units[0], 
                      input_shape=(X_train.shape[1], X_train.shape[2]), 
                      return_sequences=len(lstm_units) > 1))
        model.add(Dropout(dropout))
        
        # Couches LSTM intermédiaires
        for i in range(1, len(lstm_units)-1):
            model.add(LSTM(lstm_units[i], return_sequences=True))
            model.add(Dropout(dropout))
        
        # Dernière couche LSTM
        if len(lstm_units) > 1:
            model.add(LSTM(lstm_units[-1], return_sequences=False))
            model.add(Dropout(dropout))
        
        # Couche de sortie
        model.add(Dense(1))
        
        # Compiler le modèle
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # Entraîner le modèle
        from tensorflow.keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=100,  # Plus d'époques, mais avec early stopping
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        print(f"  ✓ Modèle entraîné")
        
        # Sauvegarder le modèle et les données
        models[symbol] = {
            'model': model,
            'data': {
                'X_test': X_test,
                'y_test': y_test,
                'dates': df.index[-len(X_test):]
            }
        }
    
    return models

if __name__ == "__main__":
    main()