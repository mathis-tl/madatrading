"""
Point d'entrée principal de l'application d'analyse de crypto-monnaies.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Ajouter le répertoire parent au PATH pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collectors.api_collector import ApiCollector
from data.collectors.yfinance_collector import YFinanceCollector
from data.storage.database import Database
from analysis.preprocessing import clean_data, normalize_data, prepare_time_series, split_data
from analysis.indicators import add_all_indicators
from models.training import Training
from models.prediction import Prediction
import config

def collect_data(symbols=None, days=365, use_api=True, save_to_db=True):
    """
    Collecte des données historiques pour les crypto-monnaies spécifiées.
    
    Args:
        symbols (list): Liste des symboles des crypto-monnaies.
        days (int): Nombre de jours d'historique à récupérer.
        use_api (bool): Si True, utilise l'API, sinon essaie de récupérer de la DB.
        save_to_db (bool): Si True, sauvegarde les données dans la base de données.
        
    Returns:
        dict: Dictionnaire de DataFrames contenant les données par symbole.
    """
    if symbols is None:
        symbols = config.CRYPTO_SYMBOLS
    
    data_dict = {}
    
    # Initialiser le collecteur alternatif (YFinance) qui est plus fiable
    print("🔧 Utilisation de Yahoo Finance pour les données...")
    yf_collector = YFinanceCollector()
    
    # Initialiser aussi l'API collector comme fallback
    api_url = config.API_ENDPOINTS[config.DEFAULT_API]
    api_collector = ApiCollector(api_url=api_url)
    
    # Initialiser la base de données
    db = Database(db_url=config.DATABASE_URL)
    
    for symbol in symbols:
        print(f"Collecte des données pour {symbol}...")
        
        try:
            if use_api:
                # Essayer d'abord YFinance (plus fiable)
                try:
                    data = yf_collector.fetch_data(symbol, days=days)
                    if data is not None and not data.empty:
                        print(f"  ✓ {len(data)} points de données collectés depuis Yahoo Finance")
                        data_dict[symbol] = data
                        continue
                except Exception as yf_error:
                    print(f"  ⚠️ Yahoo Finance échec, essai avec l'API principale")
                
                # Fallback vers l'API principale
                try:
                    data = api_collector.fetch_data(symbol, days=365)  # Limité à 365 jours
                    if data is not None and not data.empty:
                        print(f"  ✓ {len(data)} points de données collectés depuis l'API")
                        data_dict[symbol] = data
                    else:
                        print(f"  ✗ Aucune donnée reçue de l'API pour {symbol}")
                except Exception as api_error:
                    print(f"  ✗ Erreur API: {api_error}")
            else:
                # Essayer de charger depuis la DB
                try:
                    data = db.get_price_data(symbol, days=days)
                    if data is not None and not data.empty:
                        print(f"  ✓ {len(data)} points de données chargés depuis la DB")
                        data_dict[symbol] = data
                    else:
                        print(f"  ✗ Aucune donnée trouvée en DB pour {symbol}")
                except Exception:
                    print(f"  ✗ Erreur lors de la lecture de la DB pour {symbol}")
                    
        except Exception as e:
            print(f"  ✗ Erreur lors de la collecte pour {symbol}: {e}")
            continue
    
    return data_dict

def preprocess_data(data_dict):
    """
    Prétraite les données collectées en ajoutant des indicateurs techniques
    et en nettoyant les données.
    
    Args:
        data_dict (dict): Dictionnaire de DataFrames de données brutes.
        
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
        
        if len(X) == 0:
            print(f"  ✗ Pas assez de données pour {symbol}, passage au suivant")
            continue
        
        # Étape 6: Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)
        print(f"  ✓ Données divisées: {len(X_train)} échantillons d'entraînement, {len(X_test)} échantillons de test")
        
        # Étape 7: Créer et entraîner le modèle
        trainer = Training(model=None, data=(X_train, y_train))
        model = trainer.create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Configurer l'entraînement
        batch_size = 32
        
        # Ajouter early stopping pour éviter le surajustement
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

def generate_predictions(models, data_dict):
    """
    Génère des prédictions pour chaque modèle.
    
    Args:
        models (dict): Dictionnaire des modèles entraînés
        data_dict (dict): Dictionnaire des données
        
    Returns:
        dict: Prédictions pour chaque symbole
    """
    predictions = {}
    
    for symbol, model_data in models.items():
        print(f"Génération des prédictions pour {symbol}...")
        
        model = model_data['model']
        X_test = model_data['data']['X_test']
        y_test = model_data['data']['y_test']
        
        # Créer un objet de prédiction
        predictor = Prediction(model=model)
        
        # Générer les prédictions
        y_pred = predictor.predict(X_test)
        
        predictions[symbol] = {
            'predictions': y_pred,
            'actual': y_test,
            'dates': model_data['data']['dates']
        }
        
        print(f"  ✓ Prédictions générées pour {symbol}")
    
    return predictions

def visualize_results(models, data_dict, predictions):
    """
    Visualise les résultats des prédictions.
    
    Args:
        models (dict): Dictionnaire des modèles entraînés
        data_dict (dict): Dictionnaire des données
        predictions (dict): Dictionnaire des prédictions
    """
    # Créer le dossier outputs s'il n'existe pas
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    for symbol in predictions.keys():
        print(f"Visualisation des résultats pour {symbol}...")
        
        model_data = models[symbol]
        model = model_data['model']
        X_test = model_data['data']['X_test']
        y_test = model_data['data']['y_test']
        dates = model_data['data']['dates']
        
        # Créer un objet de prédiction
        predictor = Prediction(model=model)
        
        # Évaluer le modèle
        metrics = predictor.evaluate(X_test, y_test)
        print(f"  ✓ Métriques pour {symbol}:")
        print(f"    - RMSE: {metrics['rmse']:.2f}")
        print(f"    - MAE: {metrics['mae']:.2f}")
        print(f"    - MAPE: {metrics['mape']:.2f}%")
        print(f"    - R²: {metrics['r2']:.4f}")
        
        # Tracer les prédictions
        plt_obj = predictor.plot_predictions(X_test, y_test, dates, 
                                           title=f"Prédictions vs Réalité pour {symbol}")
        
        # Sauvegarder le graphique
        output_path = os.path.join(outputs_dir, f"{symbol}_predictions.png")
        plt_obj.savefig(output_path)
        print(f"  ✓ Graphique sauvegardé : {output_path}")
        plt_obj.close()

def main():
    """Fonction principale pour lancer l'analyse complète"""
    print("🚀 Lancement du Crypto AI Analyzer")
    print("=" * 50)
    
    try:
        # 1. Collecte des données
        print("📊 Collecte des données...")
        data_dict = collect_data()
        
        if not data_dict:
            print("❌ Aucune donnée collectée. Arrêt du programme.")
            return 1
        
        # 2. Prétraitement des données
        print("🔧 Prétraitement des données...")
        processed_data = preprocess_data(data_dict)
        
        if not processed_data:
            print("❌ Aucune donnée prétraitée. Arrêt du programme.")
            return 1
        
        # 3. Entraînement des modèles
        print("🤖 Entraînement des modèles...")
        models = train_models(processed_data)
        
        if not models:
            print("❌ Aucun modèle entraîné. Arrêt du programme.")
            return 1
        
        # 4. Génération des prédictions
        print("🔮 Génération des prédictions...")
        predictions = generate_predictions(models, processed_data)
        
        # 5. Visualisation des résultats
        print("📈 Génération des graphiques...")
        visualize_results(models, processed_data, predictions)
        
        print("✅ Analyse terminée avec succès !")
        print("📁 Vérifiez le dossier 'outputs/' pour les résultats")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
