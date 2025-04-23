"""
Module pour l'optimisation des hyperparamètres des modèles.
"""
import numpy as np
import pandas as pd
from itertools import product
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def grid_search(X_train, y_train, X_val, y_val):
    """
    Effectue une recherche par grille pour trouver les meilleurs hyperparamètres.
    
    Args:
        X_train: Données d'entraînement.
        y_train: Cibles d'entraînement.
        X_val: Données de validation.
        y_val: Cibles de validation.
        
    Returns:
        dict: Meilleurs hyperparamètres et résultats.
    """
    # Définir les valeurs des hyperparamètres à tester
    window_sizes = [15, 30, 45]  # Fenêtres plus petites pour capter les tendances récentes
    lstm_units = [[64, 32], [128, 64], [64, 32, 16]]
    dropout_rates = [0.2, 0.3, 0.4]
    learning_rates = [0.001, 0.0005, 0.0001]
    batch_sizes = [16, 32, 64]
    
    best_score = float('inf')
    best_params = {}
    results = []
    
    # Créer toutes les combinaisons possibles d'hyperparamètres
    param_combinations = list(product(window_sizes, lstm_units, dropout_rates, learning_rates, batch_sizes))
    total_combinations = len(param_combinations)
    
    print(f"Début de la recherche par grille avec {total_combinations} combinaisons...")
    
    for i, (window_size, units, dropout, lr, batch_size) in enumerate(param_combinations):
        print(f"Essai {i+1}/{total_combinations} - window: {window_size}, units: {units}, dropout: {dropout}, lr: {lr}, batch: {batch_size}")
        
        # Prétraiter les données pour cette taille de fenêtre (à implémenter selon votre logique)
        # Pour ce cas simplifié, nous utilisons les données telles quelles
        
        # Créer et compiler le modèle
        model = Sequential()
        
        # Première couche LSTM
        model.add(LSTM(units[0], 
                      input_shape=(X_train.shape[1], X_train.shape[2]), 
                      return_sequences=len(units) > 1))
        model.add(Dropout(dropout))
        
        # Couches LSTM intermédiaires
        for i in range(1, len(units)-1):
            model.add(LSTM(units[i], return_sequences=True))
            model.add(Dropout(dropout))
        
        # Dernière couche LSTM
        if len(units) > 1:
            model.add(LSTM(units[-1], return_sequences=False))
            model.add(Dropout(dropout))
        
        # Couche de sortie
        model.add(Dense(1))
        
        # Compiler le modèle
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Entraîner le modèle
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Évaluer le modèle
        val_loss = min(history.history['val_loss'])
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        # Enregistrer les résultats
        result = {
            'window_size': window_size,
            'lstm_units': units,
            'dropout': dropout,
            'learning_rate': lr,
            'batch_size': batch_size,
            'val_loss': val_loss,
            'rmse': rmse,
            'epochs': len(history.history['loss'])
        }
        results.append(result)
        
        # Mettre à jour les meilleurs paramètres
        if rmse < best_score:
            best_score = rmse
            best_params = {
                'window_size': window_size,
                'lstm_units': units,
                'dropout': dropout,
                'learning_rate': lr,
                'batch_size': batch_size
            }
            best_model = model
            
        print(f"  RMSE: {rmse:.4f} - Val Loss: {val_loss:.4f}")
    
    # Transformer les résultats en DataFrame pour analyse
    results_df = pd.DataFrame(results)
    
    # Tracer les résultats
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    results_df.groupby('window_size')['rmse'].mean().plot(kind='bar')
    plt.title('RMSE moyen par taille de fenêtre')
    
    plt.subplot(2, 2, 2)
    results_df.groupby('dropout')['rmse'].mean().plot(kind='bar')
    plt.title('RMSE moyen par taux de dropout')
    
    plt.subplot(2, 2, 3)
    results_df.groupby('learning_rate')['rmse'].mean().plot(kind='bar')
    plt.title('RMSE moyen par taux d\'apprentissage')
    
    plt.subplot(2, 2, 4)
    results_df.groupby('batch_size')['rmse'].mean().plot(kind='bar')
    plt.title('RMSE moyen par taille de lot')
    
    plt.tight_layout()
    plt.savefig('outputs/hyperparameter_analysis.png')
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_model': best_model,
        'results_df': results_df
    }

def data_augmentation(X, y, augmentation_factor=0.5, noise_level=0.02):
    """
    Augmente le jeu de données en ajoutant des variantes avec du bruit.
    
    Args:
        X: Features.
        y: Cibles.
        augmentation_factor: Proportion de données supplémentaires (0.5 = +50%).
        noise_level: Niveau de bruit (0.02 = ±2%).
        
    Returns:
        tuple: (X_augmented, y_augmented)
    """
    n_samples = int(X.shape[0] * augmentation_factor)
    
    # Sélectionner des échantillons aléatoires
    indices = np.random.choice(X.shape[0], n_samples, replace=True)
    X_aug = X[indices].copy()
    y_aug = y[indices].copy()
    
    # Ajouter du bruit
    X_noise = np.random.normal(0, noise_level, X_aug.shape)
    X_aug = X_aug + X_noise
    
    # Ajouter du bruit aux cibles dans une moindre mesure
    y_noise = np.random.normal(0, noise_level/2, y_aug.shape)
    y_aug = y_aug + y_noise
    
    # Concaténer avec les données originales
    X_augmented = np.vstack([X, X_aug])
    y_augmented = np.vstack([y.reshape(-1, 1), y_aug.reshape(-1, 1)]).reshape(-1)
    
    print(f"Données augmentées: {X.shape[0]} → {X_augmented.shape[0]} échantillons")
    
    return X_augmented, y_augmented