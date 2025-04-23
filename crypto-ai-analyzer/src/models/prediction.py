"""
Module pour effectuer des prédictions avec les modèles entraînés.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Prediction:
    """Classe pour effectuer des prédictions avec les modèles entraînés."""
    
    def __init__(self, model):
        """
        Initialise le module de prédiction.
        
        Args:
            model: Le modèle entraîné à utiliser pour les prédictions.
        """
        self.model = model
        
    def predict(self, X_test):
        """
        Effectue des prédictions sur les données de test.
        
        Args:
            X_test: Les données de test pour lesquelles faire des prédictions.
            
        Returns:
            numpy.ndarray: Les prédictions du modèle.
        """
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        Évalue les performances du modèle.
        
        Args:
            X_test: Les données de test.
            y_test: Les valeurs réelles.
            
        Returns:
            dict: Dictionnaire contenant les métriques d'évaluation.
        """
        # Faire des prédictions
        y_pred = self.predict(X_test)
        
        # Si les dimensions ne correspondent pas, ajuster
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        if len(y_test.shape) > 1 and y_test.shape[1] == 1:
            y_test = y_test.flatten()
        
        # Calculer les métriques
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculer le MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def plot_predictions(self, X_test, y_test, dates=None, title="Prédictions vs Réalité"):
        """
        Trace un graphique des prédictions vs les valeurs réelles.
        
        Args:
            X_test: Les données de test.
            y_test: Les valeurs réelles.
            dates (list, optional): Les dates correspondant aux prédictions.
            title (str): Titre du graphique.
        """
        # Faire des prédictions
        y_pred = self.predict(X_test)
        
        # Si les dimensions ne correspondent pas, ajuster
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        if len(y_test.shape) > 1 and y_test.shape[1] == 1:
            y_test = y_test.flatten()
        
        # Créer la figure
        plt.figure(figsize=(12, 6))
        
        if dates is not None:
            plt.plot(dates, y_test, label='Réalité')
            plt.plot(dates, y_pred, label='Prédictions', linestyle='--')
            plt.xticks(rotation=45)
        else:
            plt.plot(y_test, label='Réalité')
            plt.plot(y_pred, label='Prédictions', linestyle='--')
        
        # Ajouter les métriques d'évaluation
        metrics = self.evaluate(X_test, y_test)
        metrics_text = (f"RMSE: {metrics['rmse']:.2f}\n"
                       f"MAE: {metrics['mae']:.2f}\n"
                       f"MAPE: {metrics['mape']:.2f}%\n"
                       f"R²: {metrics['r2']:.4f}")
        
        # Ajouter le texte des métriques
        plt.text(0.02, 0.95, metrics_text, transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title(title)
        plt.xlabel('Temps')
        plt.ylabel('Valeur')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt