"""
Module pour l'entraînement des modèles de prédiction.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

class Training:
    """Classe pour l'entraînement des modèles de prédiction."""
    
    def __init__(self, model, data):
        """
        Initialise le module d'entraînement.
        
        Args:
            model: Le modèle à entraîner ou None pour en créer un nouveau.
            data: Les données d'entraînement (X_train, y_train).
        """
        self.model = model
        self.X_train, self.y_train = data
        
    def create_lstm_model(self, input_shape, output_shape=1, neurons=[64, 32]):
        """
        Crée un modèle LSTM pour la prédiction de séries temporelles.
        
        Args:
            input_shape (tuple): Forme des données d'entrée (window_size, n_features).
            output_shape (int): Nombre de valeurs à prédire.
            neurons (list): Liste des nombres de neurones pour chaque couche LSTM.
            
        Returns:
            tensorflow.keras.Model: Le modèle LSTM créé.
        """
        model = Sequential()
        
        # Première couche LSTM avec retour de séquences
        model.add(LSTM(neurons[0], 
                      input_shape=input_shape, 
                      return_sequences=len(neurons) > 1))
        model.add(Dropout(0.2))
        
        # Couches LSTM intermédiaires
        for i in range(1, len(neurons)-1):
            model.add(LSTM(neurons[i], return_sequences=True))
            model.add(Dropout(0.2))
        
        # Dernière couche LSTM sans retour de séquences
        if len(neurons) > 1:
            model.add(LSTM(neurons[-1], return_sequences=False))
            model.add(Dropout(0.2))
        
        # Couche de sortie
        model.add(Dense(output_shape))
        
        # Compiler le modèle
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def train(self, epochs=50, batch_size=32, validation_split=0.2, patience=10, 
              save_path=None):
        """
        Entraîne le modèle.
        
        Args:
            epochs (int): Nombre d'époques d'entraînement.
            batch_size (int): Taille des lots pour l'entraînement.
            validation_split (float): Portion des données à utiliser pour validation.
            patience (int): Nombre d'époques sans amélioration avant arrêt précoce.
            save_path (str): Chemin où sauvegarder le meilleur modèle.
            
        Returns:
            history: Historique d'entraînement.
        """
        # Créer le modèle s'il n'existe pas
        if self.model is None:
            input_shape = (self.X_train.shape[1], self.X_train.shape[2])
            output_shape = self.y_train.shape[1] if len(self.y_train.shape) > 1 else 1
            self.model = self.create_lstm_model(input_shape, output_shape)
        
        # Configurer les callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        ]
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            callbacks.append(ModelCheckpoint(save_path, save_best_only=True))
        
        # Entraîner le modèle
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def get_model(self):
        """
        Récupère le modèle entraîné.
        
        Returns:
            tensorflow.keras.Model: Le modèle entraîné.
        """
        return self.model