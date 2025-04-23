import unittest
import pandas as pd
import numpy as np
from src.analysis.preprocessing import normalize_data, clean_data
from src.analysis.indicators import moving_average, relative_strength_index

class TestAnalysis(unittest.TestCase):

    def test_normalize_data(self):
        # Créer un DataFrame de test
        df = pd.DataFrame({'price': [1, 2, 3, 4, 5]})
        normalized = normalize_data(df)
        self.assertEqual(len(normalized), len(df))
        # Vérifier que les valeurs sont normalisées entre 0 et 1
        self.assertTrue(normalized['price'].max() <= 1.0)
        self.assertTrue(normalized['price'].min() >= 0.0)

    def test_clean_data(self):
        # Créer un DataFrame de test avec des valeurs manquantes
        df = pd.DataFrame({'price': [1, None, 2, np.nan, 3]})
        cleaned = clean_data(df)
        self.assertEqual(len(cleaned), 3)
        self.assertFalse(cleaned['price'].isnull().any())

    def test_moving_average(self):
        # Créer un DataFrame de test
        df = pd.DataFrame({'price': [1, 2, 3, 4, 5]})
        ma = moving_average(df['price'], window_size=3)
        # Vérifier que les 2 premières valeurs sont NaN et le reste est calculé
        self.assertTrue(np.isnan(ma.iloc[0]))
        self.assertTrue(np.isnan(ma.iloc[1])) 
        self.assertEqual(ma.iloc[2], 2.0)
        self.assertEqual(ma.iloc[3], 3.0)
        self.assertEqual(ma.iloc[4], 4.0)

    def test_relative_strength_index(self):
        # Créer un DataFrame de test avec des prix alternants
        df = pd.DataFrame({'price': [10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10]})
        rsi = relative_strength_index(df['price'], window_size=14)
        # Vérifier que RSI est calculé et dans la plage 0-100
        self.assertTrue(0 <= rsi.iloc[-1] <= 100)

if __name__ == '__main__':
    unittest.main()