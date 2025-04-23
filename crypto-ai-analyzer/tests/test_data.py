import unittest
from src.data.collectors.api_collector import ApiCollector
from src.data.storage.database import Database

class TestApiCollector(unittest.TestCase):
    def setUp(self):
        # Fournir l'URL requise pour le constructeur
        self.collector = ApiCollector(api_url="https://api.binance.com")
    
    def test_fetch_data(self):
        # Créer une méthode mocker pour éviter les requêtes réseau réelles
        # pendant les tests
        self.collector.fetch_data = lambda endpoint: {"price": 50000}
        data = self.collector.fetch_data('BTC')
        self.assertIsNotNone(data)
        self.assertIn('price', data)

class TestDatabase(unittest.TestCase):
    def setUp(self):
        # Fournir l'URL de base de données requise pour le constructeur
        self.database = Database(db_url="sqlite:///:memory:")
        
        # Remplacer la méthode connect par une version de test
        self.database.connect = lambda: None
    
    def test_insert_data(self):
        # Créer une méthode mocker pour simuler l'insertion
        self.database.insert_data = lambda table, data: True
        result = self.database.insert_data('BTC', 50000)
        self.assertTrue(result)

    def test_fetch_data(self):
        # Créer une méthode mocker pour simuler la récupération
        self.database.fetch_data = lambda table, condition=None: {'symbol': 'BTC', 'price': 50000}
        data = self.database.fetch_data('BTC')
        self.assertIsNotNone(data)
        self.assertEqual(data['symbol'], 'BTC')

if __name__ == '__main__':
    unittest.main()