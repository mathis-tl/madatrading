#!/usr/bin/env python3
"""
Script de vérification des imports et de la structure du projet
pour s'assurer de la portabilité
"""

import os
import sys
import importlib
from pathlib import Path

def test_basic_imports():
    """Test des imports de base Python"""
    print(" Test des imports de base...")
    basic_modules = [
        ('os', 'Système d\'exploitation'),
        ('sys', 'Système Python'),
        ('pathlib', 'Manipulation de chemins'),
        ('json', 'Manipulation JSON'),
        ('datetime', 'Gestion des dates'),
        ('sqlite3', 'Base de données SQLite')
    ]
    
    success = 0
    for module, desc in basic_modules:
        try:
            importlib.import_module(module)
            print(f"   {module} - {desc}")
            success += 1
        except ImportError as e:
            print(f"   {module} - Erreur: {e}")
    
    return success == len(basic_modules)

def test_third_party_imports():
    """Test des imports de bibliothèques tierces"""
    print("\n Test des bibliothèques tierces...")
    third_party = [
        ('numpy', 'Calculs numériques'),
        ('pandas', 'Manipulation de données'),
        ('matplotlib', 'Graphiques'),
        ('sklearn', 'Machine Learning'),
        ('requests', 'Requêtes HTTP'),
        ('yfinance', 'Données financières')
    ]
    
    success = 0
    optional_success = 0
    optional_modules = [('tensorflow', 'Deep Learning (optionnel)')]
    
    for module, desc in third_party:
        try:
            importlib.import_module(module)
            print(f"   {module} - {desc}")
            success += 1
        except ImportError as e:
            print(f"   {module} - Erreur: {e}")
    
    # Test des modules optionnels
    for module, desc in optional_modules:
        try:
            importlib.import_module(module)
            print(f"   {module} - {desc}")
            optional_success += 1
        except ImportError as e:
            print(f"  ⚠ {module} - {desc} - Pas installé (normal)")
    
    return success >= len(third_party) - 1  # On accepte 1 échec

def test_project_structure():
    """Test de la structure du projet"""
    print("\n📁 Test de la structure du projet...")
    
    required_files = [
        'launcher.py',
        'requirements.txt',
        'README.md',
        'install.py',
        '.env.example',
        'IA_madatrading.py'
    ]
    
    required_dirs = [
        'crypto-ai-analyzer',
        'crypto-ai-analyzer/src',
        'outputs'
    ]
    
    success = 0
    
    print("  Fichiers requis:")
    for file in required_files:
        if os.path.exists(file):
            print(f"    {file}")
            success += 1
        else:
            print(f"     {file} - Manquant")
    
    print("  Dossiers requis:")
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"     {directory}")
            success += 1
        else:
            print(f"     {directory} - Manquant")
    
    return success == len(required_files) + len(required_dirs)

def test_crypto_ai_imports():
    """Test des imports du module crypto-ai-analyzer"""
    print("\n Test des imports crypto-ai-analyzer...")
    
    # Ajouter le chemin vers le module
    crypto_path = os.path.join(os.getcwd(), 'crypto-ai-analyzer', 'src')
    if crypto_path not in sys.path:
        sys.path.insert(0, crypto_path)
    
    test_imports = [
        ('config', 'Configuration'),
        ('data.collectors.api_collector', 'Collecteur API'),
        ('data.storage.database', 'Base de données'),
        ('analysis.preprocessing', 'Prétraitement'),
        ('analysis.indicators', 'Indicateurs techniques'),
        ('models.training', 'Entraînement'),
        ('models.prediction', 'Prédiction')
    ]
    
    success = 0
    for module, desc in test_imports:
        try:
            importlib.import_module(module)
            print(f"   {module} - {desc}")
            success += 1
        except ImportError as e:
            print(f"   {module} - {desc} - Erreur: {e}")
    
    return success >= len(test_imports) - 2  # On accepte 2 échecs

def test_launchers():
    """Test que les scripts de lancement existent et sont exécutables"""
    print("\n Test des scripts de lancement...")
    
    launchers = [
        ('launcher.py', 'Menu principal'),
        ('IA_madatrading.py', 'Script simple'),
        ('crypto-ai-analyzer/src/main.py', 'Module IA avancé')
    ]
    
    success = 0
    for script, desc in launchers:
        if os.path.exists(script):
            # Vérifier que le fichier n'est pas vide
            with open(script, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content and ('def ' in content or 'import ' in content):
                    print(f"   {script} - {desc}")
                    success += 1
                else:
                    print(f"  ⚠ {script} - {desc} - Fichier vide ou invalide")
        else:
            print(f"   {script} - {desc} - Manquant")
    
    return success == len(launchers)

def generate_summary_report():
    """Génère un rapport de résumé"""
    print("\n📋 RAPPORT DE PORTABILITÉ")
    print("=" * 50)
    
    tests = [
        ("Imports de base Python", test_basic_imports),
        ("Bibliothèques tierces", test_third_party_imports),
        ("Structure du projet", test_project_structure),
        ("Imports crypto-ai-analyzer", test_crypto_ai_imports),
        ("Scripts de lancement", test_launchers)
    ]
    
    results = []
    total_score = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            total_score += 1 if result else 0
        except Exception as e:
            print(f"Erreur lors du test {test_name}: {e}")
            results.append((test_name, False))
    
    print(f"\n RÉSULTATS FINAUX:")
    for test_name, result in results:
        status = " SUCCÈS" if result else "❌ ÉCHEC"
        print(f"  {status} - {test_name}")
    
    score_percentage = (total_score / len(tests)) * 100
    print(f"\n Score total: {total_score}/{len(tests)} ({score_percentage:.1f}%)")
    
    if score_percentage >= 80:
        print(" EXCELLENT! Le projet est prêt pour la portabilité!")
        return True
    elif score_percentage >= 60:
        print("⚠ CORRECT. Quelques améliorations recommandées.")
        return True
    else:
        print(" PROBLÈMES DÉTECTÉS. Corrections nécessaires.")
        return False

def main():
    """Fonction principale de vérification"""
    print(" VÉRIFICATION DE LA PORTABILITÉ DU PROJET MADATRADING")
    print("=" * 60)
    print(f" Répertoire de travail: {os.getcwd()}")
    print(f" Version Python: {sys.version}")
    print(f" Plateforme: {sys.platform}")
    
    success = generate_summary_report()
    
    if success:
        print("\n Le projet est prêt à être partagé sur GitHub!")
        print("\n Instructions pour l'utilisateur final:")
        print("1. git clone ou télécharger le ZIP")
        print("2. python install.py")
        print("3. python launcher.py")
    else:
        print("\n Des corrections sont nécessaires avant le partage.")
        print("\n Actions recommandées:")
        print("- Exécuter: python install.py")
        print("- Vérifier les imports manquants")
        print("- Relancer ce script de vérification")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
