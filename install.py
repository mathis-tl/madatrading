#!/usr/bin/env python3
"""
Script d'installation automatique pour Madatrading
Permet d'installer toutes les dépendances et de configurer l'environnement
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Execute une commande et affiche le résultat"""
    print(f"\n📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} - Succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Erreur: {e}")
        print(f"Sortie d'erreur: {e.stderr}")
        return False

def check_python_version():
    """Vérifie la version de Python"""
    version = sys.version_info
    print(f"🐍 Version de Python détectée: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 ou supérieur est requis!")
        return False
    
    print("✅ Version de Python compatible")
    return True

def install_dependencies():
    """Installe toutes les dépendances"""
    print("\n🔧 Installation des dépendances...")
    
    # Mise à jour de pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Mise à jour de pip")
    
    # Installation des dépendances depuis requirements.txt
    if os.path.exists("requirements.txt"):
        if run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "Installation depuis requirements.txt"):
            return True
    
    # Installation manuelle des dépendances principales si requirements.txt n'existe pas
    dependencies = [
        "numpy", "pandas", "matplotlib", "seaborn", "scikit-learn",
        "tensorflow", "yfinance", "requests", "plotly",
        "sqlalchemy", "jupyter", "xgboost"
    ]
    
    for dep in dependencies:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Installation de {dep}"):
            print(f"⚠️ Problème avec l'installation de {dep}, mais on continue...")
    
    return True

def setup_directories():
    """Crée les répertoires nécessaires"""
    print("\n📁 Création des répertoires...")
    
    directories = [
        "outputs",
        "crypto-ai-analyzer/outputs", 
        "crypto-ai-analyzer/models/saved_models",
        "crypto-ai-analyzer/data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Répertoire créé: {directory}")

def create_env_file():
    """Crée un fichier .env avec les paramètres par défaut"""
    print("\n⚙️ Configuration de l'environnement...")
    
    if not os.path.exists(".env"):
        env_content = """# Configuration Madatrading
API_KEY=""
DATABASE_URL="sqlite:///crypto_data.db"
DEFAULT_API="coingecko"
DEBUG=False
LOG_LEVEL="INFO"
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Fichier .env créé avec la configuration par défaut")
    else:
        print("✅ Fichier .env déjà existant")

def test_installation():
    """Test l'installation en important les modules principaux"""
    print("\n🧪 Test de l'installation...")
    
    test_imports = [
        ("numpy", "np"),
        ("pandas", "pd"), 
        ("matplotlib.pyplot", "plt"),
        ("sklearn", None),
        ("tensorflow", "tf")
    ]
    
    success_count = 0
    for module, alias in test_imports:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✅ {module} - OK")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module} - Erreur: {e}")
    
    if success_count == len(test_imports):
        print(f"\n🎉 Tous les modules ({success_count}/{len(test_imports)}) sont installés correctement!")
        return True
    else:
        print(f"\n⚠️ {success_count}/{len(test_imports)} modules installés avec succès")
        return False

def main():
    """Fonction principale d'installation"""
    print("🚀 INSTALLATION DE MADATRADING")
    print("=" * 50)
    
    # Informations système
    print(f"💻 Système d'exploitation: {platform.system()} {platform.release()}")
    print(f"🏗️ Architecture: {platform.machine()}")
    
    # Vérifications préliminaires
    if not check_python_version():
        sys.exit(1)
    
    # Installation
    try:
        setup_directories()
        create_env_file()
        
        if install_dependencies():
            if test_installation():
                print("\n🎉 INSTALLATION TERMINÉE AVEC SUCCÈS!")
                print("\n📖 Prochaines étapes:")
                print("1. Pour lancer le menu principal: python launcher.py")
                print("2. Pour analyser les cryptos directement: python IA_madatrading.py")
                print("3. Pour l'IA avancée: cd crypto-ai-analyzer && python src/main.py")
                print("4. Consultez README.md pour plus d'informations")
                return True
        
        print("\n⚠️ Installation terminée avec quelques avertissements")
        print("Essayez de lancer les applications pour voir si elles fonctionnent")
        return True
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Installation interrompue par l'utilisateur")
        return False
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
