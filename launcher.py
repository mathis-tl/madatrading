#!/usr/bin/env python3
"""
Script de lancement principal pour Madatrading
Permet de choisir quelle application lancer
"""

import os
import sys
import subprocess

def main():
    print("🚀 Bienvenue dans Madatrading AI")
    print("=" * 40)
    print("Choisissez l'application à lancer :")
    print("1. Script d'analyse simple")
    print("2. Crypto AI Analyzer (Projet complet)")
    print("3. Installation des dépendances")
    print("0. Quitter")
    print("=" * 40)
    
    choice = input("Votre choix (1-3) : ").strip()
    
    if choice == "1":
        launch_simple_script()
    elif choice == "2":
        launch_crypto_analyzer()
    elif choice == "3":
        install_dependencies()
    elif choice == "0":
        print("👋 Au revoir !")
        sys.exit(0)
    else:
        print("❌ Choix invalide. Veuillez choisir entre 1-3.")
        main()

def launch_simple_script():
    """Lance le script d'analyse simple"""
    print("📈 Lancement du script d'analyse simple...")
    try:
        subprocess.run([sys.executable, "IA_madatrading.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Erreur lors de l'exécution du script.")

def launch_crypto_analyzer():
    """Lance le projet crypto-ai-analyzer"""
    print("🔬 Lancement du Crypto AI Analyzer...")
    crypto_dir = os.path.join(os.getcwd(), "crypto-ai-analyzer")
    if os.path.exists(crypto_dir):
        os.chdir(crypto_dir)
        try:
            # Installer le package en mode développement
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
            # Lancer l'application
            subprocess.run([sys.executable, "src/main.py"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Erreur lors du lancement du Crypto AI Analyzer.")
        finally:
            os.chdir("..")
    else:
        print("❌ Le dossier crypto-ai-analyzer n'existe pas.")

def install_dependencies():
    """Installe toutes les dépendances"""
    print("📦 Installation des dépendances...")
    try:
        # Installation depuis requirements.txt
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dépendances installées avec succès !")
    except subprocess.CalledProcessError:
        print("❌ Erreur lors de l'installation des dépendances.")
        print("💡 Assurez-vous d'avoir pip installé.")

if __name__ == "__main__":
    main()
