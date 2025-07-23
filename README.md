# 🚀 Madatrading - Suite d'Analyse Crypto avec IA

Une suite d'outils d'analyse de crypto-monnaies avec IA avancée pour la prédiction des prix et l'analyse technique automatisée.

## 📋 Pré-requis

- **Python 3.8 ou supérieur**
- **Système d'exploitation**: Windows, macOS, ou Linux
- **Connexion internet** pour télécharger les données crypto

## 🚀 Installation Rapide (Recommandée)

### Étape 1: Télécharger le projet
```bash
# Via Git
git clone https://github.com/votre-username/madatrading.git
cd madatrading

# Ou téléchargez et extraire le ZIP depuis GitHub
```

### Étape 2: Installation automatique
```bash
# Lance l'installation automatique de toutes les dépendances
python install.py
```

Le script d'installation va:
- ✅ Vérifier votre version de Python
- ✅ Installer toutes les dépendances nécessaires
- ✅ Créer les dossiers requis
- ✅ Configurer l'environnement
- ✅ Tester l'installation

### Étape 3: Lancer les applications
```bash
# Lance le menu de sélection des applications
python launcher.py
```

## 🎯 Applications Disponibles

### 1. 📈 Script d'Analyse Simple
- Analyse rapide et basique de Bitcoin et Ethereum
- Prédictions simples basées sur la tendance
- Idéal pour débuter et tester le système

**Commande directe :**
```bash
python IA_madatrading.py
```

### 2. 🤖 Crypto AI Analyzer (Avancé - RECOMMANDÉ)
- Modèles d'apprentissage automatique avancés (LSTM, XGBoost)
- Analyse technique complète avec 15+ indicateurs
- Prédictions de prix multi-cryptos (BTC, ETH, XRP, ADA, SOL)
- Optimisation d'hyperparamètres automatique
- Visualisations détaillées et métriques de performance
- Sauvegarde des modèles et résultats

**Commande directe :**
```bash
cd crypto-ai-analyzer
python src/main.py
```

## 🛠️ Installation Manuelle (Alternative)

Si l'installation automatique ne fonctionne pas:

```bash
# 1. Installer les dépendances de base
pip install -r requirements.txt

# 2. Installer le package crypto-ai-analyzer
cd crypto-ai-analyzer
pip install -e .
cd ..

# 3. Créer les dossiers nécessaires
mkdir -p outputs crypto-ai-analyzer/outputs crypto-ai-analyzer/models/saved_models

# 4. Lancer une application
python launcher.py
```

## 📁 Structure du Projet

```
madatrading/
├── 📄 install.py              # Installation automatique
├── 📄 launcher.py             # Menu de lancement
├── 📄 requirements.txt        # Dépendances globales
├── 📄 README.md              # Ce fichier
├── 📄 .env.example           # Configuration exemple
├── 📈 IA_madatrading.py      # Script d'analyse simple
└── 📁 crypto-ai-analyzer/    # Module IA avancé
    ├── src/                  # Code source
    │   ├── main.py          # Point d'entrée principal
    │   ├── config.py        # Configuration
    │   ├── data/            # Collecte de données
    │   ├── analysis/        # Analyse technique
    │   ├── models/          # Modèles ML/DL
    │   └── visualization/   # Graphiques
    ├── outputs/             # Résultats/graphiques
    ├── requirements.txt     # Dépendances spécifiques
    └── setup.py            # Configuration du package
```

## ⚡ Utilisation Rapide

### Lancement via le Menu
```bash
python launcher.py
```

### Analyse Rapide (Script Simple)
```bash
python IA_madatrading.py
```

### Analyse Avancée avec IA
```bash
cd crypto-ai-analyzer
python src/main.py
```

## 🎯 Fonctionnalités Principales

### 📊 Analyse Technique Complète
- **Indicateurs de tendance :** Moyennes mobiles (MA7, MA30, MA50)
- **Oscillateurs :** RSI, MACD, Stochastique
- **Volatilité :** Bandes de Bollinger, ATR
- **Volume :** OBV (On Balance Volume)
- **Support/Résistance :** Détection automatique

### 🤖 Intelligence Artificielle
- **LSTM (Long Short-Term Memory)** pour prédictions temporelles
- **XGBoost** pour prédictions basées sur les caractéristiques
- **Random Forest** pour validation croisée
- **Hyperparameter tuning** automatique
- **Validation croisée** temporelle

### 📈 Crypto-monnaies Supportées
- **Bitcoin (BTC)**
- **Ethereum (ETH)**
- **Ripple (XRP)**
- **Cardano (ADA)**
- **Solana (SOL)**

### 🎨 Visualisations
- Graphiques de prix avec indicateurs techniques
- Heatmaps de corrélation
- Métriques de performance des modèles
- Comparaisons multi-cryptos

## 🔧 Configuration

### Variables d'Environnement (Optionnel)
Copiez `.env.example` vers `.env` et modifiez selon vos besoins:
```bash
cp .env.example .env
```

### Personnalisation
- **Cryptos à analyser**: Modifiez `crypto-ai-analyzer/src/config.py`
- **Paramètres des modèles**: Ajustez `MODEL_CONFIG` dans `config.py`

## 🚨 Résolution des Problèmes

### Problème: "Module not found"
```bash
# Réinstallez les dépendances
python install.py
```

### Problème: Erreur TensorFlow
```bash
# Pour les utilisateurs Mac avec Apple Silicon
pip install tensorflow-macos tensorflow-metal

# Pour les autres
pip install tensorflow==2.13.0
```

### Problème: Données non récupérées
- Vérifiez votre connexion internet
- Le système utilise Yahoo Finance comme source principale
- Certaines APIs ont des limitations de débit

### Problème: Erreur de mémoire
- Réduisez `window_size` dans `config.py`
- Réduisez `epochs` ou `batch_size`

## 📊 Résultats et Outputs

Les résultats sont automatiquement sauvegardés dans:
- `outputs/` : Graphiques et analyses du script principal
- `crypto-ai-analyzer/outputs/` : Résultats détaillés de l'IA
  - Graphiques de prédictions pour chaque crypto
  - Métriques de performance (MSE, MAE, R²)
  - Analyses d'hyperparamètres
  - Modèles entraînés sauvegardés

## 🔄 Mise à Jour

Pour mettre à jour le projet:
```bash
git pull origin main  # Si installé via Git
python install.py     # Réinstaller les dépendances si nécessaire
```

## 🤝 Support

- **Documentation**: Ce README et les commentaires dans le code
- **Problèmes**: Créez une issue sur GitHub
- **Questions**: Consultez le code source documenté

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 🎉 Démarrage Rapide pour les Impatients

1. **Téléchargement**: `git clone` ou télécharger le ZIP
2. **Installation**: `python install.py`
3. **Lancement**: `python launcher.py`
4. **Choix recommandé**: Option 2 - Crypto AI Analyzer
5. **Résultats**: Consultez le dossier `crypto-ai-analyzer/outputs/`

**Premier lancement réussi ?** ✅ Vous êtes prêt à analyser les cryptos avec l'IA !

**Des problèmes ?** 📧 Consultez la section "Résolution des Problèmes" ci-dessus.

## 🏆 Exemple de Résultats

Après exécution du Crypto AI Analyzer, vous obtiendrez :

```
📊 Collecte des données...
✓ 364 points de données collectés pour BTC
✓ 364 points de données collectés pour ETH
[... autres cryptos ...]

🔧 Prétraitement des données...
✓ Indicateurs techniques ajoutés
✓ Données nettoyées et normalisées

🤖 Entraînement des modèles...
✓ Modèle LSTM entraîné pour BTC (MSE: 0.045)
✓ Modèle XGBoost entraîné pour BTC (R²: 0.87)
[... autres cryptos ...]

📈 Génération des prédictions...
✓ Prédictions générées pour les 30 prochains jours

🎨 Visualisations créées...
✓ Graphiques sauvegardés dans outputs/
```

Le projet est maintenant prêt et optimisé ! 🚀
