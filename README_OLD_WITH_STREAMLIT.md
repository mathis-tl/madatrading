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

### 1. 🤖 Crypto AI Analyzer (Recommandé)
- Modèles d'apprentissage automatique (LSTM, Machine Learning)
- Prédictions de prix avancées avec métriques de performance
- Optimisation d'hyperparamètres automatique
- Analyse technique complète (RSI, MACD, Bollinger, etc.)
- Génération de graphiques de prédiction
- **Commande :** `python launcher.py` → Option 3

### 2. 📈 Script d'Analyse Simple
- Analyse rapide avec prédictions de base
- Idéal pour les tests et analyses rapides  
- Utilise des modèles simples de régression
- **Commande :** `python launcher.py` → Option 2

### 3. 📊 Application Complète (En développement)
- Interface utilisateur complète
- Tableaux de bord personnalisables
- **Commande :** `python launcher.py` → Option 1

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
├── 📈 IA_madatrading.py      # Script simple d'analyse
└── 📁 crypto-ai-analyzer/    # Module IA avancé
    ├── src/                  # Code source
    ├── outputs/              # Résultats/graphiques
    ├── requirements.txt      # Dépendances spécifiques
    └── setup.py             # Configuration du package
```

## ⚡ Utilisation Rapide

### Lancement via le Menu
```bash
python launcher.py
```

### Lancement Direct des Applications

**🤖 Analyse IA Avancée (Recommandée) :**
```bash
cd crypto-ai-analyzer
python src/main.py
```

**📈 Script Simple :**
```bash
python IA_madatrading.py
```

**🎯 Via le Menu Interactif :**
```bash
python launcher.py
# Choisir option 2 (Script simple) ou 3 (IA avancée)
```

## 🔧 Configuration

### Variables d'Environnement (Optionnel)
Copiez `.env.example` vers `.env` et modifiez selon vos besoins:
```bash
cp .env.example .env
```

### Personnalisation
- **Cryptos à analyser**: Modifiez `crypto-ai-analyzer/src/config.py`
- **Paramètres d'IA**: Ajustez les hyperparamètres dans le même fichier

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
pip install tensorflow==2.8.0
```

### Problème: Données non récupérées
- Vérifiez votre connexion internet
- Le système utilise Yahoo Finance comme source fiable par défaut

### Problème: Erreurs d'entraînement IA
- Réduisez le nombre de cryptos dans `crypto-ai-analyzer/src/config.py`
- Vérifiez que TensorFlow est correctement installé

## 📊 Résultats

Les résultats sont automatiquement sauvegardés dans:
- `crypto-ai-analyzer/outputs/` : Graphiques de prédiction et métriques de performance
- Métriques calculées : RMSE, MAE, MAPE, R² pour chaque crypto-monnaie

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

1. `git clone` ou télécharger le ZIP
2. `python install.py`
3. `python launcher.py`
4. Choisir "Crypto AI Analyzer" (option 3)
5. Admirer les prédictions IA ! 🚀

**Premier lancement réussi ?** ✅ Vous avez maintenant un système d'analyse crypto avec IA !

**Des questions ?** 📧 Consultez la section "Résolution des Problèmes" ci-dessus.
