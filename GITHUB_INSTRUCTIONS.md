# Instructions pour publier sur GitHub

## 1. Initialiser le repository Git (si pas déjà fait)
```bash
git init
git add .
git commit -m "Initial commit - Madatrading suite"
```

## 2. Créer un repository sur GitHub
1. Aller sur github.com
2. Cliquer sur "New repository"
3. Nommer le repository "madatrading"
4. Laisser public ou choisir privé selon vos préférences
5. NE PAS initialiser avec README (on a déjà le nôtre)

## 3. Connecter le repository local à GitHub
```bash
git remote add origin https://github.com/VOTRE-USERNAME/madatrading.git
git branch -M main
git push -u origin main
```

## 4. Vérification
- Le README.md s'affiche correctement sur GitHub
- Les utilisateurs peuvent cloner avec: `git clone https://github.com/VOTRE-USERNAME/madatrading.git`
- L'installation fonctionne avec: `python install.py`

## 5. Instructions pour les utilisateurs finaux (à inclure dans votre README)
```markdown
### Installation rapide
1. `git clone https://github.com/VOTRE-USERNAME/madatrading.git`
2. `cd madatrading`
3. `python install.py`
4. `python launcher.py`
```

Le projet est maintenant prêt pour GitHub ! 🚀
