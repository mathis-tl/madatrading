"""
Configuration de l'installation du package crypto-ai-analyzer
"""

from setuptools import setup, find_packages
import os

# Lire le fichier README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Analyseur IA de crypto-monnaies"

# Lire les requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    # Liste de base des dépendances si requirements.txt n'existe pas
    return [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "yfinance>=0.1.70",
        "requests>=2.25.0",
        "sqlalchemy>=1.4.0",
        "plotly>=5.0.0",
        "jupyter>=1.0.0",
        "xgboost>=1.5.0"
    ]

setup(
    name="crypto-ai-analyzer",
    version="1.0.0",
    author="Mathis Telle",
    author_email="contact@madatrading.com",
    description="Un outil d'analyse pour la crypto-monnaie utilisant l'intelligence artificielle.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/mathistelle/madatrading",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-analyzer=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
)