from setuptools import setup, find_packages

setup(
    name='crypto-ai-analyzer',
    version='0.1.0',
    author='Mathis Telle',
    author_email='votre.email@example.com',
    description='Un outil d\'analyse pour la crypto-monnaie utilisant l\'intelligence artificielle.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'requests',
        'jupyter'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)