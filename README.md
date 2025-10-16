# Product Category Classifier

Acesta este un proiect de clasificare automata a produselor pe categorii, pe baza titlului produsului. Scopul este de a usura si accelera procesul de introducere a produselor pe platforma online.

## Descriere
Modelul foloseste tehnici de preprocesare a textului si algoritmi de machine learning pentru a prezice categoria unui produs. Proiectul include:
- Notebook-uri Jupyter pentru explorarea datelor, preprocesare si antrenarea modelului.
- Scripturi Python pentru antrenarea modelului (`train_model.py`) si testare interactiva (`predict_category.py`).
- Model antrenat salvat in format `.pkl`.
- Feature engineering suplimentar pentru imbunatatirea preciziei.

## Structura proiectului
product-category-classifier/

-data/ # date brute și procesate
 
 --products.csv

-notebooks/ # notebook-uri Jupyter/Colab

-scripts/ # scripturi Python
 
 -train_model.py
 
 --predict_category.py

-models/ # modelele antrenate (.pkl)

-README.md

-requirements.txt

## Instalare
1. Cloneaza repozitoriul:
git clone https://github.com/username/product-category-classifier.git

cd product-category-classifier

2. Creeaza un mediu virtual si instaleaza dependintele:

python -m venv venv

source venv/bin/activate # Linux / Mac

venv\Scripts\activate # Windows

pip install -r requirements.txt

## Utilizare

### Antrenare model
python scripts/train_model.py

### Testare interactiva
python scripts/predict_category.py
Introduceti titlul produsului si modelul va prezice categoria.

## Contributii
Contributiile sunt binevenite! Faceti fork la proiect si trimiteti pull requests.
