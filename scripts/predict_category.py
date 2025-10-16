import re
import pickle
import numpy as np
from scipy.sparse import hstack

# Incarcare model si vectorizator
with open('models/product_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Functie curatare titlu
def clean_title(title):
    title = str(title).lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title

# Functie predictie
def predict_category(title):
    clean = clean_title(title)
    x_text = tfidf.transform([clean])
    x_features = np.array([[len(clean), len(clean.split()), int(bool(re.search(r'\d', clean)))]])
    x = hstack([x_text, x_features])
    pred = model.predict(x)
    return pred[0]

# Loop interactiva
while True:
    title = input("Introdu titlul produsului (sau 'exit' pentru a iesi): ")
    if title.lower() == 'exit':
        break
    category = predict_category(title)
    print(f"Categoria prezisa: {category}\n")
