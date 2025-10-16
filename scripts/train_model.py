import pandas as pd
import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
from scipy.sparse import hstack
import pickle

# Functie curatare titlu
def clean_title(title):
    title = str(title).lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title

# Incarcare date
df = pd.read_csv('data/products.csv')

# Stergere spatii in coloane
df.columns = df.columns.str.strip()

# Eliminare randuri fara categorie
df = df.dropna(subset=['Category Label'])

# Curata titlul
df['clean_title'] = df['Product Title'].apply(clean_title)

# Feature engineering
df['title_len'] = df['clean_title'].apply(len)
df['title_word_count'] = df['clean_title'].apply(lambda x: len(x.split()))
df['has_number'] = df['clean_title'].apply(lambda x: int(bool(re.search(r'\d', x))))

# Vectorizare TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(df['clean_title'])
X_features = df[['title_len','title_word_count','has_number']].values
X = hstack([X_text, X_features])

y = df['Category Label']

# Impartire train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definire modele
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, solver='saga'),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Multinomial NB": MultinomialNB()
}

best_model = None
best_f1 = 0

# Comparare modele
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Weighted F1-score: {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# Salvare cel mai bun model
with open('models/product_classifier.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print(f"\nCel mai bun model salvat cu F1-score: {best_f1:.4f} in folderul 'models/'")