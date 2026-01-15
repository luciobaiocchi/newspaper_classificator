# --- SCRIPT MAC: TUNING SVM ---
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. CONFIGURAZIONE (DEVE ESSERE IDENTICA SU ENTRAMBI)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 2. CARICAMENTO DATI
df = pd.read_csv("winter_project_2026/development.csv") # Assicurati del percorso corretto
duplicated_mask = df.duplicated(subset=['title', 'article'])
to_drop_ids = df[df['article'].isin(df[duplicated_mask]['article'])]['Id']
df.drop(index=to_drop_ids, errors='ignore', inplace=True)
df.dropna(inplace=True)

y = df['label']
df['text'] = df['title'] + ' ' + df['title'] + ' ' + df['article']
df.drop(columns=['label', 'title', 'article'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y, shuffle=True
)

# 3. VOCABOLARIO E PREPROCESSING
# (Copia qui la logica di estrazione vocabolario o usane uno standard per velocità se vuoi solo testare)
# Per coerenza con Colab, idealmente dovresti usare lo stesso vocab.
# Se vuoi fare veloce sul Mac, usa un TF-IDF standard per il tuning:
print("Preparazione Preprocessor...")
preprocessor = ColumnTransformer(transformers=[
    ('source', OneHotEncoder(handle_unknown='ignore'), ['source']),
    ('text', TfidfVectorizer(max_features=25000, stop_words='english', ngram_range=(1,2)), 'text'),
], remainder='drop', n_jobs=-1)

X_train_transformed = preprocessor.fit_transform(X_train)

# 4. TUNING SVM (SGD)
clf_svm = SGDClassifier(loss='modified_huber', class_weight='balanced', random_state=RANDOM_SEED)

param_svm = {
    'alpha': np.logspace(-5, -2, 10),
    'penalty': ['l2', 'elasticnet'],
    'l1_ratio': [0.15, 0.5, 0.85] # Rilevante solo se penalty='elasticnet'
}

print("🚀 Inizio Tuning SVM su CPU (M1)...")
search_svm = RandomizedSearchCV(
    clf_svm, 
    param_svm, 
    n_iter=20, 
    cv=3, 
    scoring='f1_weighted', 
    n_jobs=-1, # <--- TUTTA POTENZA CPU MAC
    verbose=1
)

search_svm.fit(X_train_transformed, y_train)

print(f"\n✅ MIGLIOR SVM: {search_svm.best_params_}")
print(f"Score: {search_svm.best_score_}")