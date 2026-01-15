import nltk
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier

# --- 0. CHECK GPU (Opzionale ma utile per debug) ---
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU Rilevata: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ ATTENZIONE: GPU non rilevata. Abilita T4 in Runtime > Cambia tipo di runtime.")
except ImportError:
    pass

# --- CONFIGURAZIONE ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- CARICAMENTO E PULIZIA ---
df = pd.read_csv("winter_project_2026/development.csv")

# Rimozione duplicati ottimizzata
duplicated_mask = df.duplicated(subset=['title', 'article'])
to_drop_ids = df[df['article'].isin(df[duplicated_mask]['article'])]['Id']
df.drop(index=to_drop_ids, errors='ignore', inplace=True)
df.dropna(inplace=True)

y = df['label']
# Feature Engineering vettorizzata (più veloce)
df['text'] = df['title'] + ' ' + df['title'] + ' ' + df['article']
df.drop(columns=['label', 'title', 'article'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y, shuffle=True
)

# --- COSTRUZIONE VOCABOLARIO (PARALLELIZZATA) ---
# Ottimizzazione: Usiamo joblib per sfruttare i core CPU durante questa fase CPU-bound
print("Estrazione vocabolario specifico per classe (Parallelizzata)...")
TOP_K_PER_CLASS = 7000 
classes = np.unique(y_train)

def extract_vocab_for_class(label, X_data, y_data):
    subset_text = X_data[y_data == label]['text']
    temp_vectorizer = TfidfVectorizer(
        max_features=TOP_K_PER_CLASS,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    temp_vectorizer.fit(subset_text)
    return temp_vectorizer.get_feature_names_out()

# Esegue le estrazioni in parallelo sui core della CPU di Colab
results = Parallel(n_jobs=-1)(
    delayed(extract_vocab_for_class)(label, X_train, y_train) for label in classes
)

final_vocabulary = set().union(*results)
final_vocabulary_list = list(final_vocabulary)
print(f"Vocabolario totale ottimizzato: {len(final_vocabulary_list)} parole.")

# --- PREPROCESSING ---
encoder = OneHotEncoder(min_frequency=1, handle_unknown='infrequent_if_exist')
master_vectorizer = TfidfVectorizer(
    vocabulary=final_vocabulary_list,
    stop_words='english',
    ngram_range=(1, 2)
)

preprocessor = ColumnTransformer(transformers=[
    ('source', encoder, ['source']),
    ('text', master_vectorizer, 'text'),
], remainder='drop', n_jobs=-1)

print("Trasformazione dati in corso...")
X_train_transformed = preprocessor.fit_transform(X_train)

# --- MODELLI OTTIMIZZATI PER GPU ---

# 1. XGBoost: Attivazione GPU (device="cuda")
clf_xgb = XGBClassifier(
    tree_method="hist", 
    device="cuda",      # <--- LA CHIAVE DELLA VELOCITÀ SU COLAB
    n_estimators=1000, 
    max_depth=6, 
    learning_rate=0.05,
    random_state=RANDOM_SEED
)

# 2. SVM: Rimane su CPU (è leggero grazie a SGD)
clf_svm = SGDClassifier(
    loss='modified_huber', 
    penalty='l2', 
    alpha=1e-4, 
    max_iter=1000, 
    tol=1e-3,
    class_weight='balanced',
    n_jobs=-1,
    random_state=RANDOM_SEED
)

voting_clf = VotingClassifier(
    estimators=[('xgb', clf_xgb), ('svm', clf_svm)],
    voting='soft',
    weights=[1, 1],
    n_jobs=1 
)

# --- TUNING SEQUENZIALE (GPU-SAFE) ---
param_dist = {
    'xgb__n_estimators': [500, 1000, 1500],
    'xgb__max_depth': [3, 6, 9],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__subsample': [0.7, 0.8, 0.9],
    'svm__alpha': np.logspace(-5, -2, 4),
    'svm__penalty': ['l2', 'elasticnet'],
    'weights': [[1, 1], [2, 1], [1, 2]]
}

# n_jobs=1 qui è fondamentale. 
# La velocità la otteniamo dalla GPU dentro XGBoost, non dal parallelismo di Sklearn.
random_search = RandomizedSearchCV(
    estimator=voting_clf,
    param_distributions=param_dist,
    n_iter=20,           
    cv=3,                
    scoring='f1_weighted',
    n_jobs=1,            # <--- IMPORTANTE: Evita conflitti sulla GPU
    verbose=3,
    random_state=RANDOM_SEED
)

print("Inizio Hyperparameter Tuning (GPU Powered)...")
random_search.fit(X_train_transformed, y_train)

# --- RISULTATI ---
print(f"\nMigliori parametri: {random_search.best_params_}")
print(f"Miglior F1-Score: {random_search.best_score_:.4f}")

best_model = random_search.best_estimator_