import nltk
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report

# --- CONFIGURAZIONE ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- CARICAMENTO E PULIZIA DATI ---
df = pd.read_csv("winter_project_2026/development.csv")
eva = pd.read_csv("winter_project_2026/evaluation.csv")

# Rimozione duplicati e gestione valori nulli
duplicated_mask = df.duplicated(subset=['title', 'article'])
to_drop_ids = df[df['article'].isin(df[duplicated_mask]['article'])]['Id']
df.drop(index=to_drop_ids, errors='ignore', inplace=True)
df.dropna(inplace=True)

y = df['label']
# Feature Engineering: unione titolo (pesato x2) e articolo
df['text'] = df['title'] + ' ' + df['title'] + ' ' + df['article']
df.drop(columns=['label', 'title', 'article'], inplace=True)

# Split Stratificato
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y, shuffle=True
)

# --- COSTRUZIONE VOCABOLARIO OTTIMIZZATO ---
print("Estrazione vocabolario specifico per classe...")
TOP_K_PER_CLASS = 7000 
final_vocabulary = set() 
classes = np.unique(y_train)

for label in classes:
    subset_text = X_train[y_train == label]['text']
    temp_vectorizer = TfidfVectorizer(
        max_features=TOP_K_PER_CLASS,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    temp_vectorizer.fit(subset_text)
    final_vocabulary.update(temp_vectorizer.get_feature_names_out())
    print(f"Classe {label}: completata.")

final_vocabulary_list = list(final_vocabulary)
print(f"Vocabolario totale ottimizzato: {len(final_vocabulary_list)} parole.")

# --- PREPROCESSING PIPELINE ---
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

# Trasformazione dati
print("Trasformazione dati in corso...")
X_train_transformed = preprocessor.fit_transform(X_train)

# --- DEFINIZIONE MODELLI ---
clf_xgb = XGBClassifier(
    tree_method="hist", 
    n_estimators=1000, 
    max_depth=6, 
    learning_rate=0.05,
    n_jobs=-1,          
    random_state=RANDOM_SEED
)

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
    n_jobs=1  # Sequenziale per evitare overhead eccessivo con RandomizedSearch
)

# --- HYPERPARAMETER TUNING ---
param_dist = {
    'xgb__n_estimators': [500, 1000, 1500],
    'xgb__max_depth': [3, 6, 9],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__subsample': [0.7, 0.8, 0.9],
    'svm__alpha': np.logspace(-5, -2, 4),
    'svm__penalty': ['l2', 'elasticnet'],
    'weights': [[1, 1], [2, 1], [1, 2]]
}

random_search = RandomizedSearchCV(
    estimator=voting_clf,
    param_distributions=param_dist,
    n_iter=20,           
    cv=3,                
    scoring='f1_weighted',
    n_jobs=-1,           
    verbose=3,
    random_state=RANDOM_SEED
)

print("Inizio Hyperparameter Tuning...")
random_search.fit(X_train_transformed, y_train)

# --- RISULTATI ---
print(f"\nMigliori parametri: {random_search.best_params_}")
print(f"Miglior F1-Score (CV): {random_search.best_score_:.4f}")

best_model = random_search.best_estimator_