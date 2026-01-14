import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 1. CARICAMENTO DATI
print("Caricamento dati...")
X = pd.read_csv("winter_project_2026/development.csv")
X_test = pd.read_csv("winter_project_2026/evaluation.csv")

# Salviamo gli ID per la submission prima di toccare qualsiasi cosa
test_ids = X_test['Id'].copy()

# 2. PULIZIA TRAIN (Qui possiamo droppare)
# Rimuovi duplicati e NaNs solo dal train
rows_duplicated = X.duplicated(subset=['article', 'title'])
X = X[~rows_duplicated]
X.dropna(inplace=True)

# Separiamo label
y = X['label']
X.drop(['label', 'Id'], inplace=True, axis=1, errors='ignore')

# 3. PULIZIA TEST (Qui NON possiamo droppare, usiamo fillna)
# Se ci sono NaN nel test, li facciamo diventare stringhe vuote per non far crashare il TF-IDF
X_test['title'] = X_test['title'].fillna("")
X_test['article'] = X_test['article'].fillna("")
# Nota: Non droppiamo 'Id' da X_test ancora, o se lo facciamo, usiamo test_ids salvato prima

# 4. FEATURE ENGINEERING (Metodo Diretto e Sicuro)
print("Creazione colonna 'text'...")
# Usiamo assegnazione diretta, molto più sicuro di concat+rename
X['text'] = X['title'] + " " + X['title'] + " " + X['article']
X_test['text'] = X_test['title'] + " " + X_test['title'] + " " + X_test['article']

# 5. PREPARAZIONE VOCABOLARIO (La tua logica custom)
print("Estrazione vocabolario specifico...")
TOP_K_PER_CLASS = 7000 
final_vocabulary = set()
classes = np.unique(y)

for label in classes:
    subset_text = X[y == label]['text']
    temp_vectorizer = TfidfVectorizer(
        max_features=TOP_K_PER_CLASS,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    temp_vectorizer.fit(subset_text)
    final_vocabulary.update(temp_vectorizer.get_feature_names_out())

final_vocabulary_list = list(final_vocabulary)
print(f"Vocabolario totale: {len(final_vocabulary_list)} parole.")

# 6. DEFINIZIONE PREPROCESSOR
encoder = OneHotEncoder(min_frequency=1, handle_unknown='infrequent_if_exist')
# Vectorizer con vocabolario forzato
master_vectorizer = TfidfVectorizer(
    vocabulary=final_vocabulary_list, 
    stop_words='english',
    ngram_range=(1, 2)
)

# Se ti da ancora errore, metti n_jobs=1 per vedere il messaggio vero
preprocessor_custom = ColumnTransformer(transformers=[
    ('source', encoder, ['source']),
    ('text', master_vectorizer, 'text'),
], remainder='drop', n_jobs=1) # Messo a 1 per sicurezza in debug

# 7. MODELLO (Logistic Regression o XGBoost)
# Usiamo LogisticRegression come nel tuo esempio (o cambia con clf XGBoost)
clf = LogisticRegression(
    random_state=RANDOM_SEED,
    C=1, penalty='l2', 
    solver='saga', 
    class_weight='balanced', 
    n_jobs=-1
)

# 8. PIPELINE COMPLETA (Fit & Predict)
full_pipeline = Pipeline([
    ('preprocessor', preprocessor_custom),
    ('classifier', clf)
])

print("Addestramento in corso...")
full_pipeline.fit(X, y)

print("Predizione sul Test Set...")
y_pred = full_pipeline.predict(X_test)

# 9. CREAZIONE FILE SUBMISSION (Corretto)
print("Salvataggio submission.csv...")
submission = pd.DataFrame({
    'Id': test_ids,    # Usiamo gli ID salvati all'inizio
    'Predicted': y_pred # O 'label' o 'Category' a seconda delle regole della gara
})

submission.to_csv('submission.csv', index=False)
print("Fatto! File pronto.")