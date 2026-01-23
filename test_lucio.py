import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import re
import string
from bs4 import BeautifulSoup

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 1. CARICAMENTO DATI
print("Caricamento dati...")
X = pd.read_csv("winter_project_2026/development.csv")
X_test = pd.read_csv("winter_project_2026/evaluation.csv")

# Salviamo gli ID per la submission prima di toccare qualsiasi cosa
test_ids = X_test['Id'].copy()

rows_duplicated = X.duplicated(subset=['article', 'title'])
X = X[~rows_duplicated]
X.dropna(inplace=True)

y = X['label']
X.drop(['label', 'Id'], inplace=True, axis=1, errors='ignore')

X_test['title'] = X_test['title'].fillna("")
X_test['article'] = X_test['article'].fillna("")

print("Preprocessing testi...")
def clean_text(text):
    # 1. FIX ANTI-CRASH: Se non è una stringa (es. è NaN/float), restituisci vuoto
    if not isinstance(text, str):
        return ""
    
    # 2. FIX LOGICO: BeautifulSoup va PRIMA delle regex
    # Altrimenti rimuovi le < > e BS4 non riconosce più l'HTML
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ") 
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip() # Rimuove spazi doppi creati dalle regex
    return text
# Convertiamo preventivamente in stringa per sicurezza (.astype(str))
# Questo risolve alla radice il problema del 'float object'
X['article'] = X['article'].astype(str).apply(clean_text)
X_test['article'] = X_test['article'].astype(str).apply(clean_text)

# Facciamo lo stesso per i titoli (visto che li usi per la colonna 'text')
X['title'] = X['title'].fillna("").astype(str).apply(clean_text)
X_test['title'] = X_test['title'].fillna("").astype(str).apply(clean_text)
    
print("Creazione colonna 'text'...")


X['text'] = X['title'] + " " + X['title'] + " " + X['article']+ " " + X['title']
X_test['text'] = X_test['title'] + " " + X_test['title'] + " " + X_test['article']+ " " + X_test['title']

TOP_K_PER_CLASS = 7000 
final_vocabulary = set() 
encoder = OneHotEncoder(min_frequency=100, handle_unknown='infrequent_if_exist')
print("Estrazione vocabolario specifico per classe...")
classes = np.unique(y)
for label in classes:
    subset_text = X[y == label]['text']
    temp_vectorizer = TfidfVectorizer(
        max_features=TOP_K_PER_CLASS,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        sublinear_tf=True
    )
    temp_vectorizer.fit(subset_text)
    words = temp_vectorizer.get_feature_names_out()
    final_vocabulary.update(words)
    print(f"Classe {label}: trovate {len(words)} parole specifiche.")
final_vocabulary_list = list(final_vocabulary)
print(f"Vocabolario totale ottimizzato: {len(final_vocabulary_list)} parole uniche.")

master_vectorizer = TfidfVectorizer(
    vocabulary=final_vocabulary_list,
    stop_words='english',
    ngram_range=(1, 2),
    sublinear_tf=True
)
preprocessor_custom = ColumnTransformer(transformers=[
    ('source', encoder, ['source']),
    ('text', master_vectorizer, 'text'),
], remainder='drop', n_jobs=-1)

X_train_final = preprocessor_custom.fit_transform(X)
X_test_final = preprocessor_custom.transform(X_test)

clf = LogisticRegression(
    random_state=RANDOM_SEED,
    C=1, 
    penalty='l2', 
    solver='saga',
    class_weight='balanced',
    n_jobs=-1)

clf.fit(X_train_final, y)
y_pred = clf.predict(X_test_final)


# 9. CREAZIONE FILE SUBMISSION (Corretto)
print("Salvataggio submission.csv...")
submission = pd.DataFrame({
    'Id': test_ids,    # Usiamo gli ID salvati all'inizio
    'Predicted': y_pred # O 'label' o 'Category' a seconda delle regole della gara
})

submission.to_csv('submission.csv', index=False)
print("Fatto! File pronto.")