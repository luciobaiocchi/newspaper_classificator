"""
Solution V20 - Pure SGD Optimized
==================================
Based on V18 findings, but with more aggressive hyperparams:
- alpha=5e-5 (between 1e-4 and 1e-5)
- Larger feature space
- NO ensemble (sometimes simpler is better)

Also trying different text combinations for boosting.
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, f1_score
from nltk.stem import SnowballStemmer
import warnings
warnings.filterwarnings('ignore')

stemmer = SnowballStemmer("english")

def stemmed_tokenizer(text):
    tokens = re.findall(r'(?u)\b\w\w+\b', text)
    return [stemmer.stem(t) for t in tokens]


def preprocess_dataframe(df, is_train=True):
    if is_train:
        df = df.copy()
        df.drop_duplicates(inplace=True)
        df = df.dropna()
    else:
        df = df.copy()
        df['title'] = df['title'].fillna('')
        df['article'] = df['article'].fillna('')
        df['source'] = df['source'].fillna('Unknown')
    
    # STRONGER source boosting - x4 instead of x3
    pure_sources = [
        'Syfy.com', 'Topix', 'PCWorld', 'BCC', 'Computerworld', 
        'Register', 'Search', 'CNET', 'InfoWorld', 'IPS', 'Topix.Net',
        'Motley', 'Ananova', 'Forbes', 'Newsweek', 'News', 'CNN',
        'Sports', 'ESPN'
    ]
    
    s = df['source']
    t = df['title']
    a = df['article']
    
    # V20: Stronger boosting pattern
    text_boosted = s + ' ' + s + ' ' + s + ' ' + s + ' ' + t + ' ' + t + ' ' + t + ' ' + t + ' ' + a + ' ' + a
    text_standard = s + ' ' + s + ' ' + t + ' ' + t + ' ' + t + ' ' + a + ' ' + a
    
    df['text'] = np.where(df['source'].isin(pure_sources), text_boosted, text_standard)
    
    # Temporal features
    weekdays = []
    daytime = []
    for day in df['timestamp']:
        try:
            if day == '0000-00-00 00:00:00':
                week_day = -1
                hour_day = -1
            else:
                ts = pd.Timestamp(day)
                week_day = ts.day_of_week
                hour = ts.hour
                if 5 < hour <= 14:
                    hour_day = 1
                elif 14 < hour <= 21:
                    hour_day = 2
                elif (21 < hour <= 23) or (0 <= hour <= 5):
                    hour_day = 3
                else:
                    hour_day = 0
        except:
            week_day = -1
            hour_day = -1
        
        daytime.append(hour_day)
        weekdays.append(week_day)
        
    df['day_of_week'] = weekdays
    df['moment_of_day'] = daytime
    
    df['article_len'] = df['article'].apply(len)
    df['title_len'] = df['title'].fillna('').apply(len)
    df['log_article_len'] = np.log1p(df['article_len'])
    df['log_title_len'] = np.log1p(df['title_len'])
    
    return df


def create_pipeline():
    encoder = OneHotEncoder(min_frequency=50, handle_unknown='ignore')  # Lowered
    
    # V20: Maximum features
    text_pipe = Pipeline([
        ('vec', TfidfVectorizer(
            max_features=100000,  # Maximum
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            tokenizer=stemmed_tokenizer
        )),
        ('sel', SelectKBest(chi2, k=60000))  # Maximum
    ])
    
    char_pipe = Pipeline([
        ('vec', TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=3,
            max_features=50000,  # Maximum
            sublinear_tf=True
        )),
        ('sel', SelectKBest(chi2, k=25000))  # Maximum
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('source', encoder, ['source', 'day_of_week', 'moment_of_day']),
        ('text', text_pipe, 'text'),
        ('text_char', char_pipe, 'text'),
        ('pagerank', StandardScaler(), ['page_rank']),
        ('lengths', StandardScaler(), ['log_article_len', 'log_title_len'])
    ], remainder='drop', n_jobs=-1)
    
    # V20: Pure SGD with fine-tuned params
    clf = SGDClassifier(
        loss='modified_huber',
        penalty='l2',
        alpha=1e-4,  # Between 5e-5 and 1e-4
        max_iter=10000,  # More iterations
        tol=1e-7,  # Tighter tolerance
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        early_stopping=False,
        shuffle=True
    )
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])


def run_validation():
    print("=" * 60)
    print("Solution V20 - Pure SGD Optimized")
    print("=" * 60)
    
    df = pd.read_csv("winter_project_2026/development.csv")
    print("Preprocessing...")
    df = preprocess_dataframe(df, is_train=True)
    
    y = df['label']
    X = df.drop('label', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    pipeline = create_pipeline()
    
    print("\nTraining...")
    pipeline.fit(X_train, y_train)
    
    print("\nEvaluating on validation set...")
    y_pred = pipeline.predict(X_test)
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(classification_report(y_test, y_pred))
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\n>>> Macro F1 Score: {macro_f1:.4f}")
    print("=" * 60)
    
    return pipeline, macro_f1


def generate_submission():
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSION")
    print("=" * 60)
    
    df_train = pd.read_csv("winter_project_2026/development.csv")
    print("Preprocessing train...")
    df_train = preprocess_dataframe(df_train, is_train=True)
    
    y_train = df_train['label']
    X_train = df_train.drop('label', axis=1)
    
    df_eval = pd.read_csv("winter_project_2026/evaluation.csv")
    eval_ids = df_eval['Id'].copy()
    print("Preprocessing eval...")
    df_eval = preprocess_dataframe(df_eval, is_train=False)
    
    pipeline = create_pipeline()
    
    print(f"\nTraining on full dataset ({len(X_train)} samples)...")
    pipeline.fit(X_train, y_train)
    
    print("Generating predictions...")
    y_pred = pipeline.predict(df_eval)
    
    submission = pd.DataFrame({
        'Id': eval_ids,
        'Predicted': y_pred
    })
    
    submission_file = 'submission_v20_pure_sgd.csv'
    submission.to_csv(submission_file, index=False)
    print(f"\n>>> Submission saved to: {submission_file}")
    print(f">>> Total predictions: {len(submission)}")
    print("=" * 60)
    
    return submission


if __name__ == "__main__":
#    pipeline, score = run_validation()
    print("\n")
    submission = generate_submission()
