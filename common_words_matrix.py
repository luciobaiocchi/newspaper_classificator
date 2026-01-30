
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os

# Ensure output directory exists
os.makedirs('final_img', exist_ok=True)

# Load Data
try:
    df = pd.read_csv("winter_project_2026/development.csv")
    print(f"Loaded dataset with {len(df)} samples")
except FileNotFoundError:
    print("Error: 'winter_project_2026/development.csv' not found.")
    exit(1)

# Helper function to get top N words for a given label
def get_top_words(text_series, n=100):
    # Using CountVectorizer to find "most common" words (highest frequency)
    # Tfidf is good for importance, but user asked for "common". 
    # However, to filter out stopwords efficiently and be consistent with prior analysis 
    # that might have used TFIDF, let's use CountVectorizer with English stop words.
    
    vec = CountVectorizer(stop_words='english', max_features=10000)
    X = vec.fit_transform(text_series.dropna())
    sum_words = X.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return set([w[0] for w in words_freq[:n]])

labels = sorted(df['label'].unique())
top_words_per_label = {}

print("Extracting top 100 words for each label...")
for label in labels:
    # Combining title and article for better representation
    text_data = df[df['label'] == label]['article'].fillna('') + " " + df[df['label'] == label]['title'].fillna('')
    top_words_per_label[label] = get_top_words(text_data, 100)

# Initialize Matrix
n_labels = len(labels)
overlap_matrix = np.zeros((n_labels, n_labels), dtype=int)

print("Calculating overlaps...")
for i in range(n_labels):
    for j in range(n_labels):
        intersection = top_words_per_label[labels[i]].intersection(top_words_per_label[labels[j]])
        overlap_matrix[i, j] = len(intersection)

# Plotting
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")
ax = sns.heatmap(overlap_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=labels, yticklabels=labels)

plt.title('Common Words in Top 100 Keywords (Pairwise Class Overlap)')
plt.xlabel('Class Label')
plt.ylabel('Class Label')

# Highlight 0 vs 5 specifically if needed, but the heatmap shows it all.
plt.tight_layout()

# Save
pdf_path = 'final_img/common_words_matrix.pdf'
png_path = 'final_img/common_words_matrix.png'

plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.savefig(png_path, dpi=300, bbox_inches='tight')

print(f"Plots saved to:\n{pdf_path}\n{png_path}")
