# News Article Classification

Supervised machine learning pipeline for multi-class news article classification developed at **Politecnico di Torino**.

This project focuses on building an efficient and robust classifier capable of categorizing web-scraped news articles while handling noisy HTML content, metadata signals, and high-dimensional text representations.

🏆 **Final Result:** Ranked **3rd out of 200 groups** in the course competition.

---

## Authors

- Leonardo Passafiume  
- Lucio Baiocchi  

Politecnico di Torino

---

## Problem Overview

The goal of this project is to classify news articles into **seven categories**:

| Label | Category |
|------|----------|
| 0 | International News |
| 1 | Business |
| 2 | Technology |
| 3 | Entertainment |
| 4 | Sports |
| 5 | General News |
| 6 | Health |

### Dataset

The dataset contains:

- **~80,000 labeled articles** for training and validation  
- **20,000 unlabeled samples** for evaluation  

Each sample includes:

- Article title
- Raw HTML article content
- News source
- Timestamp
- PageRank score

The dataset presents several challenges:

- **Class imbalance**
- **Noisy HTML artifacts**
- **Semantic overlap between categories**
- **Short article texts**

---

## Project Highlights

Key elements that contributed to the final performance:

- Advanced **feature engineering**
- Exploitation of **metadata signals**
- Hybrid **word + character TF-IDF representations**
- Efficient **linear models for high-dimensional sparse data**

The final system achieved:

- **Validation Macro F1:** 0.726  
- **Public Test Macro F1:** **0.741**

---

## Methodology

### 1. Data Exploration

Initial analysis revealed several important dataset characteristics:

- Slight **class imbalance**
- Strong **vocabulary overlap** between certain classes  
  (especially *International News* and *General News*)
- Certain news sources are **highly correlated with specific categories**

Example:

- `ESPN` → almost exclusively **Sports**
- `PCWorld` → almost exclusively **Technology**

These patterns were exploited in later stages.

---

## Feature Engineering

### Text Boosting Strategy

A key idea in our pipeline is **metadata boosting**.

Sources with high label purity were identified and given higher importance by **replicating metadata fields** in the input text.

Two configurations were used:

**Pure Sources**
source ×4 + title ×4 + article ×2

**Standard Sources**

source ×2 + title ×3 + article ×2


This approach increases the **TF-IDF weight** of highly informative metadata.

---

### Additional Features

#### Temporal Features

Extracted from the article timestamp:

- Day of the week
- Time-of-day bins:
  - Morning → 06:00–14:59
  - Afternoon → 15:00–20:59
  - Night → 21:00–05:59

#### Structural Features

- Article length
- Title length

Lengths were transformed using:

log(1 + x)


to capture magnitude rather than raw counts.

---

## Feature Pipeline

The final preprocessing pipeline uses a **ColumnTransformer** with multiple parallel feature extractors.

### Text Features

1. **Word-level TF-IDF**
2. **Character-level TF-IDF n-grams**
3. **Snowball stemming**
4. **Feature selection (SelectKBest)**

### Categorical Features

- News source
- Temporal features

Encoded with:

OneHotEncoder(min_frequency=50)


### Numerical Features

- PageRank score
- Log-transformed length features

Standardized using:

StandardScaler

---

## Models Evaluated

Several models were tested:

| Model | Validation Macro F1 |
|------|---------------------|
| Naive Baseline | 0.443 |
| Random Forest | 0.689 |
| Linear SVM (LinearSVC) | 0.703 |
| **SGDClassifier (SVM)** | **0.726** |

The best performance was achieved using:

SGDClassifier


with an SVM-style loss and strong regularization.

Reasons for choosing this model:

- Efficient on **high-dimensional sparse data**
- Scales well to large datasets
- Flexible regularization options

---

## Hyperparameter Optimization

Hyperparameters were tuned using:

GridSearchCV (5-fold cross-validation)


Parameters explored included:

### TF-IDF Parameters

- Word vocabulary size
- Character n-gram vocabulary
- Feature selection thresholds

### Classifier Parameters

- Loss function
- Regularization strength (`alpha`)
- Penalty (`l2`, `elasticnet`)

---

## Error Analysis

Confusion matrix analysis revealed the main classification challenges:

### Major Confusion Pairs

- **International News ↔ General News**
- **Entertainment ↔ International News**

This behavior reflects the **lexical similarity between these categories**, where many high-frequency words overlap.

### Strongest Performing Classes

- **Sports**
- **Technology**

These benefited heavily from **high-purity news sources**.

---

## Final Result

🏆 **3rd place out of 200 groups** in the course competition.

Final score:
Public Macro F1: 0.741


---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- TF-IDF Vectorization
- Support Vector Machines
- SGDClassifier

