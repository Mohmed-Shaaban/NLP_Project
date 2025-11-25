# **NLP Project – Text Preprocessing & N-Gram Sentence Probability**

## **1. Project Overview**
This project applies standard NLP preprocessing steps to a linguistic dataset and then computes sentence probabilities using an **N-Gram (Bigram) Language Model** with the **Markov assumption**.

The project satisfies the requirements of the first evaluation:

1. **Data Selection & Preprocessing**
2. **N-Gram Sentence Probability Calculation**

---

## **2. Dataset Description (UD_English-EWT)**

The dataset used is **Universal Dependencies English Web Treebank (EWT)** in **CONLL-U format**.

### **Why I Chose This Dataset**
- A standard benchmark used in NLP research.
- Already segmented into **sentences**, ready for processing.
- Contains **lemmas**, POS tags, morphological features → ideal for preprocessing.
- Works perfectly with Python libraries like `conllu`.
- High quality, well-structured, and widely used.

---

## **3. Understanding the CONLL-U Format**

Each sentence is represented by a block of rows.  
Each row has 10 columns:

| Column | Meaning | Example |
|--------|---------|---------|
| **ID** | Token index | 1 |
| **FORM** | Word as it appears in text | killed |
| **LEMMA** | Base form | kill |
| **UPOS** | Universal POS tag | VERB |
| **XPOS** | Language-specific POS | VBD |
| **FEATS** | Morphological features | Tense=Past |
| **HEAD** | Head token index | 7 |
| **DEPREL** | Dependency relation | obj |
| **DEPS** | Enhanced dependencies | _ |
| **MISC** | Other info | SpaceAfter=No |

This structure makes it easy to extract (FORM, LEMMA) pairs.

---

## **4. Preprocessing Steps**

### ✔ Lower casing  
Convert all tokens to lowercase.

### ✔ Lemmatization  
Use the `lemma` column from the dataset.

### ✔ Remove punctuation  
Using regex to detect punctuation-only tokens.

### ✔ Remove numbers  
Remove tokens that are integers or floats.

### ✔ Remove stopwords  
Using:  
```python
from nltk.corpus import stopwords
```

### ✔ Remove empty tokens  
Ensure only clean tokens remain.

### After preprocessing example  
**Before:**  
`American forces killed Shaikh Abdullah al-Ani, near the Syrian border.`

**After:**  
`american force kill shaikh abdullah ani syrian border`

---

## **5. N-Gram Language Model**

We compute probabilities using **Bigram Model** with **Add-1 Smoothing**.

## **Mathematical Formulas**

### **Sentence Probability (Bigram Model)**

$$
P(w_1, w_2, \ldots, w_n)
= P(w_1) \times \prod_{i=2}^{n} P(w_i \mid w_{i-1})
$$

---

### **Bigram Conditional Probability (Add-1 Smoothing)**

$$
P(w_i \mid w_{i-1})
= \frac{\text{Count}(w_{i-1},\, w_i) + 1}{\text{Count}(w_{i-1}) + V}
$$

Where:

- **Count(wᵢ₋₁, wᵢ)** = frequency of the bigram  
- **Count(wᵢ₋₁)** = frequency of the previous word  
- **V** = vocabulary size  
- Add-1 smoothing prevents zero probabilities  



## **6. Output Format (Before & After Preprocessing)**

The script prints each sentence like this:

```
=====================================================
Sentence #1 (Original):
American forces killed Shaikh Abdullah al-Ani, near the Syrian border.

After Preprocessing:
american force kill shaikh abdullah ani syrian border

Bigram Probability:
2.481e-07
=====================================================
```

This format is used for all 10 sentences.

---

## **7. Project Structure**

```
NLP_PROJECT/
│
├── .vscode/                  
│
├── data/                     
│   ├── features/             # Extracted features from dataset
│   │   ├── features_binary.csv    # Binary features per token
│   │   ├── features_count.csv     # Count-based features
│   │   └── features_tfidf.csv     # TF-IDF weighted features
│   │
│   ├── processed/            # Processed datasets
│   │   └── sentence_probs.csv      
│   │
│   └── raw/                  # Raw dataset files
│       └── en_ewt-ud-train.conllu  
│
├── src/                      
│   ├── features/             # Feature extraction scripts
│   │   └── feature_extraction.py   # Extract binary, count, and TF-IDF features
│   │
│   └── preprocessing/        # Preprocessing scripts
│       └── preprocess_and_markov.py  # Text cleaning, tokenization, and N-Gram sentence probability
│
├── venv/                     # Virtual environment
│
└── README.md                 

```

---

# Feature Extraction

After preprocessing, the **feature extraction module** converts cleaned text into numerical vectors suitable for machine learning and NLP tasks. Algorithms cannot operate on raw text, so structured feature matrices are essential.

## Objectives
- Generate multiple vectorized representations of the text
- Compare how different feature types capture linguistic characteristics
- Save feature matrices in reusable CSV files
- Prepare data for modeling, clustering, or statistical analysis

## Extracted Feature Types

### 1. Count Vectorization
- Converts each document into word frequency counts  
- Simple and interpretable; ideal as a baseline  
- **Output:** `data/features/features_count.csv`

### 2. Binary Vectorization
- Marks each token as `1` (present) or `0` (absent)  
- Ignores frequency; focuses on vocabulary usage  
- **Output:** `data/features/features_binary.csv`

### 3. TF-IDF (Term Frequency – Inverse Document Frequency)
- Weighs terms by importance across the dataset  
- Highlights informative words and down-weights common words  
- **Output:** `data/features/features_tfidf.csv`

## Implementation Details
**Script:** `src/features/feature_extraction.py`

**Steps:**
1. Load preprocessed text (cleaned tokens)  
2. Initialize vectorizers using scikit-learn:  
   - `CountVectorizer`  
   - `TfidfVectorizer`  
   - Binary variant of `CountVectorizer`  
3. Generate feature matrices: Count, Binary, TF-IDF  
4. Save matrices as CSV in `data/features/`

> All matrices share the same vocabulary to ensure consistency.

## Why These Features?

| Feature Type | Strength                  | Best Use Case              |
|--------------|--------------------------|----------------------------|
| Count        | Simple & interpretable   | Baseline models           |
| Binary       | Removes frequency bias   | When word presence matters |
| TF-IDF       | Highlights important words | Most ML & NLP tasks      |

> Using multiple feature types allows flexibility for experimentation with different models.

## Future Extensions
- N-gram features (bigrams, trigrams)  
- Word embeddings (Word2Vec, GloVe)  
- Sentence embeddings (BERT, SBERT)  
- POS-tag or dependency-based features
---

---

## **9. How to Run**

### Install dependencies:
```bash
pip install nltk conllu tqdm
```

### 9.1 Run preprocessing and Markov probability script:
```bash
python preprocess_and_markov.py
```
### 9.2 Run feature extraction:
```bash
python feature_extraction.py
```

---

## **10. (Important)**

### ❓ Why did you choose this dataset?
Because it is a high-quality, linguistically annotated dataset widely used in NLP research and ideal for applying preprocessing and language modeling.

### ❓ Why lemmatization instead of stemming?
Lemmatization keeps meaningful base forms (kill, run, eat), unlike stemming which may distort words.

### ❓ What is the Markov assumption?
Each word depends only on the previous word (bigram).

### ❓ Why remove stopwords?
They add noise and do not contribute to sentence meaning.

### ❓ What is feature extraction in this project?

Feature extraction converts tokens into structured representations for machine learning:

Binary features: Whether a token has a certain property (e.g., POS tag, capitalization).

Count-based features: How often a token or property occurs.

TF-IDF features: How important a token is in the dataset relative to other tokens.

---

## **10. Conclusion**

This project demonstrates:

- Full NLP preprocessing pipeline  
- Clean handling of CONLL-U datasets  
- Construction of **unigram & bigram** models  
- Probability calculation for **10 sentences**  
- Extraction of binary, count-based, and TF-IDF features for each token
-  Clean formatted output

The project meets all required evaluation criteria.

---
