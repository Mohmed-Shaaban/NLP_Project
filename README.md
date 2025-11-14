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
📁 NLP_Project
│
├── preprocess_and_markov.py   # Main code
├── en_ewt-ud-train.conllu     # Dataset
└── README.md                  # Documentation
```

---

## **8. How to Run**

### Install dependencies:
```bash
pip install nltk conllu tqdm
```

### Run the script:
```bash
python preprocess_and_markov.py
```

---

## **9. Oral Exam Answers (Important)**

### ❓ Why did you choose this dataset?
Because it is a high-quality, linguistically annotated dataset widely used in NLP research and ideal for applying preprocessing and language modeling.

### ❓ Why lemmatization instead of stemming?
Lemmatization keeps meaningful base forms (kill, run, eat), unlike stemming which may distort words.

### ❓ What is the Markov assumption?
Each word depends only on the previous word (bigram).

### ❓ Why remove stopwords?
They add noise and do not contribute to sentence meaning.

---

## **10. Conclusion**

This project demonstrates:

- Full NLP preprocessing pipeline  
- Clean handling of CONLL-U datasets  
- Construction of **unigram & bigram** models  
- Probability calculation for **10 sentences**  
- Clean formatted output  

The project meets all required evaluation criteria.

---
