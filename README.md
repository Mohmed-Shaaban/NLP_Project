📘 NLP Preprocessing & N-Gram Language Model Project
Course Evaluation – Text Normalization & Sentence Probability Calculation
⭐ Project Overview

This project demonstrates essential NLP preprocessing techniques and applies N-gram language modeling (Markov assumption) to compute the probability of sentences from a dataset in CONLL-U format.

The workflow includes:

Dataset selection (UD English EWT)

Text preprocessing

Segmentation

Tokenization

Lowercasing

Stopword removal

Removing numbers & punctuation

Lemmatization

Building N-gram counts

Computing sentence probabilities

Displaying clean, well-formatted output

This project is designed to satisfy the first evaluation requirements for the NLP course.

📂 Dataset Used: UD English EWT (CONLLU format)

We use a well-known linguistic dataset: Universal Dependencies – English Web Treebank (EWT).

Why choose this dataset?

✔ Free and publicly available
✔ Clean, annotated, linguistically valid
✔ Contains high-quality segmentation, POS tags, lemmas, dependencies
✔ Perfect for NLP preprocessing tasks
✔ Lemmas are included → helpful for normalization

Structure of Each Row (CONLL-U)
Column	Meaning
ID	Token index inside the sentence
FORM	Original word in the text
LEMMA	Normalized dictionary form
UPOS	Universal part of speech
XPOS	Language-specific part of speech
FEATS	Morphological features (Gender, Case, Number...)
HEAD	Governor (dependency parent)
DEPREL	Dependency relation type
DEPS	Enhanced dependencies
MISC	Extra metadata (SpaceAfter, alignment…)
⚙️ Installation & Setup
1️⃣ Install required libraries
pip install nltk tqdm conllu

2️⃣ Download NLTK resources
import nltk
nltk.download("stopwords")

🧹 Preprocessing Steps (Explained)

Your code performs:

✓ 1. Token extraction

Reads each sentence from the CONLLU file.

✓ 2. Lowercasing

All text converted to lowercase.

✓ 3. Lemmatization

Using the LEMMA column (if available).

✓ 4. Removing:

Stopwords

Numbers

Punctuation

Empty tokens

Non-alphabetic symbols

✓ 5. Return clean, normalized tokens
🧮 N-Gram Model (Markov Assumption)

Given a sentence:
I love natural language processing

The 2-gram probability is:

𝑃
(
𝑆
)
=
𝑃
(
𝐼
)
×
𝑃
(
𝑙
𝑜
𝑣
𝑒
∣
𝐼
)
×
𝑃
(
𝑛
𝑎
𝑡
𝑢
𝑟
𝑎
𝑙
∣
𝑙
𝑜
𝑣
𝑒
)
×
𝑃
(
𝑙
𝑎
𝑛
𝑔
𝑢
𝑎
𝑔
𝑒
∣
𝑛
𝑎
𝑡
𝑢
𝑟
𝑎
𝑙
)
×
𝑃
(
𝑝
𝑟
𝑜
𝑐
𝑒
𝑠
𝑠
𝑖
𝑛
𝑔
∣
𝑙
𝑎
𝑛
𝑔
𝑢
𝑎
𝑔
𝑒
)
P(S)=P(I)×P(love∣I)×P(natural∣love)×P(language∣natural)×P(processing∣language)

We compute:

N-gram counts

Conditional probabilities

Final probability per sentence

▶️ Running the Project

Modify the file path:

conllu_path = "path/to/dataset.conllu"


Then run:

python main.py


The output will show:

🟦 Original Sentence
🟩 After Preprocessing
🟧 N-gram Probability
📤 Output Example (Styled)
====================================
Sentence #1 (Original):
I really love learning NLP and I enjoy text processing.

Preprocessed:
love learn nlp enjoy text processing

2-Gram Probability:
1.2357e-12
====================================

📌 Project Files
File	Description
main.py	Contains the full preprocessing + N-gram probability calculation
README.md	Project documentation
dataset.conllu	The dataset used
🧑‍🏫 Why This Project Is Important (Interview / Exam Points)

Shows understanding of text normalization pipelines

Demonstrates applying probability using N-gram models

Works with real linguistic datasets (CONLLU)

Uses lemmatization, which is more advanced than stemming

Shows ability to produce clean, structured output

Demonstrates practical NLP skills (preprocessing + modeling)

🙌 Author

Mohamed — NLP Course Project
Faculty of Computers & Information
Mansoura University
