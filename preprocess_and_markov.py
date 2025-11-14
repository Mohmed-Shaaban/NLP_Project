# ============================
# FIX SSL for Windows (NLTK)
# ============================
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ============================
# IMPORTS
# ============================
from conllu import parse_incr
import nltk
from nltk.corpus import stopwords
import math
from collections import Counter
from tqdm import tqdm
import re
import os

# ============================
# DOWNLOAD STOPWORDS
# ============================
nltk.download("stopwords")
STOP = set(stopwords.words("english"))

# ============================
# HELPER: REMOVE NUMBERS / PUNCT
# ============================
def is_punct_or_num(token):
    if re.fullmatch(r"\d+(\.\d+)?", token):
        return True
    if re.fullmatch(r"\W+", token):
        return True
    return False

# ============================
# READ SENTENCES FROM CONLLU
# ============================
def read_conllu_sentences(conllu_path):
    with open(conllu_path, "r", encoding="utf-8") as fh:
        for tokenlist in parse_incr(fh):
            tokens = []
            for t in tokenlist:
                if isinstance(t["id"], int):
                    tokens.append((t["form"], t.get("lemma")))
            yield tokens

# ============================
# PREPROCESS TOKENS
# ============================
def preprocess_tokens(token_tuples, lowercase=True, remove_stopwords=True, use_lemma=True):
    out = []
    for form, lemma in token_tuples:
        t = lemma if (use_lemma and lemma and lemma != "_") else form
        if lowercase:
            t = t.lower()
        if is_punct_or_num(t):
            continue
        if remove_stopwords and t in STOP:
            continue
        if len(t.strip()) == 0:
            continue
        out.append(t)
    return out

# ============================
# BUILD NGRAM COUNTS
# ============================
def build_ngram_counts(sentences):
    unigram = Counter()
    bigram = Counter()
    for sent in sentences:
        tokens = ["<s>"] + sent + ["</s>"]
        for i in range(len(tokens)):
            unigram[tokens[i]] += 1
            if i >= 1:
                bigram[(tokens[i-1], tokens[i])] += 1
    return unigram, bigram

# ============================
# BIGRAM PROBABILITY (ADD-ONE)
# ============================
def sentence_prob_bigram(sentence_tokens, unigram, bigram, V):
    tokens = ["<s>"] + sentence_tokens + ["</s>"]
    logprob = 0.0
    for i in range(1, len(tokens)):
        prev = tokens[i-1]
        cur = tokens[i]
        num = bigram.get((prev, cur), 0) + 1
        den = unigram.get(prev, 0) + V
        logprob += math.log(num / den)
    return math.exp(logprob), logprob

# ============================
# BEAUTIFUL OUTPUT WITH BEFORE/AFTER PREPROCESSING
# ============================
def print_sentence_probs_beautiful(sentences_raw, sentences_proc, unigram, bigram, V):
    print("\n" + "="*70)
    print("        Sentence Probabilities (Before & After Preprocessing)")
    print("="*70 + "\n")
    
    for i, (raw, proc) in enumerate(zip(sentences_raw, sentences_proc)):
        prob, logp = sentence_prob_bigram(proc, unigram, bigram, V)
        raw_str = " ".join(raw)
        proc_str = " ".join(proc)
        
        print(f"Sentence {i+1}:")
        print(f"  Before Preprocessing: {raw_str}")
        print(f"  After  Preprocessing: {proc_str}")
        print(f"  Probability   : {prob:.4e}")
        print(f"  Log-Probability: {logp:.4f}")
        print("-"*70)

# ============================
# MAIN FUNCTION
# ============================
def main(conllu_path, n_examples=10, use_lemma=True):
    print(f"Reading dataset from: {conllu_path}\n")
    sentences_raw = []
    sentences_proc = []

    for token_tuples in tqdm(read_conllu_sentences(conllu_path)):
        raw_tokens = [t[0] for t in token_tuples if isinstance(t[0], str)]
        proc_tokens = preprocess_tokens(token_tuples, use_lemma=use_lemma)
        
        if proc_tokens:
            sentences_raw.append(raw_tokens)
            sentences_proc.append(proc_tokens)
    
    print(f"\nTotal preprocessed sentences collected: {len(sentences_proc)}\n")

    # build ngrams
    unigram, bigram = build_ngram_counts(sentences_proc)
    V = len(unigram)
    print(f"Vocabulary size: {V}\n")

    # take first n_examples sentences
    print_sentence_probs_beautiful(
        sentences_raw[:n_examples],
        sentences_proc[:n_examples],
        unigram, bigram, V
    )

# ============================
# RUN
# ============================
if __name__ == "__main__":
    # ضع هنا المسار الصحيح بعد فك الضغط من zip
    conllu_file = r"C:\Users\Mohme\Downloads\UD_English-EWT-master\en_ewt-ud-train.conllu"

    if not os.path.exists(conllu_file):
        print("\nERROR: File not found! Please extract the zip and set correct path.\n")
    else:
        main(conllu_file, n_examples=10, use_lemma=True)
