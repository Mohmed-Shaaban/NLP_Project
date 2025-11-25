import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ===============================
# Load preprocessed sentences
# ===============================
df = pd.read_csv("sentence_probs.csv")

#Use the 'processed_sentence' column for feature extraction
sentences = df["processed_sentence"].astype(str).tolist()

print("Loaded", len(sentences), "sentences")

# ===============================
# 1) Binary Encoding
# ===============================
binary_vectorizer = CountVectorizer(binary=True)
binary_features = binary_vectorizer.fit_transform(sentences)

# Save
pd.DataFrame(binary_features.toarray(), 
             columns=binary_vectorizer.get_feature_names_out()
            ).to_csv("features_binary.csv", index=False)

print("Binary Encoding → features_binary.csv saved")

# ===============================
# 2) Count Vectorizer (Bag of Words)
# ===============================
count_vectorizer = CountVectorizer()
count_features = count_vectorizer.fit_transform(sentences)

pd.DataFrame(count_features.toarray(),
             columns=count_vectorizer.get_feature_names_out()
            ).to_csv("features_count.csv", index=False)

print("Count Vectorizer → features_count.csv saved")

# ===============================
# 3) TF-IDF
# ===============================
tfidf_vectorizer = TfidfVectorizer()
tfidf_features = tfidf_vectorizer.fit_transform(sentences)

pd.DataFrame(tfidf_features.toarray(),
             columns=tfidf_vectorizer.get_feature_names_out()
            ).to_csv("features_tfidf.csv", index=False)

print("TF-IDF → features_tfidf.csv saved")

print("\n🎉 Done! Generated 3 feature matrices.")
