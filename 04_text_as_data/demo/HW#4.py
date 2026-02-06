###############################################################################
# Text as Data Pipelines Tutorial: Python
# Author: Jared Edgerton
# Date: (use your local date/time)
#
# This script demonstrates (lightweight workflow):
#   1) Tokenization + basic text preprocessing
#   2) A classic topic model (LDA)
#   3) Word-embedding regression (Word2Vec -> document vectors -> Ridge regression)
#   4) A BERT-based topic model (BERTopic)
#
# Week context:
# - Text as Data Pipelines
# - Coding lab: tokenization; embeddings; topic models; basic transformer workflow.
# - Pre-class video: practical text pipeline architecture.
#
# Teaching note (important):
# - This file is intentionally written as a sequential workflow so students can
#   see how the pipeline unfolds.
# - No user-defined functions (no def ...).
# - Minimal "magic": explicit steps and prints.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Recommended installs (run once in terminal):
#
#   pip install pandas numpy matplotlib scikit-learn gensim
#
# For BERTopic (heavier; may take a bit to install):
#
#   pip install bertopic sentence-transformers umap-learn hdbscan
#
# If hdbscan fails on Windows, consider:
#   - conda install -c conda-forge hdbscan
#   - then pip install bertopic sentence-transformers umap-learn
#
# NOTE: Students can run Parts 1–3 without BERTopic if installation is a barrier.
# (BERTopic is included because it is part of this week's material.)

#pip install numpy pandas matplotlib seaborn scikit-learn gensim sentence-transformers bertopic umap-learn hdbscan wordcloud pyLDAvis

import os
import re
import random
import tarfile
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ast
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
import hdbscan
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.lda_model
import seaborn as sns
from sklearn.manifold import TSNE

# Reproducibility
random.seed(123)
np.random.seed(123)

# Create project folders (safe to run repeatedly)
os.makedirs("HW4/outputs/data_raw", exist_ok=True)
os.makedirs("HW4/outputs/data_processed", exist_ok=True)
os.makedirs("HW4/outputs/figures", exist_ok=True)
os.makedirs("HW4/outputs/outputs", exist_ok=True)
os.makedirs("HW4/outputs/src", exist_ok=True)

# -----------------------------------------------------------------------------
# Part 0: Load the CMU Movie Summary Corpus from MovieSummaries.tar.gz
# -----------------------------------------------------------------------------
# Expected archive location (based on your screenshot):
#   04_text_as_data/demo/MovieSummaries.tar.gz
#
# You can download here: http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
#
# This block:
#   1) extracts the archive (if needed)
#   2) loads plot summaries + metadata
#   3) builds df with columns: doc_id, text, y_outcome, true_topic (optional)
#
# Teaching note:
# - We keep this explicit (no helper functions) so students can follow every step.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Part 1: Tokenization + basic preprocessing
# -----------------------------------------------------------------------------
# For many "classic" text-as-data workflows, we build a document-term matrix (DTM)
# with CountVectorizer or TF-IDF. This implicitly defines a tokenizer + vocabulary.

df = pd.read_csv("week_movie_corpus.csv")

vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # words with >=2 letters
    min_df=5
)

X_counts = vectorizer.fit_transform(df["text"])
vocab = vectorizer.get_feature_names_out()

print("\n--- Document-term matrix (counts) ---")
print("Shape:", X_counts.shape)  # (n_docs, n_terms)
print("Vocabulary size:", len(vocab))
print("Example vocab terms:", vocab[:20])

# Top terms by total count (quick diagnostic)
term_totals = np.asarray(X_counts.sum(axis=0)).ravel()
top_idx = term_totals.argsort()[::-1][:15]
top_terms = pd.DataFrame({"term": vocab[top_idx], "total_count": term_totals[top_idx]})
print("\n--- Top terms by total count ---")
print(top_terms)

top_terms.to_csv("HW4/outputs/week_top_terms.csv", index=False)

# Generate a dictionary of word frequencies from the CountVectorizer results
word_freqs = dict(zip(vocab, term_totals))

# Create the Word Cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis'
).generate_from_frequencies(word_freqs)

# Plot
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Document Terms (min_df=5)", fontsize=16)
plt.savefig("HW4/outputs/figures/wordcloud.png", dpi=200)
plt.show()

# -----------------------------------------------------------------------------
# Part 2: Classic topic model (LDA)
# -----------------------------------------------------------------------------
n_topics = 6
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=123,
    learning_method="batch"
)
lda.fit(X_counts)

# Topic-word distributions
topic_word = lda.components_  # shape: (K, n_terms)

print("\n--- LDA topics: top words ---")
n_top_words = 10
for k in range(n_topics):
    top_word_idx = topic_word[k].argsort()[::-1][:n_top_words]
    words = vocab[top_word_idx]
    weights = topic_word[k][top_word_idx]
    print(f"\nTopic {k}:")
    for w, wt in zip(words, weights):
        print(f"  {w:15s} {wt:,.2f}")

# Document-topic proportions
doc_topic = lda.transform(X_counts)  # shape: (n_docs, K)
df_lda = df.copy()
df_lda["lda_topic"] = doc_topic.argmax(axis=1)
df_lda["lda_topic_prob"] = doc_topic.max(axis=1)

print("\n--- LDA: dominant topic counts ---")
print(df_lda["lda_topic"].value_counts().sort_index())

df_lda.to_csv("HW4/outputs/data_processed/week_with_lda_topics.csv", index=False)

topic_counts = df_lda["lda_topic"].value_counts().sort_index()
plt.figure(figsize=(8, 4))
plt.bar(topic_counts.index.astype(str), topic_counts.values)
plt.title("LDA: Dominant Topic Counts (Movie Plots)")
plt.xlabel("Dominant topic")
plt.ylabel("Number of documents")
plt.tight_layout()
plt.savefig("HW4/outputs/figures/week_lda_dominant_topic_counts.png", dpi=200)
plt.show()
plt.close()


lda_display = pyLDAvis.lda_model.prepare(
    lda,
    X_counts,
    vectorizer,
    mds='tsne'
)
pyLDAvis.save_html(lda_display, 'HW4/outputs/outputs/lda_visualization.html')

# -----------------------------------------------------------------------------
# Part 4: BERT-based topic model (BERTopic)
# -----------------------------------------------------------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(df["text"].tolist(), show_progress_bar=True)

print("\n--- Transformer embedding matrix ---")
print("Shape:", embeddings.shape)

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=123
)

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(df["text"].tolist(), embeddings)

df_bert = df.copy()
df_bert["bertopic_topic"] = topics
df_bert["bertopic_max_prob"] = np.max(probs, axis=1)

print("\n--- BERTopic: topic counts ---")
print(pd.Series(topics).value_counts().sort_index())

topic_info = topic_model.get_topic_info()
print("\n--- BERTopic: topic info (head) ---")
print(topic_info.head(10))

df_bert.to_csv("HW4/outputs/data_processed/week_with_bertopic.csv", index=False)
topic_info.to_csv("HW4/outputs/week_bertopic_topic_info.csv", index=False)

topic_counts_bt = topic_info.loc[topic_info["Topic"] != -1, ["Topic", "Count"]]
plt.figure(figsize=(8, 4))
plt.bar(topic_counts_bt["Topic"].astype(str), topic_counts_bt["Count"])
plt.title("BERTopic: Topic Counts (Excluding Outliers)")
plt.xlabel("Topic")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("HW4/outputs/figures/week_bertopic_topic_counts.png", dpi=200)
plt.show()
plt.close()

fig_hierarchy = topic_model.visualize_hierarchy()
fig_hierarchy.show()
fig_hierarchy.write_html("HW4/outputs/topic_hierarchy.html")

fig_docs = topic_model.visualize_documents(df["text"].tolist(), embeddings=embeddings)
fig_docs.show()
fig_docs.write_html("HW4/outputs/document_map.html")

n_topics_excl_outliers = (topic_info["Topic"] != -1).sum()
outlier_share = (pd.Series(topics) == -1).mean()

print("Number of topics (excluding outliers):", n_topics_excl_outliers)
print("Outlier share (Topic = -1):", round(outlier_share, 4))