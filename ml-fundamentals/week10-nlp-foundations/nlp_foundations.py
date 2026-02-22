"""
Phase 2 — Week 10: NLP Foundations
=====================================
Covers: SpaCy, NLTK, tokenization, POS, NER, TF-IDF, text classification
Job relevance: 80% of AI/ML job postings list NLP/SpaCy/NLTK skills
This is PRE-LLM NLP — the foundation interviewers still test even for LLM roles.
"""

import re
import string
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_20newsgroups

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("vader_lexicon", quiet=True)


# ── Text Preprocessing Pipeline ──────────────────────────────────────────────
class TextPreprocessor:
    """
    Standard NLP preprocessing pipeline.
    Each step matters for downstream model performance.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        stemming: bool = False,
        lemmatization: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization

        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()

        # Remove URLs and special characters
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        tokens = text.split()

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        if self.lemmatization:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        elif self.stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return " ".join(tokens)

    def preprocess_batch(self, texts: list[str]) -> list[str]:
        return [self.preprocess(t) for t in texts]


# ── NLTK: POS Tagging ─────────────────────────────────────────────────────────
def pos_tagging_nltk(text: str) -> list[tuple[str, str]]:
    """
    Part-of-speech tagging with NLTK.
    Used in information extraction, grammar parsing, and NER preprocessing.
    """
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags


# ── SpaCy: Named Entity Recognition ──────────────────────────────────────────
def ner_spacy(texts: list[str], model: str = "en_core_web_sm") -> list[dict]:
    """
    Named Entity Recognition with SpaCy.
    SpaCy is the industry standard for production NLP pipelines.
    Entity types: PERSON, ORG, GPE (geopolitical), DATE, MONEY, etc.
    """
    try:
        nlp = spacy.load(model)
    except OSError:
        # Fallback: use blank model for demo
        print(f"Note: SpaCy model '{model}' not found. Run: python -m spacy download {model}")
        return [{"text": t, "entities": [], "note": "model not available"} for t in texts]

    results = []
    for doc_text in texts:
        doc = nlp(doc_text)
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "description": spacy.explain(ent.label_),
            }
            for ent in doc.ents
        ]
        results.append({"text": doc_text, "entities": entities, "n_tokens": len(doc)})
    return results


# ── TF-IDF Text Representation ────────────────────────────────────────────────
def tfidf_analysis(corpus: list[str], top_n: int = 10) -> dict:
    """
    TF-IDF: Term Frequency × Inverse Document Frequency.
    The workhorse feature representation before word embeddings dominated.
    Still used in production for search, recommendations, and classification.
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # Top TF-IDF terms per document (first 3 docs)
    top_terms_per_doc = []
    for i in range(min(3, tfidf_matrix.shape[0])):
        row = tfidf_matrix[i].toarray().flatten()
        top_idx = row.argsort()[-top_n:][::-1]
        top_terms_per_doc.append([
            (feature_names[j], float(row[j])) for j in top_idx if row[j] > 0
        ])

    return {
        "vocabulary_size": len(feature_names),
        "matrix_shape": tfidf_matrix.shape,
        "top_terms_doc_0": top_terms_per_doc[0] if top_terms_per_doc else [],
        "top_terms_doc_1": top_terms_per_doc[1] if len(top_terms_per_doc) > 1 else [],
    }


# ── Text Classification Pipeline ─────────────────────────────────────────────
def text_classification_pipeline(
    categories: list[str] = None,
    n_samples: int = 2000,
) -> dict:
    """
    Full text classification pipeline: TF-IDF + Logistic Regression.
    This is the baseline every NLP system should beat.
    """
    if categories is None:
        categories = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]

    newsgroups = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )

    preprocessor = TextPreprocessor()
    X = preprocessor.preprocess_batch(newsgroups.data[:n_samples])
    y = newsgroups.target[:n_samples]

    # Pipeline 1: TF-IDF + Logistic Regression
    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=500, C=1.0, random_state=42)),
    ])

    # Pipeline 2: BoW + Naive Bayes (fast baseline)
    nb_pipeline = Pipeline([
        ("bow", CountVectorizer(max_features=10000)),
        ("clf", MultinomialNB(alpha=0.1)),
    ])

    lr_scores = cross_val_score(lr_pipeline, X, y, cv=5, scoring="accuracy")
    nb_scores = cross_val_score(nb_pipeline, X, y, cv=5, scoring="accuracy")

    return {
        "categories": categories,
        "n_samples": len(X),
        "logreg_tfidf": {
            "mean_accuracy": float(lr_scores.mean()),
            "std": float(lr_scores.std()),
        },
        "naive_bayes_bow": {
            "mean_accuracy": float(nb_scores.mean()),
            "std": float(nb_scores.std()),
        },
    }


# ── Sentiment Analysis ────────────────────────────────────────────────────────
def sentiment_analysis_vader(texts: list[str]) -> pd.DataFrame:
    """
    VADER (Valence Aware Dictionary and Sentiment Reasoner).
    Rule-based, optimized for social media / short texts. No training required.
    """
    sia = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        scores = sia.polarity_scores(text)
        results.append({
            "text": text[:60] + "..." if len(text) > 60 else text,
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "compound": scores["compound"],
            "sentiment": (
                "positive" if scores["compound"] >= 0.05
                else "negative" if scores["compound"] <= -0.05
                else "neutral"
            ),
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    # ── POS tagging ──────────────────────────────────────────────────────────
    print("=== POS Tagging (NLTK) ===")
    sample = "The machine learning model achieved 94% accuracy on the test dataset."
    tags = pos_tagging_nltk(sample)
    print(f"Text: {sample}")
    print(f"Tags: {tags}\n")

    # ── NER with SpaCy ───────────────────────────────────────────────────────
    print("=== Named Entity Recognition (SpaCy) ===")
    texts = [
        "Apple CEO Tim Cook announced new AI chips at WWDC in San Francisco.",
        "Google DeepMind released AlphaFold 3 in London last May, impressing researchers worldwide.",
    ]
    ner_results = ner_spacy(texts)
    for r in ner_results:
        print(f"\nText: {r['text'][:80]}")
        for ent in r["entities"]:
            print(f"  [{ent['label']}] {ent['text']} — {ent['description']}")

    # ── Text preprocessing ───────────────────────────────────────────────────
    print("\n=== Text Preprocessing ===")
    proc = TextPreprocessor()
    raw = "The AI models are running faster! Check https://example.com for updates."
    cleaned = proc.preprocess(raw)
    print(f"Raw:     {raw}")
    print(f"Cleaned: {cleaned}")

    # ── TF-IDF ───────────────────────────────────────────────────────────────
    print("\n=== TF-IDF Analysis ===")
    corpus = [
        "machine learning models require feature engineering and cross validation",
        "deep learning with pytorch and tensorflow for image classification tasks",
        "natural language processing uses tokenization and part of speech tagging",
        "mlops pipelines automate model training deployment and monitoring in production",
    ]
    tfidf = tfidf_analysis(corpus)
    print(f"Vocabulary size: {tfidf['vocabulary_size']}")
    print(f"Matrix shape: {tfidf['matrix_shape']}")
    print(f"Top terms (doc 0): {tfidf['top_terms_doc_0'][:5]}")

    # ── Text Classification ──────────────────────────────────────────────────
    print("\n=== Text Classification (20 Newsgroups) ===")
    clf_result = text_classification_pipeline(n_samples=1000)
    print(f"TF-IDF + LogReg: {clf_result['logreg_tfidf']['mean_accuracy']:.4f} ± "
          f"{clf_result['logreg_tfidf']['std']:.4f}")
    print(f"BoW + NaiveBayes: {clf_result['naive_bayes_bow']['mean_accuracy']:.4f} ± "
          f"{clf_result['naive_bayes_bow']['std']:.4f}")

    # ── Sentiment Analysis ───────────────────────────────────────────────────
    print("\n=== Sentiment Analysis (VADER) ===")
    reviews = [
        "This AI model is absolutely incredible! Best performance I've seen.",
        "The model keeps failing and the documentation is terrible.",
        "Accuracy is 87%. Latency could be better but acceptable for production.",
    ]
    sentiment_df = sentiment_analysis_vader(reviews)
    print(sentiment_df[["text", "compound", "sentiment"]].to_string(index=False))
