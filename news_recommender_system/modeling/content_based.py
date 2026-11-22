"""Content-based recommenders: TF-IDF and sentence-transformer article similarity.

Every candidate article of an impression is scored by its mean cosine similarity to the
articles in the user's click history; candidates without a usable history keep their
original order. The article similarity matrix is dense (``n_news x n_news``), so the full
MIND-small validation news set needs roughly 16 GB of RAM.

The TF-IDF model reads the history of the first impression of each user in the validation
data, the sentence-transformer model the history of the last one.
"""

from pathlib import Path
import string
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from news_recommender_system.modeling.ranking import rank_positions, write_predictions

SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"


def combine_news_text(news: pd.DataFrame) -> pd.Series:
    """Concatenate title, abstract, category and subcategory of every article."""
    news = news.fillna("")
    return news["Title"] + " " + news["Abstract"] + " " + news["Category"] + " " + news["SubCat"]


def _preprocess_for_tfidf(texts: pd.Series) -> pd.Series:
    """Lowercase, tokenize, and drop punctuation and English stopwords."""
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    for resource in ("punkt_tab", "stopwords"):
        nltk.download(resource, quiet=True)
    stop_words = set(stopwords.words("english"))

    def preprocess(text: str) -> str:
        tokens = [t for t in word_tokenize(text.lower()) if t not in string.punctuation]
        return " ".join(t for t in tokens if t not in stop_words)

    return texts.apply(preprocess)


def tfidf_similarity(news: pd.DataFrame) -> np.ndarray:
    """Cosine similarity between all articles based on TF-IDF of their text."""
    texts = _preprocess_for_tfidf(combine_news_text(news))
    tfidf = TfidfVectorizer(stop_words="english", min_df=0.001, max_df=0.999, ngram_range=(1, 3))
    vectors = tfidf.fit_transform(texts)
    return cosine_similarity(vectors, vectors)


def sentence_transformer_similarity(
    news: pd.DataFrame, model_name: str = SENTENCE_TRANSFORMER_MODEL
) -> np.ndarray:
    """Cosine similarity between all articles based on sentence-transformer embeddings."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        combine_news_text(news).tolist(), show_progress_bar=True, normalize_embeddings=True
    )
    return cosine_similarity(embeddings, embeddings)


def rank_for_user(
    history: object,
    impression_news: list[str],
    news_id_to_idx: dict[str, int],
    similarity: np.ndarray,
) -> list[str]:
    """Order ``impression_news`` by mean similarity to the articles in ``history``."""
    if pd.isna(history) or not isinstance(history, str) or not history.strip():
        return impression_news

    clicked_indices = [news_id_to_idx[n] for n in history.split() if n in news_id_to_idx]
    if not clicked_indices:
        return impression_news

    scores = []
    for news_id in impression_news:
        news_idx = news_id_to_idx.get(news_id)
        if news_idx is None:
            scores.append((news_id, 0.0))
        else:
            scores.append((news_id, float(np.mean(similarity[news_idx, clicked_indices]))))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [news_id for news_id, _ in scores]


def content_rankings(
    behaviors_val: pd.DataFrame,
    news_ids: list[str],
    similarity: np.ndarray,
    history_from: str = "first",
) -> Iterator[tuple[object, list[int]]]:
    """Yield ``(impression_id, ranks)`` for every validation impression.

    ``history_from`` selects which impression of a user provides the click history:
    ``"first"`` or ``"last"`` in time order.
    """
    news_id_to_idx = {news_id: idx for idx, news_id in enumerate(news_ids)}
    user_history = (
        behaviors_val.drop_duplicates("User", keep=history_from).set_index("User")["History"]
    ).to_dict()

    for impression_id, user_id, impressions in zip(
        behaviors_val["ImpressionId"], behaviors_val["User"], behaviors_val["Impressions"]
    ):
        candidates = [entry.rsplit("-", 1)[0] for entry in impressions.split()]
        ranked = rank_for_user(user_history[user_id], candidates, news_id_to_idx, similarity)
        yield impression_id, rank_positions(ranked, candidates)


def generate_content_predictions(
    behaviors_val: pd.DataFrame,
    news_val: pd.DataFrame,
    output_path: Path,
    method: str = "tfidf",
) -> None:
    """Write content-based predictions (``method`` is ``"tfidf"`` or ``"sbert"``)."""
    if method == "tfidf":
        similarity = tfidf_similarity(news_val.fillna(""))
        history_from = "first"
    elif method == "sbert":
        similarity = sentence_transformer_similarity(news_val)
        history_from = "last"
    else:
        raise ValueError(f"Unknown method: {method!r}")

    news_ids = news_val["NewsId"].tolist()
    write_predictions(
        content_rankings(behaviors_val, news_ids, similarity, history_from), output_path
    )
