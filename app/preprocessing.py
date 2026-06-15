"""
NLP Preprocessing Pipeline — 7 étapes identiques au notebook TALN.
"""

import re
import string
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Lazy NLTK download
def _ensure_nltk():
    import nltk
    for pkg in ["stopwords", "punkt", "wordnet", "punkt_tab", "averaged_perceptron_tagger_eng"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


NEGATIONS = {
    "no", "not", "nor", "never", "neither", "nothing", "nobody", "nowhere", "none",
    "cannot", "cant", "wont", "dont", "didnt", "doesnt", "isnt", "wasnt",
    "wouldnt", "shouldnt", "couldnt", "neednt", "hasnt", "hadnt",
}

_PONCTUATION = set(string.punctuation)

@lru_cache(maxsize=1)
def _get_nltk_resources():
    _ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    stop_words = set(stopwords.words("english")) - NEGATIONS
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer, word_tokenize, sent_tokenize


def preprocess(text: str) -> tuple[str, list[str]]:
    """
    Apply the 7-step NLP pipeline and return (clean_text, steps_applied).
    """
    steps = []
    stop_words, lemmatizer, word_tokenize, sent_tokenize = _get_nltk_resources()

    # Step 1 — Clean URLs, @mentions, #hashtags, non-ASCII
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    steps.append("1. Nettoyage (URLs, @mentions, #hashtags, non-ASCII)")

    # Step 2 — Lowercase
    text = text.lower()
    steps.append("2. Lowercase")

    # Step 3 — Sentence tokenisation
    sentences = sent_tokenize(text)
    steps.append(f"3. Segmentation en phrases ({len(sentences)} phrase(s))")

    # Step 4 — Word tokenisation
    tokens = [tok for sent in sentences for tok in word_tokenize(sent)]
    steps.append(f"4. Tokenisation ({len(tokens)} tokens)")

    # Step 5 — Remove punctuation
    tokens = [t for t in tokens if t not in _PONCTUATION and not all(c in _PONCTUATION for c in t)]
    steps.append(f"5. Suppression ponctuation ({len(tokens)} tokens restants)")

    # Step 6 — Remove stopwords (preserve negations)
    tokens = [t for t in tokens if t not in stop_words]
    steps.append(f"6. Suppression stopwords, négations préservées ({len(tokens)} tokens)")

    # Step 7 — Lemmatisation
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    steps.append("7. Lemmatisation")

    return " ".join(tokens), steps
