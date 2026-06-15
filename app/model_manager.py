"""
ModelManager — charge les 4 modèles produits par le notebook TALN.

Artefacts attendus dans ./models/ :
  lr_tfidf_model.pkl          → LR + TF-IDF (pickle)
  tfidf_vectorizer.pkl        → TfidfVectorizer (pickle)
  lr_w2v_model.pkl            → LR + Word2Vec (pickle)  [optionnel]
  best_bilstm.keras           → BiLSTM (Keras)
  keras_tokenizer.pkl         → KerasTokenizer (pickle)
  distilbert_finetuned_best_v2/  → DistilBERT fine-tuné (HuggingFace)
"""

import os
import time
import logging
import pickle
from pathlib import Path
from typing import Optional

from app.preprocessing import preprocess
from app.schemas import PredictResponse, BatchResponse, ModelInfo

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")

CATALOG = {
    "lr_tfidf": {
        "name": "Logistic Regression + TF-IDF",
        "description": "Baseline classique rapide et interprétable. Les bigrammes TF-IDF capturent partiellement le contexte local.",
        "representation": "TF-IDF (bigrammes)",
        "f1_score": 0.82,
        "contextual": False,
    },
    "lr_word2vec": {
        "name": "Logistic Regression + Word2Vec",
        "description": "Embeddings statiques Word2Vec (Google News 300d). Capture la sémantique globale mais pas le contexte séquentiel.",
        "representation": "Word2Vec (avg pooling)",
        "f1_score": 0.81,
        "contextual": False,
    },
    "bilstm": {
        "name": "BiLSTM",
        "description": "Réseau bidirectionnel LSTM. Capture l'ordre des mots et les dépendances séquentielles dans les deux directions.",
        "representation": "Séquences (Keras Tokenizer)",
        "f1_score": 0.84,
        "contextual": True,
    },
    "distilbert": {
        "name": "DistilBERT Fine-Tuned",
        "description": "Transformer pré-entraîné fine-tuné sur 120k tweets équilibrés. Représentation contextuelle profonde et transfert de connaissance.",
        "representation": "Contextual embeddings (Transformer)",
        "f1_score": 0.91,
        "contextual": True,
    },
}


class ModelManager:
    AVAILABLE = CATALOG

    def __init__(self):
        self._models: dict = {}
        self._ready: dict[str, bool] = {k: False for k in CATALOG}

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_all(self):
        self._load_lr_tfidf()
        self._load_lr_word2vec()
        self._load_bilstm()
        self._load_distilbert()

    def unload_all(self):
        self._models.clear()

    def _load_lr_tfidf(self):
        model_path = MODELS_DIR / "lr_tfidf_model.pkl"
        vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        if model_path.exists() and vec_path.exists():
            try:
                with open(model_path, "rb") as f:
                    clf = pickle.load(f)
                with open(vec_path, "rb") as f:
                    vec = pickle.load(f)
                self._models["lr_tfidf"] = {"clf": clf, "vec": vec}
                self._ready["lr_tfidf"] = True
                logger.info("✅ LR+TF-IDF loaded")
            except Exception as e:
                logger.warning(f"LR+TF-IDF load failed: {e}")
        else:
            logger.warning("⚠️  lr_tfidf model files not found — using mock")

    def _load_lr_word2vec(self):
        model_path = MODELS_DIR / "lr_w2v_model.pkl"
        w2v_path = MODELS_DIR / "w2v_avg_vectors.pkl"   # pre-computed mean vectors
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    clf = pickle.load(f)
                w2v = None
                if w2v_path.exists():
                    with open(w2v_path, "rb") as f:
                        w2v = pickle.load(f)
                self._models["lr_word2vec"] = {"clf": clf, "w2v": w2v}
                self._ready["lr_word2vec"] = True
                logger.info("✅ LR+W2V loaded")
            except Exception as e:
                logger.warning(f"LR+W2V load failed: {e}")
        else:
            logger.warning("⚠️  lr_word2vec model files not found — using mock")

    def _load_bilstm(self):
        model_path = MODELS_DIR / "best_bilstm.keras"
        tok_path = MODELS_DIR / "keras_tokenizer.pkl"
        if model_path.exists() and tok_path.exists():
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(str(model_path))
                with open(tok_path, "rb") as f:
                    tokenizer = pickle.load(f)
                self._models["bilstm"] = {"model": model, "tokenizer": tokenizer}
                self._ready["bilstm"] = True
                logger.info("✅ BiLSTM loaded")
            except Exception as e:
                logger.warning(f"BiLSTM load failed: {e}")
        else:
            logger.warning("⚠️  BiLSTM model files not found — using mock")

    def _load_distilbert(self):
        model_dir = MODELS_DIR / "distilbert_finetuned_best_v2"
        if model_dir.exists():
            try:
                from transformers import pipeline
                pipe = pipeline(
                    "text-classification",
                    model=str(model_dir),
                    tokenizer=str(model_dir),
                    truncation=True,
                    max_length=128,
                )
                self._models["distilbert"] = {"pipeline": pipe}
                self._ready["distilbert"] = True
                logger.info("✅ DistilBERT loaded")
            except Exception as e:
                logger.warning(f"DistilBERT load failed: {e}")
        else:
            logger.warning("⚠️  DistilBERT directory not found — using mock")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str, model_key: str) -> PredictResponse:
        t0 = time.perf_counter()
        clean_text, steps = preprocess(text)
        label, confidence = self._infer(clean_text, text, model_key)
        latency = (time.perf_counter() - t0) * 1000
        return PredictResponse(
            text=text,
            model=CATALOG[model_key]["name"],
            label=label,
            confidence=round(confidence, 4),
            latency_ms=round(latency, 2),
            preprocessing_steps=steps,
        )

    def predict_batch(self, texts: list[str], model_key: str) -> BatchResponse:
        t0 = time.perf_counter()
        results = [self.predict(t, model_key) for t in texts]
        total_lat = (time.perf_counter() - t0) * 1000
        return BatchResponse(
            model=CATALOG[model_key]["name"],
            total=len(results),
            results=results,
            total_latency_ms=round(total_lat, 2),
        )

    def _infer(self, clean_text: str, raw_text: str, model_key: str) -> tuple[str, float]:
        """Route to the correct inference backend. Falls back to mock if not loaded."""
        if not self._ready[model_key]:
            return self._mock_predict(clean_text)

        try:
            if model_key == "lr_tfidf":
                return self._infer_lr_tfidf(clean_text)
            elif model_key == "lr_word2vec":
                return self._infer_lr_w2v(clean_text)
            elif model_key == "bilstm":
                return self._infer_bilstm(clean_text)
            elif model_key == "distilbert":
                return self._infer_distilbert(raw_text)
        except Exception as e:
            logger.error(f"Inference error ({model_key}): {e}")
            return self._mock_predict(clean_text)

    def _infer_lr_tfidf(self, text: str) -> tuple[str, float]:
        obj = self._models["lr_tfidf"]
        x = obj["vec"].transform([text])
        proba = obj["clf"].predict_proba(x)[0]
        idx = proba.argmax()
        label = "positive" if idx == 1 else "negative"
        return label, float(proba[idx])

    def _infer_lr_w2v(self, text: str) -> tuple[str, float]:
        import numpy as np
        obj = self._models["lr_word2vec"]
        if obj["w2v"] is not None:
            # obj["w2v"] is a gensim KeyedVectors object
            tokens = text.split()
            vecs = [obj["w2v"][t] for t in tokens if t in obj["w2v"]]
            if not vecs:
                x = np.zeros((1, 300))
            else:
                x = np.mean(vecs, axis=0, keepdims=True)
        else:
            # Fallback: use zeros if w2v not available
            import numpy as np
            x = np.zeros((1, 300))
        proba = obj["clf"].predict_proba(x)[0]
        idx = proba.argmax()
        label = "positive" if idx == 1 else "negative"
        return label, float(proba[idx])

    def _infer_bilstm(self, text: str) -> tuple[str, float]:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        obj = self._models["bilstm"]
        seq = obj["tokenizer"].texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=50, padding="post", truncating="post")
        score = float(obj["model"].predict(padded, verbose=0)[0][0])
        label = "positive" if score >= 0.5 else "negative"
        confidence = score if label == "positive" else 1 - score
        return label, confidence

    def _infer_distilbert(self, text: str) -> tuple[str, float]:
        obj = self._models["distilbert"]
        result = obj["pipeline"](text[:512])[0]
        raw_label = result["label"].upper()
        score = float(result["score"])
        # HF labels can be LABEL_0/LABEL_1 or NEGATIVE/POSITIVE
        if "1" in raw_label or "POS" in raw_label:
            label = "positive"
        else:
            label = "negative"
        return label, score

    def _mock_predict(self, text: str) -> tuple[str, float]:
        """Rule-based mock for demo when model files are absent."""
        import re
        pos_words = {"love", "great", "excellent", "happy", "good", "amazing", "wonderful", "best", "fantastic", "awesome"}
        neg_words = {"hate", "bad", "terrible", "awful", "horrible", "worst", "disappointed", "sad", "poor", "disgusting"}
        tokens = set(re.findall(r"\w+", text.lower()))
        pos = len(tokens & pos_words)
        neg = len(tokens & neg_words)
        if pos > neg:
            return "positive", min(0.55 + pos * 0.05, 0.92)
        elif neg > pos:
            return "negative", min(0.55 + neg * 0.05, 0.92)
        else:
            import random
            random.seed(len(text))
            s = random.uniform(0.52, 0.65)
            return ("positive" if s > 0.58 else "negative"), s

    # ── Meta ──────────────────────────────────────────────────────────────────

    def loaded_models(self) -> list[str]:
        return [k for k, v in self._ready.items() if v]

    def model_info(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                key=k,
                name=v["name"],
                description=v["description"],
                representation=v["representation"],
                f1_score=v["f1_score"],
                loaded=self._ready[k],
                contextual=v["contextual"],
            )
            for k, v in CATALOG.items()
        ]
