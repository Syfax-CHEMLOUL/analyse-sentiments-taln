"""
export_models.py — À exécuter APRÈS le notebook pour exporter les artefacts nécessaires à l'API.

Usage (dans le même dossier que le notebook) :
    python export_models.py

Artefacts produits dans ./models/ :
  ├── lr_tfidf_model.pkl        ← Logistic Regression (TF-IDF)
  ├── tfidf_vectorizer.pkl      ← TfidfVectorizer fitted
  ├── lr_w2v_model.pkl          ← Logistic Regression (Word2Vec) [si disponible]
  ├── keras_tokenizer.pkl       ← KerasTokenizer fitted
  ├── best_bilstm.keras         ← Copié depuis le checkpoint
  └── distilbert_finetuned_best_v2/  ← Fine-tuned DistilBERT
"""

import os
import shutil
import pickle
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  Export des artefacts modèles → ./models/")
print("=" * 60)

# ── 1. LR + TF-IDF ───────────────────────────────────────────
# Le notebook sauvegarde déjà lr_tfidf_model.pkl — on copie + on ajoute le vectorizer
if Path("lr_tfidf_model.pkl").exists():
    shutil.copy("lr_tfidf_model.pkl", MODELS_DIR / "lr_tfidf_model.pkl")
    print("✅  lr_tfidf_model.pkl copié")
else:
    print("⚠️   lr_tfidf_model.pkl introuvable — relancez la section 8 du notebook")

# Pour le vectorizer, ajoutez ceci à la fin de la cellule TF-IDF dans le notebook :
# with open('tfidf_vectorizer.pkl', 'wb') as f: pickle.dump(tfidf_vec, f)
if Path("tfidf_vectorizer.pkl").exists():
    shutil.copy("tfidf_vectorizer.pkl", MODELS_DIR / "tfidf_vectorizer.pkl")
    print("✅  tfidf_vectorizer.pkl copié")
else:
    print("⚠️   tfidf_vectorizer.pkl introuvable.")
    print("     Ajoutez dans le notebook (section TF-IDF) :")
    print("       import pickle")
    print("       with open('tfidf_vectorizer.pkl','wb') as f: pickle.dump(tfidf_vec, f)")

# ── 2. LR + Word2Vec ─────────────────────────────────────────
if Path("lr_w2v_model.pkl").exists():
    shutil.copy("lr_w2v_model.pkl", MODELS_DIR / "lr_w2v_model.pkl")
    print("✅  lr_w2v_model.pkl copié")
else:
    print("⚠️   lr_w2v_model.pkl introuvable.")
    print("     Ajoutez dans le notebook (section Word2Vec) :")
    print("       with open('lr_w2v_model.pkl','wb') as f: pickle.dump(clf_w2v_best, f)")

# ── 3. BiLSTM ────────────────────────────────────────────────
bilstm_src = Path("best_bilstm.keras")
if bilstm_src.exists():
    shutil.copy(bilstm_src, MODELS_DIR / "best_bilstm.keras")
    print("✅  best_bilstm.keras copié")
else:
    print("⚠️   best_bilstm.keras introuvable — relancez la section 9 du notebook")

# Keras tokenizer — ajoutez dans le notebook après le fit :
# with open('keras_tokenizer.pkl','wb') as f: pickle.dump(keras_tokenizer, f)
if Path("keras_tokenizer.pkl").exists():
    shutil.copy("keras_tokenizer.pkl", MODELS_DIR / "keras_tokenizer.pkl")
    print("✅  keras_tokenizer.pkl copié")
else:
    print("⚠️   keras_tokenizer.pkl introuvable.")
    print("     Ajoutez dans le notebook (après keras_tokenizer.fit_on_texts) :")
    print("       with open('keras_tokenizer.pkl','wb') as f: pickle.dump(keras_tokenizer, f)")

# ── 4. DistilBERT ────────────────────────────────────────────
distilbert_src = Path("distilbert_finetuned_best_v2")
if distilbert_src.exists() and distilbert_src.is_dir():
    dest = MODELS_DIR / "distilbert_finetuned_best_v2"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(distilbert_src, dest)
    print("✅  distilbert_finetuned_best_v2/ copié")
else:
    print("⚠️   distilbert_finetuned_best_v2/ introuvable — relancez la section E1 du notebook")

print()
print("=" * 60)
print("  Structure finale :")
for p in sorted(MODELS_DIR.rglob("*"))[:20]:
    indent = "  " + "  " * (len(p.parts) - len(MODELS_DIR.parts) - 1)
    print(f"{indent}{'📁' if p.is_dir() else '📄'} {p.name}")
print("=" * 60)
print("  Lancez ensuite : python main.py")
