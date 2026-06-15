# 🚀 Sentiment Analysis — FastAPI Interface

Interface web et API REST pour le pipeline NLP d'analyse de sentiments.  
Projet TALN · 4ème année Ingénierie IA · UMMTO 2025/2026 · Dr. LAZIB Lydia

---

## 📁 Structure

```
sentiment-api/
├── main.py                         # Point d'entrée Uvicorn
├── export_models.py                # Script d'export des artefacts du notebook
├── requirements.txt
│
├── app/
│   ├── main.py                     # Application FastAPI (routes, lifespan)
│   ├── model_manager.py            # Chargement et inférence des 4 modèles
│   ├── preprocessing.py            # Pipeline NLP 7 étapes (identique au notebook)
│   └── schemas.py                  # Schémas Pydantic (requêtes / réponses)
│
├── models/                         # Artefacts ML (générés par export_models.py)
│   ├── lr_tfidf_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── lr_w2v_model.pkl
│   ├── keras_tokenizer.pkl
│   ├── best_bilstm.keras
│   └── distilbert_finetuned_best_v2/
│
├── templates/
│   └── index.html                  # Interface web (dark UI)
│
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## ⚡ Démarrage rapide

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Exporter les modèles depuis le notebook

Assurez-vous d'abord d'avoir exécuté le notebook complet, puis depuis le **même dossier** :

```bash
# Depuis le dossier contenant le notebook :
cp -r sentiment-api/ .
python sentiment-api/export_models.py
```

> Le script copie automatiquement `lr_tfidf_model.pkl`, `best_bilstm.keras`, `distilbert_finetuned_best_v2/` etc. vers `sentiment-api/models/`.

⚠️ **Ajouts nécessaires dans le notebook** (2 lignes) :

```python
# Après tfidf_vec.fit_transform(...) :
with open('tfidf_vectorizer.pkl', 'wb') as f: pickle.dump(tfidf_vec, f)

# Après keras_tokenizer.fit_on_texts(...) :
with open('keras_tokenizer.pkl', 'wb') as f: pickle.dump(keras_tokenizer, f)

# Après clf_w2v = LogisticRegression(...).fit(...) :
with open('lr_w2v_model.pkl', 'wb') as f: pickle.dump(clf_w2v_best, f)
```

### 3. Lancer le serveur

```bash
cd sentiment-api
python main.py
# ou
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Accédez à **http://localhost:8000**

---

## 🌐 Endpoints API

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/` | Interface web |
| `GET` | `/health` | Statut de l'API et modèles chargés |
| `GET` | `/models` | Informations sur les 4 modèles |
| `POST` | `/predict` | Prédiction simple |
| `POST` | `/predict/batch` | Prédiction sur un lot (max 50 textes) |
| `POST` | `/compare` | Comparaison simultanée des 4 modèles |
| `GET` | `/docs` | Documentation Swagger interactive |
| `GET` | `/redoc` | Documentation ReDoc |

### Exemple — `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely loved this movie!", "model": "lr_tfidf"}'
```

```json
{
  "text": "I absolutely loved this movie!",
  "model": "Logistic Regression + TF-IDF",
  "label": "positive",
  "confidence": 0.9312,
  "latency_ms": 4.2,
  "preprocessing_steps": [
    "1. Nettoyage (URLs, @mentions, #hashtags, non-ASCII)",
    "2. Lowercase",
    "3. Segmentation en phrases (1 phrase(s))",
    "4. Tokenisation (5 tokens)",
    "5. Suppression ponctuation (5 tokens restants)",
    "6. Suppression stopwords, négations préservées (3 tokens)",
    "7. Lemmatisation"
  ]
}
```

### Modèles disponibles (`model` field)

| Clé | Modèle | F1 |
|-----|--------|-----|
| `lr_tfidf` | Logistic Regression + TF-IDF | ~82% |
| `lr_word2vec` | Logistic Regression + Word2Vec | ~81% |
| `bilstm` | BiLSTM bidirectionnel | ~84% |
| `distilbert` | DistilBERT Fine-Tuned | ~91% |

---

## 🔄 Mode dégradé

Si les fichiers modèles sont absents, l'API fonctionne en **mode mock** (règles lexicales simples) pour permettre le développement de l'interface sans GPU.

---

## 🎓 Contexte académique

Module **Traitement Automatique du Langage Naturel (TALN)**  
4ème année Ingénierie Intelligence Artificielle — **UMMTO**  
Année universitaire 2025/2026 — **Dr. LAZIB Lydia**
