# Bibliothèque RAG 100 % locale

Interrogez vos livres et documents (PDF, DOCX, EPUB, HTML, TXT, MD ; français ou anglais) avec un modèle de langage qui tourne dans **LM Studio**, sans qu'aucune donnée ne quitte votre machine.

```
 Navigateur ──► Application (Gradio) ──► LM Studio (hôte)  : embeddings + modèle de chat
                 │         │
                 │         └──► Qdrant (conteneur)         : vecteurs des passages
                 └────────────► SQLite (data/rag.sqlite)    : texte, sections, pages, index BM25 (FTS5)
```

**Ce que fait la chaîne**

1. **Extraction** : `pymupdf4llm` pour les PDF (titres, tableaux, numéros de page, OCR Tesseract fra+eng des pages scannées), `python-docx`, `ebooklib`, `BeautifulSoup`.
2. **Découpage structurel parent/enfant** : le texte est découpé selon les titres ; de petits passages (~1 500 caractères, ~400 tokens) sont indexés, et le LLM reçoit la section parente (≤ 4 000 caractères) pour avoir du contexte.
3. **Recherche hybride** : vecteurs (Qdrant, similarité cosinus) + BM25 (SQLite FTS5, insensible aux accents), fusionnés par *Reciprocal Rank Fusion*, puis reranking optionnel par cross-encoder.
4. **Génération** : le modèle ne répond qu'à partir des extraits numérotés, cite `[1]`, `[2]`… et dit quand l'information est absente. Les sources (livre, section, page, extrait) s'affichent à côté de la réponse.
5. **Espaces** : chaque document appartient à un espace (thème). On interroge toute la bibliothèque ou seulement certains espaces ; le filtre s'applique aux deux branches de la recherche.

Le texte reste dans SQLite : changer de modèle d'embedding ne demande **pas** de ré-extraire les livres, seulement de *réindexer* (un bouton).

---

## Démarrage rapide

### 1. LM Studio (sur votre machine)

- Téléchargez un **modèle d'embedding multilingue** (par exemple un GGUF de BGE-M3) et un **modèle de chat** (les familles Mistral Small, Qwen3 ou Gemma 3 fonctionnent bien en français ; choisissez selon votre mémoire).
- Onglet **Developer** : démarrez le serveur local (port `1234`) et chargez les deux modèles (ou laissez LM Studio les charger à la demande).
- Réglez la **longueur de contexte** du modèle de chat à **8 192 tokens au minimum** (16 000 si possible). Trop court, le contexte est tronqué sans message d'erreur.

### 2. Lancer la stack

```bash
cd rag-stack
cp .env.example .env        # facultatif
docker compose up -d --build
```

Ouvrez **http://localhost:7860**.

### 3. Premier réglage (2 minutes)

1. Onglet **⚙️ Réglages** → **Tester la connexion et lister les modèles**.
2. Choisissez le *modèle de langage* et le *modèle d'embedding*, puis **Enregistrer**.
3. Onglet **📚 Bibliothèque** → déposez quelques PDF, choisissez un espace (ex. « IA & gouvernance »), **Indexer**.
4. Onglet **💬 Interroger** → posez une question.

Pour 50 livres, préférez le dépôt en lot : copiez-les dans `./inbox/<Espace>/…` (un sous-dossier = un espace) puis cliquez sur **Indexer le dossier inbox** (ou `docker compose exec app python -m app.cli inbox`).

---

## Utilisation

**Interroger.** Cochez des espaces pour restreindre la recherche (aucun coché = tout). Le curseur « Nombre de passages » règle combien d'extraits sont fournis au modèle. Les questions de suivi (« et chez l'autre auteur ? ») sont reformulées automatiquement avant la recherche.

**Bibliothèque.** Nommez vos fichiers `Auteur_Titre_Année.pdf` : le nom sert de titre dans les citations. Un fichier identique est ignoré ; un fichier modifié (même nom) remplace l'ancienne version ; un fichier réimporté dans un autre espace est simplement déplacé. L'accordéon *Gérer un document* permet de déplacer ou supprimer, *Maintenance* de réindexer.

**Réglages.** Modèles, température, taille du contexte, reranker, consignes données au modèle (modifiables). Les préfixes d'embedding (« query: » / « passage: ») ne servent qu'aux modèles qui l'exigent comme multilingual-e5 ; ils sont vides pour BGE-M3.

### Ligne de commande

```bash
docker compose exec app python -m app.cli stats
docker compose exec app python -m app.cli ingest /inbox -c "Statistiques"
docker compose exec app python -m app.cli ask "Qu'est-ce que la recherche hybride ?" -c IA
docker compose exec app python -m app.cli reindex
docker compose exec app python -m app.cli eval /data/eval.csv
```

---

## Mesurer la qualité (à faire dès le début)

Sans mesure, on règle à l'aveugle. Préparez 30 à 50 questions dont vous connaissez le livre (et idéalement la page) de réponse, dans `data/eval.csv` (modèle : `eval.example.csv`) :

```csv
question,expected_doc,expected_page,collection
"Comment est défini le biais de sélection ?",Statistiques_Manuel,214,Statistiques
```

`python -m app.cli eval /data/eval.csv` affiche recall@1/3/5/10, le MRR et la liste des échecs. Changez **un seul paramètre à la fois** (taille des passages, reranker, nombre de passages, modèle d'embedding) et comparez.

Pistes d'amélioration, par ordre de rendement habituel : qualité de l'extraction (vérifiez `data/extracted/*.md`), activation du reranker, ajustement de `CHILD_CHARS`/`PARENT_CHARS`, choix du modèle d'embedding.

---

## Reranker (optionnel)

Il améliore souvent nettement la précision, au prix de quelques secondes par question. Il tourne **sur CPU dans le conteneur** (LM Studio n'expose pas de reranking).

```bash
echo "WITH_RERANKER=1" >> .env
docker compose up -d --build          # l'image grossit (PyTorch CPU)
```

Puis cochez « Reranker » dans Réglages. Au premier usage, le modèle `BAAI/bge-reranker-v2-m3` est téléchargé depuis Hugging Face dans `data/hf`. En environnement sans Internet : téléchargez-le sur une machine connectée, copiez le dossier dans `data/hf`, ou pointez `RERANKER_MODEL` vers un chemin local.

---

## Données, sauvegarde, mise à jour

Tout l'état est dans `./data` : `rag.sqlite` (texte + index lexical), `qdrant/` (vecteurs), `extracted/` (texte extrait en Markdown), `library/` (copies des fichiers téléversés), `settings.json`.

- **Sauvegarde** : arrêtez la stack (`docker compose stop`) et copiez `./data`. `rag.sqlite` suffit à tout reconstruire : si `qdrant/` est perdu, `reindex` régénère les vecteurs.
- **Mise à jour du code** : `docker compose up -d --build`.
- **Changer de modèle d'embedding** : choisissez-le dans Réglages, puis *Bibliothèque → Réindexer tout*. Tant que ce n'est pas fait, les questions sont refusées avec un message explicite (mélanger deux modèles donnerait des résultats faux).

## Dépannage

| Symptôme | Piste |
|---|---|
| « Impossible de joindre LM Studio » | Le serveur local est-il démarré (onglet Developer) ? Sous Linux, un serveur lié à `127.0.0.1` n'est pas joignable depuis un conteneur : activez, à ma connaissance, l'option de service sur le réseau local de LM Studio. Vous pouvez aussi lancer l'application hors Docker (voir plus bas). |
| Liste de modèles vide | Aucun modèle téléchargé, ou serveur LM Studio sans modèle chargé. |
| Réponses hors sujet ou tronquées | Longueur de contexte trop courte dans LM Studio ; réduisez « Contexte max » dans Réglages. |
| PDF « OCR requis » | Scan illisible ou sans Tesseract. L'image Docker l'inclut ; hors Docker, installez `tesseract-ocr` + les langues, ou passez le fichier à `ocrmypdf`. |
| Résultats lexicaux pauvres en français | L'index BM25 ne fait pas de lemmatisation (« photosynthèse » ≠ « photosynthèses »). La branche vectorielle compense ; le reranker aide. |
| Indexation lente | Le goulot est l'embedding dans LM Studio. Les PDF volumineux et l'OCR prennent plusieurs minutes. |

### Sans Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export QDRANT_PATH=./data/qdrant                       # Qdrant embarqué, sans serveur
export LMSTUDIO_BASE_URL=http://localhost:1234/v1
python -m app.ui                                        # http://localhost:7860
```

(Ou lancez `docker run -p 6333:6333 qdrant/qdrant` et exportez `QDRANT_URL=http://localhost:6333`.)

## Sécurité

L'interface n'écoute que sur `127.0.0.1` par défaut. Pour l'ouvrir au réseau local, modifiez le mapping de port dans `docker-compose.yml` **et** définissez `APP_USER`/`APP_PASSWORD`. L'application est mono-équipe : il n'y a pas de droits d'accès par document ni par utilisateur ; si vous avez un corpus à accès restreint, séparez-le en deux déploiements.

## Limites connues

- Les questions de **synthèse globale** (« quels thèmes traversent l'ensemble des livres ? ») sont mal servies par un RAG classique, qui ne ramène que quelques passages. Pistes : résumés hiérarchiques par chapitre, ou GraphRAG.
- Les **tableaux chiffrés** extraits de PDF sont lisibles par le modèle mais restent fragiles ; pour des calculs, passez par des données structurées.
- Pas de gestion de versions d'un même livre autre que le remplacement par nom de fichier.

## Structure du code

| Fichier | Rôle |
|---|---|
| `app/ui.py` | Interface Gradio (3 onglets) |
| `app/ingest.py` | Ingestion, réindexation, dossier inbox |
| `app/extract.py`, `app/chunk.py` | Extraction (PDF/DOCX/EPUB/HTML/TXT) et découpage parent/enfant |
| `app/retrieve.py` | Recherche vectorielle + BM25 + RRF + reranker |
| `app/answer.py` | Prompt, reformulation des questions de suivi, sources |
| `app/llm.py`, `app/store.py`, `app/db.py` | Client LM Studio, Qdrant, SQLite/FTS5 |
| `app/evaluate.py`, `app/cli.py` | Évaluation recall@k, ligne de commande |
| `tests/` | Tests de bout en bout avec un faux serveur LM Studio |

## Tests

```bash
pip install pytest && pytest tests -q
```

Les tests utilisent un faux serveur LM Studio (embeddings « sac de mots » déterministes) et Qdrant en mémoire : ils vérifient la chaîne complète (extraction des 5 formats, ingestion, recherche hybride, filtres par espace, remplacement/suppression, réindexation, détection d'un changement de modèle, réponse en streaming, évaluation). Ils ne remplacent pas un essai avec vos vrais modèles.
