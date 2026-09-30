# Auxiliary IR experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # Notebook 1 — Construction de la collection *MS MARCO answer-passage* (version corrigée)
#
# ## Objectif
# On reproduit la **logique expérimentale** du papier, mais avec **MS MARCO** à la place de **GOV2/TREC** :
#
# 1. charger MS MARCO Document et MS MARCO Passage ;
# 2. sélectionner aléatoirement des requêtes admissibles ;
# 3. faire une récupération initiale **SDM top-50** sur la **vraie collection** ;
# 4. identifier les documents positifs à l'aide des **qrels document** ;
# 5. aligner des **passages positifs** dans les documents pour obtenir des **spans silver** ;
# 6. découper les documents en fenêtres fixes **50 / overlap 25** ;
# 7. labelliser les fenêtres selon leur chevauchement avec ces spans ;
# 8. exporter une collection propre pour le notebook de modélisation.
#
# ## Corrections intégrées
# - **Pas de pool enrichi en documents positifs** dans le protocole principal.
# - **Pas de sélection des requêtes les plus faciles** : tirage aléatoire avec `seed`.
# - Les labels de fenêtres sont des **silver labels** (pas des annotations humaines).
# - On prépare des **spans** dans les documents pour permettre ensuite une évaluation plus proche du papier.

# %%

# =====================================================================
# 1) Installation des dépendances
# =====================================================================
# Décommente ces lignes si nécessaire dans Colab / Jupyter.
# NOTEBOOK: !pip -q install python-terrier ir_datasets pandas numpy tqdm

# %%

# =====================================================================
# 2) Imports et configuration générale
# =====================================================================
import os
import re
import json
import math
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import ir_datasets
import pyterrier as pt

if not pt.started():
    pt.init()

# -------------------------------
# Seed et paramètres principaux
# -------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Nombre de requêtes finales (82 pour coller au papier ; 100 si vous voulez plus)
NUM_FINAL_QUERIES = 82

# Récupération initiale
TOPK_DOCS = 50

# Fenêtrage des documents (comme dans le papier)
WINDOW_SIZE = 50
WINDOW_OVERLAP = 25

# Limitation pratique pour éviter qu'une requête avec trop de passages positifs explose le temps
MAX_POS_PASSAGES_PER_QID = 20

# Limitation pratique sur le nombre de documents positifs traités par requête
MAX_POS_DOCS_PER_QID = None   # mettre par ex. 10 si besoin de réduire le coût

# Dossiers de travail
WORKDIR = Path.cwd()
ARTIFACT_DIR = WORKDIR / "artifacts_notebook1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Index PyTerrier complet de MS MARCO document
INDEX_PATH = ARTIFACT_DIR / "pt_msmarco_doc_index"

# Fichiers export
PATH_QUERIES = ARTIFACT_DIR / "queries_final.tsv"
PATH_TOPDOCS = ARTIFACT_DIR / "topdocs_top50.tsv"
PATH_SPANS = ARTIFACT_DIR / "silver_spans.tsv"
PATH_WINDOWS = ARTIFACT_DIR / "windows.tsv"
PATH_WINDOW_LABELS = ARTIFACT_DIR / "silver_window_labels.tsv"
PATH_METADATA = ARTIFACT_DIR / "dataset_metadata.json"

print("Dossier de sortie :", ARTIFACT_DIR)

# %%

# =====================================================================
# 3) Helpers texte / tokens / spans
# =====================================================================
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

def norm_qid(x):
    """Normalise les IDs de requêtes pour éviter les '123.0' vs '123'."""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def tok(text: str):
    """Tokenisation simple et reproductible."""
    if text is None:
        return []
    return TOKEN_PATTERN.findall(str(text).lower())

def clean_query(text: str) -> str:
    """Nettoyage léger de la requête pour PyTerrier."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_doc_text(doc_obj) -> str:
    """Concatène titre + corps du document MS MARCO Document."""
    title = getattr(doc_obj, "title", "") or ""
    body = getattr(doc_obj, "body", "") or ""
    return f"{title} {body}".strip()

def make_windows(tokens, win=50, overlap=25):
    """Construit des fenêtres fixes 50/25 sur une séquence de tokens."""
    step = max(1, win - overlap)
    out = []
    for start in range(0, len(tokens), step):
        end = min(start + win, len(tokens))
        chunk = tokens[start:end]
        if not chunk:
            break
        out.append((start, end, chunk))
        if end >= len(tokens):
            break
    return out

def multiset_overlap(a, b):
    """Intersection multiset (compte les répétitions de tokens)."""
    ca, cb = Counter(a), Counter(b)
    return sum(min(ca[t], cb[t]) for t in (ca.keys() & cb.keys()))

def bigrams(tokens):
    if len(tokens) < 2:
        return set()
    return set(zip(tokens[:-1], tokens[1:]))

def exact_sublist_span(doc_tokens, passage_tokens):
    """Cherche un match exact de passage_tokens dans doc_tokens.
    Retourne (start, end) ou None.
    """
    m = len(passage_tokens)
    n = len(doc_tokens)
    if m == 0 or m > n:
        return None
    first = passage_tokens[0]
    candidate_starts = [i for i, t in enumerate(doc_tokens) if t == first]
    for s in candidate_starts:
        if s + m <= n and doc_tokens[s:s+m] == passage_tokens:
            return (s, s + m)
    return None

def fuzzy_align_span(doc_tokens, passage_tokens, min_ratio=0.45):
    """Aligne approximativement un passage dans un document.

    Stratégie :
    1) essai exact ;
    2) sinon balayage de fenêtres de tailles proches de la longueur du passage ;
    3) score = combinaison F1 lexical + rappel de bigrams.

    Retour :
        dict avec start, end, score, recall, precision, bigram_recall
        ou None si aucun alignement raisonnable.
    """
    # 1) Match exact
    exact = exact_sublist_span(doc_tokens, passage_tokens)
    if exact is not None:
        s, e = exact
        return {
            "start_token": s,
            "end_token": e,
            "align_score": 1.0,
            "lex_recall": 1.0,
            "lex_precision": 1.0,
            "bigram_recall": 1.0,
            "align_type": "exact"
        }

    # 2) Match approximatif
    m = len(passage_tokens)
    n = len(doc_tokens)
    if m == 0 or n == 0:
        return None

    # Tailles de fenêtres candidates autour de la longueur du passage
    min_len = max(5, int(m * 0.7))
    max_len = min(n, int(m * 1.3) + 3)

    p_bi = bigrams(passage_tokens)
    best = None

    # Pour rester raisonnable en coût, on sous-échantillonne les tailles si nécessaire
    candidate_lengths = list(range(min_len, max_len + 1))
    if len(candidate_lengths) > 12:
        idx = np.linspace(0, len(candidate_lengths) - 1, 12, dtype=int)
        candidate_lengths = [candidate_lengths[i] for i in idx]

    for L in candidate_lengths:
        for s in range(0, n - L + 1):
            window = doc_tokens[s:s+L]

            inter = multiset_overlap(window, passage_tokens)
            rec = inter / max(1, len(passage_tokens))
            prec = inter / max(1, len(window))
            f1 = 0.0 if (rec + prec) == 0 else (2 * rec * prec) / (rec + prec)

            w_bi = bigrams(window)
            bi_rec = len(w_bi & p_bi) / max(1, len(p_bi))

            score = 0.75 * f1 + 0.25 * bi_rec

            if best is None or score > best["align_score"]:
                best = {
                    "start_token": s,
                    "end_token": s + L,
                    "align_score": float(score),
                    "lex_recall": float(rec),
                    "lex_precision": float(prec),
                    "bigram_recall": float(bi_rec),
                    "align_type": "fuzzy"
                }

    if best is None or best["align_score"] < min_ratio:
        return None
    return best

def label_window_from_span(window_start, window_end, span_start, span_end):
    """Donne un label gradué à une fenêtre selon son chevauchement avec un span.

    Labels :
      0 = non pertinent
      1 = acceptable
      2 = good
      3 = excellent
      4 = perfect
    """
    inter = max(0, min(window_end, span_end) - max(window_start, span_start))
    if inter <= 0:
        return {
            "label": 0,
            "overlap_tokens": 0,
            "span_recall": 0.0,
            "window_precision": 0.0
        }

    span_len = max(1, span_end - span_start)
    win_len = max(1, window_end - window_start)

    recall = inter / span_len
    precision = inter / win_len

    # Règles simples et transparentes
    if recall >= 0.90 and precision >= 0.70:
        label = 4   # perfect
    elif recall >= 0.75 and precision >= 0.50:
        label = 3   # excellent
    elif recall >= 0.50:
        label = 2   # good
    else:
        label = 1   # acceptable

    return {
        "label": label,
        "overlap_tokens": int(inter),
        "span_recall": float(recall),
        "window_precision": float(precision)
    }

# %%

# =====================================================================
# 4) Chargement des datasets MS MARCO
# =====================================================================
# Ici on utilise directement les collections officielles :
# - msmarco-document/train
# - msmarco-passage/train
#
# Cela remplace GOV2 + TREC dans votre adaptation.

doc_ds = ir_datasets.load("msmarco-document/train")
pass_ds = ir_datasets.load("msmarco-passage/train")

doc_store = doc_ds.docs_store()
pass_store = pass_ds.docs_store()

# Requêtes document
queries = {str(q.query_id): q.text for q in doc_ds.queries_iter()}

# Qrels document
doc_pos = defaultdict(set)
for qr in tqdm(doc_ds.qrels_iter(), desc="Chargement qrels doc"):
    if qr.relevance > 0:
        doc_pos[str(qr.query_id)].add(str(qr.doc_id))

# Qrels passage
pass_pos = defaultdict(set)
for qr in tqdm(pass_ds.qrels_iter(), desc="Chargement qrels passage"):
    if qr.relevance > 0:
        pass_pos[str(qr.query_id)].add(str(qr.doc_id))

print("Nb requêtes (document) :", len(queries))
print("Nb qids avec docs positifs :", sum(1 for qid in queries if len(doc_pos[qid]) > 0))
print("Nb qids avec passages positifs :", sum(1 for qid in queries if len(pass_pos[qid]) > 0))

# %%

# =====================================================================
# 5) Sélection des requêtes admissibles (sans biais vers les plus faciles)
# =====================================================================
# Critère minimal : la requête doit avoir au moins un document positif ET au moins
# un passage positif dans MS MARCO.
#
# On ne trie PAS par nombre de positifs.
# On fait un tirage aléatoire reproductible.

eligible_qids = [
    qid for qid in queries
    if len(doc_pos[qid]) > 0 and len(pass_pos[qid]) > 0
]

print("Nb qids admissibles :", len(eligible_qids))

rng = random.Random(SEED)
selected_qids = eligible_qids.copy()
rng.shuffle(selected_qids)
selected_qids = selected_qids[:NUM_FINAL_QUERIES]

topics = pd.DataFrame({
    "qid": [norm_qid(qid) for qid in selected_qids],
    "query": [clean_query(queries[norm_qid(qid)]) for qid in selected_qids]
})

topics = topics[topics["query"].str.len() > 0].reset_index(drop=True)

print("Nb requêtes finales :", len(topics))
display(topics.head())

# %%

# =====================================================================
# 6) Construction / chargement de l'index complet MS MARCO document
# =====================================================================
# ATTENTION :
# - cette cellule peut être longue à la première exécution ;
# - ensuite, l'index est réutilisable.
#
# blocks=True est utile pour SDM.

def msmarco_docs_iter():
    for d in doc_ds.docs_iter():
        did = str(d.doc_id)
        text = get_doc_text(d)
        if text:
            yield {"docno": did, "text": text}

if not INDEX_PATH.exists():
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    print("Construction de l'index complet MS MARCO document...")
    indexer = pt.IterDictIndexer(
        str(INDEX_PATH),
        meta={"docno": 64},
        overwrite=True,
        threads=1,
        blocks=True
    )
    indexref = indexer.index(msmarco_docs_iter())
else:
    print("Index déjà présent. Réutilisation de :", INDEX_PATH)
    # Quand l'index existe déjà, PyTerrier retrouve le data.properties dans ce dossier
    indexref = str(INDEX_PATH)

index = pt.IndexFactory.of(indexref)
print("Index prêt :", indexref)

# %%

# =====================================================================
# 7) Récupération initiale top-50 par requête
# =====================================================================
# On essaie d'abord SDM, comme dans le papier.
# Si SDM échoue dans l'environnement, on garde un fallback explicite.

retrieval_model_used = None
results = None

try:
    sdm_pipe = pt.rewrite.SDM() >> pt.terrier.Retriever(
        index,
        wmodel="DirichletLM",
        num_results=TOPK_DOCS
    )
    results = sdm_pipe.transform(topics.copy())
    retrieval_model_used = "SDM + DirichletLM"
except Exception as e:
    print("[WARN] SDM indisponible dans cet environnement. Fallback vers BM25.")
    print("Erreur SDM :", repr(e))
    bm25_pipe = pt.terrier.Retriever(index, wmodel="BM25", num_results=TOPK_DOCS)
    results = bm25_pipe.transform(topics.copy())
    retrieval_model_used = "BM25 (fallback)"

print("Modèle de récupération utilisé :", retrieval_model_used)
print("Nb résultats bruts :", len(results))
display(results.head())

# %%

# =====================================================================
# 8) Construire le top-50 propre + labels document
# =====================================================================
rows = []
for _, r in results.iterrows():
    qid = norm_qid(r["qid"])
    doc_id = str(r["docno"])
    rank = int(r["rank"]) + 1
    score = float(r["score"])
    doc_label = 1 if doc_id in doc_pos[qid] else 0
    rows.append([qid, queries[qid], doc_id, rank, score, doc_label])

df_topdocs = pd.DataFrame(rows, columns=[
    "qid", "query_raw", "doc_id", "rank", "score", "doc_label"
])

df_topdocs = (
    df_topdocs
    .sort_values(["qid", "rank"])
    .groupby("qid", as_index=False)
    .head(TOPK_DOCS)
    .reset_index(drop=True)
)

# Statistiques rapides
qid_has_pos_doc = df_topdocs.groupby("qid")["doc_label"].sum().gt(0)
num_q_with_pos_doc = int(qid_has_pos_doc.sum())

print("Nb requêtes finales :", df_topdocs['qid'].nunique())
print("Nb requêtes avec >=1 doc positif dans top-50 :", num_q_with_pos_doc)

display(df_topdocs.head())

# %%

# =====================================================================
# 8) Construire le top-50 propre + labels document
# =====================================================================
rows = []
for _, r in results.iterrows():
    qid = norm_qid(r["qid"])
    doc_id = str(r["docno"])
    rank = int(r["rank"]) + 1
    score = float(r["score"])
    doc_label = 1 if doc_id in doc_pos[qid] else 0
    rows.append([qid, queries[qid], doc_id, rank, score, doc_label])

df_topdocs = pd.DataFrame(rows, columns=[
    "qid", "query_raw", "doc_id", "rank", "score", "doc_label"
])

df_topdocs = (
    df_topdocs
    .sort_values(["qid", "rank"])
    .groupby("qid", as_index=False)
    .head(TOPK_DOCS)
    .reset_index(drop=True)
)

# Statistiques rapides
qid_has_pos_doc = df_topdocs.groupby("qid")["doc_label"].sum().gt(0)
num_q_with_pos_doc = int(qid_has_pos_doc.sum())

print("Nb requêtes finales :", df_topdocs['qid'].nunique())
print("Nb requêtes avec >=1 doc positif dans top-50 :", num_q_with_pos_doc)

display(df_topdocs.head())

# %%

# =====================================================================
# 9) Charger les textes de passages positifs par requête
# =====================================================================
# Ici on récupère les passages positifs MS MARCO passage pour servir de supervision
# distante / silver supervision.

pos_pass_texts_by_qid = defaultdict(list)

for qid in tqdm(sorted(df_topdocs["qid"].unique()), desc="Chargement passages positifs"):
    pos_pids = list(pass_pos[qid])
    if MAX_POS_PASSAGES_PER_QID is not None:
        pos_pids = pos_pids[:MAX_POS_PASSAGES_PER_QID]

    for pid in pos_pids:
        p = pass_store.get(pid)
        if p is None:
            continue

        # Le champ texte peut s'appeler "text" selon le dataset
        ptext = getattr(p, "text", None)
        if ptext is None:
            ptext = str(p)

        if ptext and str(ptext).strip():
            pos_pass_texts_by_qid[qid].append(str(ptext))

stats_pass = pd.Series({qid: len(v) for qid, v in pos_pass_texts_by_qid.items()})
print("Nb requêtes avec au moins 1 passage positif chargé :", int((stats_pass > 0).sum()))
print("Moyenne de passages positifs chargés par qid :", round(float(stats_pass.mean()), 2) if len(stats_pass) else 0.0)

# %%

# =====================================================================
# 10) Construire des spans silver dans les documents positifs récupérés
# =====================================================================
# On ne traite que les documents positifs dans le top-50, car ce sont les documents
# qui jouent le rôle le plus proche de ceux annotés dans le papier.

df_pos_docs = df_topdocs[df_topdocs["doc_label"] == 1].copy()

if MAX_POS_DOCS_PER_QID is not None:
    df_pos_docs = (
        df_pos_docs.sort_values(["qid", "rank"])
        .groupby("qid", as_index=False)
        .head(MAX_POS_DOCS_PER_QID)
        .reset_index(drop=True)
    )

span_rows = []

for (qid, doc_id), g in tqdm(
    df_pos_docs.groupby(["qid", "doc_id"]),
    desc="Alignement passages -> spans"
):
    # Texte du document
    d = doc_store.get(str(doc_id))
    if d is None:
        continue

    doc_text = get_doc_text(d)
    doc_tokens = tok(doc_text)
    if len(doc_tokens) == 0:
        continue

    passage_texts = pos_pass_texts_by_qid.get(qid, [])
    if len(passage_texts) == 0:
        continue

    # On essaie plusieurs passages positifs pour ce qid.
    # On garde tous les spans raisonnables, car un document peut contenir plusieurs réponses.
    seen_spans = set()

    for passage_text in passage_texts:
        passage_tokens = tok(passage_text)
        if len(passage_tokens) == 0:
            continue

        aligned = fuzzy_align_span(doc_tokens, passage_tokens, min_ratio=0.45)
        if aligned is None:
            continue

        key = (aligned["start_token"], aligned["end_token"])
        if key in seen_spans:
            continue
        seen_spans.add(key)

        span_rows.append([
            qid,
            str(doc_id),
            int(g["rank"].iloc[0]),
            float(g["score"].iloc[0]),
            aligned["start_token"],
            aligned["end_token"],
            aligned["end_token"] - aligned["start_token"],
            aligned["align_score"],
            aligned["lex_recall"],
            aligned["lex_precision"],
            aligned["bigram_recall"],
            aligned["align_type"],
            " ".join(doc_tokens[aligned["start_token"]:aligned["end_token"]])
        ])

df_spans = pd.DataFrame(span_rows, columns=[
    "qid", "doc_id", "doc_rank", "doc_score",
    "span_start_token", "span_end_token", "span_len",
    "align_score", "lex_recall", "lex_precision", "bigram_recall",
    "align_type", "span_text"
])

print("Nb spans silver trouvés :", len(df_spans))
display(df_spans.head())

# %%

# =====================================================================
# 10) Construire des spans silver dans les documents positifs récupérés
# =====================================================================
# On ne traite que les documents positifs dans le top-50, car ce sont les documents
# qui jouent le rôle le plus proche de ceux annotés dans le papier.

df_pos_docs = df_topdocs[df_topdocs["doc_label"] == 1].copy()

if MAX_POS_DOCS_PER_QID is not None:
    df_pos_docs = (
        df_pos_docs.sort_values(["qid", "rank"])
        .groupby("qid", as_index=False)
        .head(MAX_POS_DOCS_PER_QID)
        .reset_index(drop=True)
    )

span_rows = []

for (qid, doc_id), g in tqdm(
    df_pos_docs.groupby(["qid", "doc_id"]),
    desc="Alignement passages -> spans"
):
    # Texte du document
    d = doc_store.get(str(doc_id))
    if d is None:
        continue

    doc_text = get_doc_text(d)
    doc_tokens = tok(doc_text)
    if len(doc_tokens) == 0:
        continue

    passage_texts = pos_pass_texts_by_qid.get(qid, [])
    if len(passage_texts) == 0:
        continue

    # On essaie plusieurs passages positifs pour ce qid.
    # On garde tous les spans raisonnables, car un document peut contenir plusieurs réponses.
    seen_spans = set()

    for passage_text in passage_texts:
        passage_tokens = tok(passage_text)
        if len(passage_tokens) == 0:
            continue

        aligned = fuzzy_align_span(doc_tokens, passage_tokens, min_ratio=0.45)
        if aligned is None:
            continue

        key = (aligned["start_token"], aligned["end_token"])
        if key in seen_spans:
            continue
        seen_spans.add(key)

        span_rows.append([
            qid,
            str(doc_id),
            int(g["rank"].iloc[0]),
            float(g["score"].iloc[0]),
            aligned["start_token"],
            aligned["end_token"],
            aligned["end_token"] - aligned["start_token"],
            aligned["align_score"],
            aligned["lex_recall"],
            aligned["lex_precision"],
            aligned["bigram_recall"],
            aligned["align_type"],
            " ".join(doc_tokens[aligned["start_token"]:aligned["end_token"]])
        ])

df_spans = pd.DataFrame(span_rows, columns=[
    "qid", "doc_id", "doc_rank", "doc_score",
    "span_start_token", "span_end_token", "span_len",
    "align_score", "lex_recall", "lex_precision", "bigram_recall",
    "align_type", "span_text"
])

print("Nb spans silver trouvés :", len(df_spans))
display(df_spans.head())

# %%

# =====================================================================
# 11) Générer les fenêtres 50/25 + silver labels à partir des spans
# =====================================================================
window_rows = []
label_rows = []

# Pour accélérer, on groupe les spans par (qid, doc_id)
spans_by_qid_doc = defaultdict(list)
for _, r in df_spans.iterrows():
    spans_by_qid_doc[(r["qid"], r["doc_id"])].append({
        "span_start": int(r["span_start_token"]),
        "span_end": int(r["span_end_token"]),
        "span_len": int(r["span_len"]),
        "align_score": float(r["align_score"])
    })

docs_to_process = df_topdocs[["qid", "doc_id", "rank", "score"]].drop_duplicates()

for _, r in tqdm(docs_to_process.iterrows(), total=len(docs_to_process), desc="Fenêtrage + labels"):
    qid = r["qid"]
    doc_id = str(r["doc_id"])
    doc_rank = int(r["rank"])
    doc_score = float(r["score"])

    d = doc_store.get(doc_id)
    if d is None:
        continue

    doc_text = get_doc_text(d)
    doc_tokens = tok(doc_text)
    if len(doc_tokens) == 0:
        continue

    windows = make_windows(doc_tokens, win=WINDOW_SIZE, overlap=WINDOW_OVERLAP)
    spans = spans_by_qid_doc.get((qid, doc_id), [])

    for window_id, (w_start, w_end, w_tokens) in enumerate(windows):
        window_text = " ".join(w_tokens)

        window_rows.append([
            qid, doc_id, doc_rank, doc_score,
            window_id, w_start, w_end, len(w_tokens), window_text
        ])

        # Si aucun span silver dans ce document, la fenêtre est non pertinente
        if len(spans) == 0:
            label_rows.append([
                qid, doc_id, window_id, 0, 0, 0.0, 0.0, None, None, None
            ])
            continue

        # Sinon, on prend le meilleur label obtenu contre tous les spans de ce doc
        best = None
        best_meta = None
        for i, sp in enumerate(spans):
            meta = label_window_from_span(
                window_start=w_start,
                window_end=w_end,
                span_start=sp["span_start"],
                span_end=sp["span_end"]
            )
            if best is None or meta["label"] > best["label"] or (
                meta["label"] == best["label"] and meta["overlap_tokens"] > best["overlap_tokens"]
            ):
                best = meta
                best_meta = (i, sp)

        label_rows.append([
            qid,
            doc_id,
            window_id,
            int(best["label"]),
            int(best["overlap_tokens"]),
            float(best["span_recall"]),
            float(best["window_precision"]),
            int(best_meta[1]["span_start"]),
            int(best_meta[1]["span_end"]),
            float(best_meta[1]["align_score"])
        ])

df_windows = pd.DataFrame(window_rows, columns=[
    "qid", "doc_id", "doc_rank", "doc_score",
    "window_id", "window_start_token", "window_end_token", "window_len", "window_text"
])

df_window_labels = pd.DataFrame(label_rows, columns=[
    "qid", "doc_id", "window_id",
    "silver_label",
    "overlap_tokens",
    "span_recall",
    "window_precision",
    "matched_span_start",
    "matched_span_end",
    "matched_span_align_score"
])

print("Nb fenêtres :", len(df_windows))
print("Distribution des labels :")
display(df_window_labels["silver_label"].value_counts(dropna=False).sort_index())

# %%

# =====================================================================
# 12) Contrôles qualité rapides
# =====================================================================
# Quelques stats utiles pour vérifier que la construction est cohérente.

print("Nb requêtes finales :", df_topdocs['qid'].nunique())
print("Nb docs top-50 :", len(df_topdocs))
print("Nb docs positifs top-50 :", int(df_topdocs['doc_label'].sum()))
print("Nb spans silver :", len(df_spans))
print("Nb fenêtres labellisées :", len(df_window_labels))

# Taux de requêtes avec au moins un span silver
q_with_span = df_spans["qid"].nunique() if len(df_spans) else 0
print("Nb requêtes avec au moins un span silver :", q_with_span)

# Taux de docs positifs ayant au moins un span
if len(df_pos_docs) > 0:
    covered_docs = df_spans[["qid", "doc_id"]].drop_duplicates().shape[0]
    print("Docs positifs couverts par au moins 1 span :", covered_docs, "/", len(df_pos_docs))

# Vérifier quelques exemples de fenêtres positives
merged_preview = (
    df_windows.merge(df_window_labels, on=["qid", "doc_id", "window_id"], how="inner")
    .query("silver_label > 0")
    .sort_values(["silver_label", "span_recall", "window_precision"], ascending=[False, False, False])
)

display(merged_preview.head(10))

# %%

# =====================================================================
# 13) Exports finaux
# =====================================================================
topics.to_csv(PATH_QUERIES, sep="\t", index=False)
df_topdocs.to_csv(PATH_TOPDOCS, sep="\t", index=False)
df_spans.to_csv(PATH_SPANS, sep="\t", index=False)
df_windows.to_csv(PATH_WINDOWS, sep="\t", index=False)
df_window_labels.to_csv(PATH_WINDOW_LABELS, sep="\t", index=False)

metadata = {
    "seed": SEED,
    "num_final_queries": int(topics["qid"].nunique()),
    "topk_docs": TOPK_DOCS,
    "window_size": WINDOW_SIZE,
    "window_overlap": WINDOW_OVERLAP,
    "max_pos_passages_per_qid": MAX_POS_PASSAGES_PER_QID,
    "max_pos_docs_per_qid": MAX_POS_DOCS_PER_QID,
    "retrieval_model_used": retrieval_model_used,
    "notes": [
        "Pas de pool enrichi en positifs dans ce protocole principal.",
        "Les labels de fenêtres sont des silver labels, pas des annotations humaines.",
        "Les spans sont obtenus par alignement exact ou fuzzy des passages positifs MS MARCO passage dans les documents MS MARCO document."
    ],
    "exports": {
        "queries_final": str(PATH_QUERIES),
        "topdocs_top50": str(PATH_TOPDOCS),
        "silver_spans": str(PATH_SPANS),
        "windows": str(PATH_WINDOWS),
        "silver_window_labels": str(PATH_WINDOW_LABELS)
    }
}

with open(PATH_METADATA, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Exports terminés :")
print(" -", PATH_QUERIES)
print(" -", PATH_TOPDOCS)
print(" -", PATH_SPANS)
print(" -", PATH_WINDOWS)
print(" -", PATH_WINDOW_LABELS)
print(" -", PATH_METADATA)

# %%
# =========================
# Sauvegarde vers Google Drive (Colab)
# =========================

from google.colab import drive
from pathlib import Path
import shutil
import os
from datetime import datetime

# 1) Monter Google Drive
drive.mount('/content/drive')

# 2) Dossier source local produit par le notebook
LOCAL_ARTIFACTS_DIR = Path("artifacts_notebook1")

# 3) Dossier cible dans ton Drive
DRIVE_BASE_DIR = Path("/content/drive/MyDrive/IR_project_backup")
RUN_NAME = f"notebook1_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DRIVE_RUN_DIR = DRIVE_BASE_DIR / RUN_NAME

# 4) Vérification
if not LOCAL_ARTIFACTS_DIR.exists():
    raise FileNotFoundError(f"Le dossier source n'existe pas: {LOCAL_ARTIFACTS_DIR.resolve()}")

# 5) Créer le dossier cible
DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)

# 6) Copier tout le contenu du dossier artifacts_notebook1 vers Drive
for item in LOCAL_ARTIFACTS_DIR.iterdir():
    src = item
    dst = DRIVE_RUN_DIR / item.name
    if item.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Sauvegarde terminée.")
print(f"Dossier local sauvegardé : {LOCAL_ARTIFACTS_DIR.resolve()}")
print(f"Dossier Drive : {DRIVE_RUN_DIR}")
print("Fichiers copiés :")
for p in sorted(DRIVE_RUN_DIR.iterdir()):
    print(" -", p.name)

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## Ce que produit ce notebook
#
# - `queries_final.tsv` : les requêtes retenues ;
# - `topdocs_top50.tsv` : top-50 documents récupérés par requête avec label document ;
# - `silver_spans.tsv` : spans silver alignés dans les documents ;
# - `windows.tsv` : toutes les fenêtres 50/25 ;
# - `silver_window_labels.tsv` : labels de fenêtres basés sur les spans ;
# - `dataset_metadata.json` : paramètres et traçabilité du run.
