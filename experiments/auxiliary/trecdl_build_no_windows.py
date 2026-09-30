# Auxiliary IR experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # Notebook 1 — Construction de données avec TREC-DL Passage, sans fenêtrage
#
# Objectif : reconstruire la partie **data** du projet en suivant l'esprit de l'article *Récupération de Passages et Recherche de Réponses*, mais en remplaçant MS MARCO train sparse par **TREC-DL Passage 2019 + 2020**.
#
# Décision méthodologique retenue :
# - on utilise directement les **passages TREC-DL** comme unités de récupération ;
# - on **ne découpe pas** les passages en fenêtres 50/25, car TREC-DL fournit déjà des jugements humains au niveau passage ;
# - on garde la logique de l'article : requêtes → récupération top-50 avec SDM → comparaison avec des labels de pertinence gradués.
#
# Pourquoi TREC-DL Passage ?
# - les jugements sont humains et gradués (`0, 1, 2, 3`) ;
# - plusieurs passages peuvent être pertinents pour une même requête ;
# - la densité des labels est beaucoup plus proche de l'article que MS MARCO train ;
# - on évite l'annotation LLM comme vérité terrain principale.
#
# Exports principaux :
# - `queries_final.tsv`
# - `trec_dl_qrels_passages.tsv`
# - `top_passages_top50.tsv`
# - `passage_labels.tsv`
# - `summary_by_qid.tsv`
# - `dataset_metadata.json`
#
# ⚠️ Différence avec l'article : l'article part de documents GOV2 puis récupère des fenêtres/passages. Ici, on travaille directement avec les passages annotés de TREC-DL. C'est moins fidèle à la forme exacte, mais plus naturel pour cette collection et beaucoup plus solide que MS MARCO train sparse.

# %%
# =====================================================================
# 1) Installation des dépendances
# =====================================================================
# NOTEBOOK: !pip -q install python-terrier ir_datasets pandas numpy tqdm

# %%
# =====================================================================
# 2) Imports, paramètres et chemins
# =====================================================================
import os
import re
import json
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import ir_datasets
import pyterrier as pt

# Initialisation PyTerrier
if hasattr(pt, "java"):
    if not pt.java.started():
        pt.java.init()
else:
    if not pt.started():
        pt.init()

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------
# Choix des datasets TREC-DL Passage
# ---------------------------------------------------------------------
TREC_DL_DATASETS = [
    "msmarco-passage/trec-dl-2019",
    "msmarco-passage/trec-dl-2020",
]

# Corpus complet MS MARCO Passage utilisé par TREC-DL.
CORPUS_DATASET_ID = "msmarco-passage/train"

# 82 permet de rester proche de l'article.
NUM_QUERIES_FINAL = 82

# Adaptation : dans l'article c'est top-50 documents, ici top-50 passages.
TOPK_INITIAL = 50

# Pertinence forte : approximation de excellent/perfect.
MIN_STRONG_RELEVANCE = 2

# Dossiers
WORKDIR = Path.cwd()
ARTIFACT_DIR = WORKDIR / "artifacts_trec_dl_notebook1_no_windows"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_DIR = ARTIFACT_DIR / "pt_msmarco_passage_index"

# Exports
PATH_QUERIES = ARTIFACT_DIR / "queries_final.tsv"
PATH_QRELS = ARTIFACT_DIR / "trec_dl_qrels_passages.tsv"
PATH_TOP_PASSAGES = ARTIFACT_DIR / "top_passages_top50.tsv"
PATH_PASSAGE_LABELS = ARTIFACT_DIR / "passage_labels.tsv"
PATH_QID_SUMMARY = ARTIFACT_DIR / "summary_by_qid.tsv"
PATH_METADATA = ARTIFACT_DIR / "dataset_metadata.json"

print("Dossier de travail :", WORKDIR)
print("Dossier artefacts :", ARTIFACT_DIR)
print("Datasets TREC-DL :", TREC_DL_DATASETS)

# %% [markdown]
# ## 3) Chargement des requêtes et qrels TREC-DL
#
# Les qrels TREC-DL Passage contiennent des jugements gradués :
#
# - `0` : non pertinent ;
# - `1` : lié / partiellement pertinent ;
# - `2` : pertinent ;
# - `3` : très pertinent.
#
# Mapping utilisé pour se rapprocher de l'article :
#
# | TREC-DL | Nom utilisé dans le projet | Rôle |
# |---:|---|---|
# | 0 | `non_relevant` | non pertinent |
# | 1 | `acceptable` | faible / partiel |
# | 2 | `excellent` | pertinent fort |
# | 3 | `perfect` | très pertinent |
#
# Remarque : l'article contient aussi une catégorie `good`. Comme TREC-DL n'a que trois niveaux positifs, on ne peut pas reproduire exactement les quatre niveaux de l'article. Pour l'évaluation principale, on utilise `relevance >= 2`, par analogie avec `excellent/perfect`.

# %%
# =====================================================================
# 3) Chargement des requêtes et qrels TREC-DL
# =====================================================================

def norm_qid(qid):
    return str(qid).strip()

def trec_dl_to_article_label(rel):
    # Mapping lisible entre les labels numériques TREC-DL et les labels proches de l'article.
    rel = int(rel)
    if rel <= 0:
        return "non_relevant"
    if rel == 1:
        return "acceptable"
    if rel == 2:
        return "excellent"
    return "perfect"

all_queries = []
all_qrels = []

for ds_id in TREC_DL_DATASETS:
    print("Chargement :", ds_id)
    ds = ir_datasets.load(ds_id)

    # Queries
    for q in ds.queries_iter():
        query_text = getattr(q, "text", None)
        if query_text is None:
            query_text = getattr(q, "title", None)
        if query_text is None:
            query_text = str(q)

        all_queries.append({
            "qid": norm_qid(q.query_id),
            "query": str(query_text),
            "source_dataset": ds_id,
        })

    # Qrels
    for qr in ds.qrels_iter():
        all_qrels.append({
            "qid": norm_qid(qr.query_id),
            "passage_id": str(qr.doc_id),
            "relevance": int(qr.relevance),
            "source_dataset": ds_id,
        })

df_queries_all = pd.DataFrame(all_queries).drop_duplicates(["qid", "query"])
df_qrels_all = pd.DataFrame(all_qrels).drop_duplicates(["qid", "passage_id"])

df_qrels_all["article_label_name"] = df_qrels_all["relevance"].map(trec_dl_to_article_label)
df_qrels_all["label_any"] = (df_qrels_all["relevance"] > 0).astype(int)
df_qrels_all["label_strong"] = (df_qrels_all["relevance"] >= MIN_STRONG_RELEVANCE).astype(int)

print("Nombre total de requêtes :", df_queries_all["qid"].nunique())
print("Nombre total de qrels :", len(df_qrels_all))
print("Distribution des labels :")
display(df_qrels_all["relevance"].value_counts().sort_index())

stats_qid = (
    df_qrels_all.groupby("qid")
    .agg(
        num_judged_passages=("passage_id", "nunique"),
        num_relevant_passages_any=("label_any", "sum"),
        num_relevant_passages_strong=("label_strong", "sum"),
        max_relevance=("relevance", "max"),
    )
    .reset_index()
)

print("
Statistiques par requête :")
display(stats_qid.describe())

display(stats_qid.sort_values("num_relevant_passages_strong", ascending=False).head(10))

# %% [markdown]
# ## 4) Sélection des 82 requêtes finales
#
# On tire aléatoirement parmi les requêtes ayant au moins un passage fortement pertinent (`relevance >= 2`).
#
# Cela évite le biais de sélectionner seulement les requêtes les plus faciles, tout en garantissant que chaque requête finale possède au moins une vérité terrain exploitable.

# %%
# =====================================================================
# 4) Sélection des requêtes finales
# =====================================================================

eligible = stats_qid[stats_qid["num_relevant_passages_strong"] > 0].copy()

print("Requêtes éligibles avec au moins un passage relevance >= 2 :", len(eligible))

if len(eligible) < NUM_QUERIES_FINAL:
    print("[WARN] Pas assez de requêtes éligibles. On prendra toutes les requêtes disponibles.")
    selected_qids = eligible["qid"].tolist()
else:
    selected_qids = (
        eligible.sample(n=NUM_QUERIES_FINAL, random_state=SEED)
        .sort_values("qid")["qid"].tolist()
    )

topics = (
    df_queries_all[df_queries_all["qid"].isin(selected_qids)]
    [["qid", "query"]]
    .drop_duplicates("qid")
    .sort_values("qid")
    .reset_index(drop=True)
)

df_qrels = df_qrels_all[df_qrels_all["qid"].isin(selected_qids)].copy()

print("Nombre de requêtes finales :", topics["qid"].nunique())
print("Nombre de qrels sélectionnés :", len(df_qrels))
print("Distribution labels sélectionnés :")
display(df_qrels["relevance"].value_counts().sort_index())

selected_stats = stats_qid[stats_qid["qid"].isin(selected_qids)].copy()
display(selected_stats.describe())

# %% [markdown]
# ## 5) Construction ou chargement de l'index complet MS MARCO Passage
#
# On indexe le corpus complet `msmarco-passage/train` pour faire le premier retrieval.
#
# ⚠️ Cette étape peut être longue lors de la première exécution. Si l'index existe déjà dans le dossier des artefacts, il est réutilisé.

# %%
# =====================================================================
# 5) Construction ou chargement de l'index complet MS MARCO Passage
# =====================================================================

corpus_ds = ir_datasets.load(CORPUS_DATASET_ID)

def passage_iter_for_index():
    for d in corpus_ds.docs_iter():
        yield {
            "docno": str(d.doc_id),
            "text": str(d.text),
        }

index_data = INDEX_DIR / "data.properties"
if index_data.exists():
    print("Index existant trouvé :", INDEX_DIR)
    index_ref = pt.IndexRef.of(str(index_data))
else:
    print("Construction de l'index complet MS MARCO Passage...")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    indexer = pt.IterDictIndexer(
        str(INDEX_DIR),
        overwrite=True,
        meta={
            "docno": 32,
            "text": 4096,
        },
        blocks=True
    )
    index_ref = indexer.index(tqdm(passage_iter_for_index(), desc="Indexation passages"))

print("Index prêt :", index_ref)

# %% [markdown]
# ## 6) Retrieval initial SDM top-50 passages
#
# L'article utilise SDM pour sélectionner les top-50 documents. Ici, adaptation à TREC-DL Passage : on utilise SDM pour sélectionner les top-50 passages.
#
# Si SDM échoue dans l'environnement PyTerrier, la cellule bascule vers BM25 afin de ne pas bloquer l'exécution. Le modèle utilisé est enregistré dans les métadonnées.

# %%
# =====================================================================
# 6) Retrieval initial : SDM top-50 passages
# =====================================================================

topics_pt = topics[["qid", "query"]].copy()
topics_pt["qid"] = topics_pt["qid"].astype(str)
topics_pt["query"] = topics_pt["query"].astype(str)

retrieval_model_used = "SDM"

try:
    print("Tentative SDM...")
    # SDM réécrit la requête, puis Terrier score avec BM25 comme base.
    sdm = pt.rewrite.SDM() >> pt.terrier.Retriever(index_ref, wmodel="BM25", num_results=TOPK_INITIAL)
    run_initial = sdm.transform(topics_pt)
except Exception as e:
    print("[WARN] SDM a échoué. Fallback vers BM25.")
    print("Erreur SDM :", repr(e))
    retrieval_model_used = "BM25_fallback"
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25", num_results=TOPK_INITIAL)
    run_initial = bm25.transform(topics_pt)

run_initial = run_initial.copy()
run_initial["qid"] = run_initial["qid"].astype(str)
run_initial["passage_id"] = run_initial["docno"].astype(str)

keep_cols = [c for c in ["qid", "query", "docno", "passage_id", "rank", "score"] if c in run_initial.columns]
df_top_passages = run_initial[keep_cols].copy()
df_top_passages = df_top_passages.sort_values(["qid", "rank"]).reset_index(drop=True)

print("Modèle utilisé :", retrieval_model_used)
print("Nombre de lignes top passages :", len(df_top_passages))
print("Nombre moyen de passages récupérés par requête :", df_top_passages.groupby("qid").size().mean())
display(df_top_passages.head())

# %% [markdown]
# ## 7) Fusion des top passages avec les labels TREC-DL
#
# On ajoute les labels humains TREC-DL aux passages récupérés.
#
# Important :
# - un passage récupéré peut ne pas avoir été jugé par TREC-DL ;
# - dans ce cas, `is_judged = False` et `relevance = 0` par défaut ;
# - `label_strong = 1` signifie `relevance >= 2`, notre approximation de `excellent/perfect`.

# %%
# =====================================================================
# 7) Fusion top passages + labels TREC-DL
# =====================================================================

labels_for_merge = df_qrels[["qid", "passage_id", "relevance"]].copy()
labels_for_merge["qid"] = labels_for_merge["qid"].astype(str)
labels_for_merge["passage_id"] = labels_for_merge["passage_id"].astype(str)

df_top_passages = df_top_passages.merge(
    labels_for_merge,
    on=["qid", "passage_id"],
    how="left"
)

df_top_passages["is_judged"] = df_top_passages["relevance"].notna()
df_top_passages["relevance"] = df_top_passages["relevance"].fillna(0).astype(int)
df_top_passages["article_label_name"] = df_top_passages["relevance"].map(trec_dl_to_article_label)
df_top_passages["label_any"] = (df_top_passages["relevance"] > 0).astype(int)
df_top_passages["label_strong"] = (df_top_passages["relevance"] >= MIN_STRONG_RELEVANCE).astype(int)

print("Distribution labels dans les top-50 récupérés :")
display(df_top_passages["relevance"].value_counts().sort_index())

print("Nombre de passages top-50 jugés :", int(df_top_passages["is_judged"].sum()), "/", len(df_top_passages))
print("Nombre de passages top-50 pertinents forts :", int(df_top_passages["label_strong"].sum()))

display(df_top_passages.head())

# %% [markdown]
# ## 8) Ajout du texte des passages
#
# On ajoute le texte pour faciliter :
# - le Notebook 2 de modélisation ;
# - l'analyse qualitative ;
# - l'affichage des passages récupérés.
#
# Aucun fenêtrage n'est effectué : chaque ligne reste un passage TREC-DL/MS MARCO.

# %%
# =====================================================================
# 8) Ajouter le texte des passages récupérés
# =====================================================================

doc_store = corpus_ds.docs_store()

def get_passage_text(pid):
    d = doc_store.get(str(pid))
    if d is None:
        return ""
    return str(d.text)

unique_pids = df_top_passages["passage_id"].drop_duplicates().tolist()
pid_to_text = {}

for pid in tqdm(unique_pids, desc="Chargement textes passages"):
    pid_to_text[pid] = get_passage_text(pid)

df_top_passages["passage_text"] = df_top_passages["passage_id"].map(pid_to_text).fillna("")
df_top_passages["passage_len_tokens"] = df_top_passages["passage_text"].map(lambda x: len(re.findall(r"\w+", str(x).lower())))

print("Passages sans texte :", int((df_top_passages["passage_text"].str.len() == 0).sum()))
display(df_top_passages.head())

# %% [markdown]
# ## 9) Construction du fichier `passage_labels.tsv`
#
# Ce fichier remplace l'ancien `silver_window_labels.tsv`.
#
# Ici, les labels ne sont pas des pseudo-labels : ce sont les jugements TREC-DL humains, ramenés dans un format pratique pour la suite.

# %%
# =====================================================================
# 9) Construction passage_labels.tsv
# =====================================================================

passage_labels_cols = [
    "qid",
    "passage_id",
    "rank",
    "score",
    "is_judged",
    "relevance",
    "article_label_name",
    "label_any",
    "label_strong",
    "passage_len_tokens",
]

passage_labels = df_top_passages[passage_labels_cols].copy()

print("Nombre de labels de passages top-50 :", len(passage_labels))
print("Distribution article_label_name :")
display(passage_labels["article_label_name"].value_counts())

display(passage_labels.head())

# %% [markdown]
# ## 10) Contrôles qualité
#
# Cette cellule vérifie :
# - combien de requêtes ont au moins un passage jugé dans le top-50 ;
# - combien ont au moins un passage pertinent dans le top-50 ;
# - combien ont au moins un passage fortement pertinent (`relevance >= 2`) dans le top-50.
#
# Ce contrôle est important : même si TREC-DL est dense, un modèle initial faible peut ne pas retrouver de passage pertinent dans son top-50.

# %%
# =====================================================================
# 10) Contrôles qualité
# =====================================================================

summary_by_qid = (
    df_top_passages.groupby("qid")
    .agg(
        num_retrieved_passages=("passage_id", "count"),
        num_judged=("is_judged", "sum"),
        num_rel_any=("label_any", "sum"),
        num_rel_strong=("label_strong", "sum"),
        max_relevance=("relevance", "max"),
    )
    .reset_index()
)

summary_by_qid = topics[["qid", "query"]].merge(summary_by_qid, on="qid", how="left")
for c in ["num_retrieved_passages", "num_judged", "num_rel_any", "num_rel_strong", "max_relevance"]:
    summary_by_qid[c] = summary_by_qid[c].fillna(0).astype(int)

print("Résumé par requête :")
display(summary_by_qid.describe(include="all"))

print("
Nombre de requêtes avec au moins un passage top-50 jugé :")
print(int((summary_by_qid["num_judged"] > 0).sum()), "/", len(summary_by_qid))

print("
Nombre de requêtes avec au moins un passage top-50 pertinent :")
print("relevance > 0 :", int((summary_by_qid["num_rel_any"] > 0).sum()), "/", len(summary_by_qid))
print("relevance >= 2 :", int((summary_by_qid["num_rel_strong"] > 0).sum()), "/", len(summary_by_qid))

print("
Distribution max relevance par requête :")
display(summary_by_qid["max_relevance"].value_counts().sort_index())

display(summary_by_qid.sort_values("num_rel_strong", ascending=False).head(10))

# %% [markdown]
# ## 11) Exports
#
# Fichiers exportés pour le Notebook 2 :
#
# - `queries_final.tsv` : requêtes finales ;
# - `trec_dl_qrels_passages.tsv` : qrels TREC-DL des requêtes sélectionnées ;
# - `top_passages_top50.tsv` : passages récupérés top-50 + texte + labels ;
# - `passage_labels.tsv` : fichier compact des labels des passages récupérés ;
# - `summary_by_qid.tsv` : résumé qualité par requête ;
# - `dataset_metadata.json` : métadonnées expérimentales.
#
# Il n'y a plus de `windows.tsv`, parce qu'on n'utilise plus le fenêtrage.

# %%
# =====================================================================
# 11) Exports
# =====================================================================

topics.to_csv(PATH_QUERIES, sep="	", index=False)
df_qrels.to_csv(PATH_QRELS, sep="	", index=False)
df_top_passages.to_csv(PATH_TOP_PASSAGES, sep="	", index=False)
passage_labels.to_csv(PATH_PASSAGE_LABELS, sep="	", index=False)
summary_by_qid.to_csv(PATH_QID_SUMMARY, sep="	", index=False)

metadata = {
    "project": "IR answer passage retrieval replication/adaptation",
    "dataset_type": "TREC-DL Passage direct passages no windows",
    "trec_dl_datasets": TREC_DL_DATASETS,
    "corpus_dataset_id": CORPUS_DATASET_ID,
    "num_queries_final": int(topics["qid"].nunique()),
    "topk_initial": TOPK_INITIAL,
    "retrieval_model_used": retrieval_model_used,
    "min_strong_relevance": MIN_STRONG_RELEVANCE,
    "unit_of_retrieval": "passage",
    "uses_windows": False,
    "label_mapping": {
        "0": "non_relevant",
        "1": "acceptable",
        "2": "excellent",
        "3": "perfect",
    },
    "num_qrels_selected": int(len(df_qrels)),
    "num_top_passages": int(len(df_top_passages)),
    "num_judged_top_passages": int(df_top_passages["is_judged"].sum()),
    "num_strong_relevant_top_passages": int(df_top_passages["label_strong"].sum()),
    "num_queries_with_strong_relevant_in_top50": int((summary_by_qid["num_rel_strong"] > 0).sum()),
    "notes": (
        "Direct TREC-DL passage labels are used. No 50/25 window segmentation is applied. "
        "This differs from the GOV2 document-window setup but avoids artificial passage lengths."
    ),
}

PATH_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

print("Exports terminés dans :", ARTIFACT_DIR)
for p in [PATH_QUERIES, PATH_QRELS, PATH_TOP_PASSAGES, PATH_PASSAGE_LABELS, PATH_QID_SUMMARY, PATH_METADATA]:
    print("-", p.name, "OK" if p.exists() else "MANQUANT")

# %% [markdown]
# ## 12) Sauvegarde Google Drive sécurisée
#
# Cellule optionnelle pour Colab. Elle vérifie les fichiers importants avant de les copier dans Drive.

# %%
# =====================================================================
# 12) Backup Google Drive sécurisé
# =====================================================================
# Exécute cette cellule si tu es sur Colab et que tu veux sauvegarder les artefacts.

from google.colab import drive
from datetime import datetime

drive.mount('/content/drive')

DRIVE_BASE_DIR = Path("/content/drive/MyDrive/IR_project_backup")
RUN_NAME = f"trec_dl_no_windows_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DRIVE_RUN_DIR = DRIVE_BASE_DIR / RUN_NAME

expected = [
    PATH_QUERIES,
    PATH_QRELS,
    PATH_TOP_PASSAGES,
    PATH_PASSAGE_LABELS,
    PATH_QID_SUMMARY,
    PATH_METADATA,
]

missing = [str(p) for p in expected if not p.exists()]
if missing:
    raise FileNotFoundError("Backup refusé. Fichiers manquants :
" + "
".join(missing))

DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)

for item in ARTIFACT_DIR.iterdir():
    src = item
    dst = DRIVE_RUN_DIR / item.name
    if item.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Sauvegarde terminée.")
print("Dossier Drive :", DRIVE_RUN_DIR)
print("Fichiers copiés :")
for p in sorted(DRIVE_RUN_DIR.iterdir()):
    print(" -", p.name)

# %% [markdown]
# ## Notes méthodologiques pour le rapport
#
# Formulation possible :
#
# > Nous remplaçons MS MARCO train par TREC-DL Passage 2019+2020 afin d'obtenir des jugements humains plus denses et gradués. Contrairement à la première version basée sur MS MARCO train, TREC-DL fournit plusieurs passages pertinents par requête. Nous utilisons directement les passages comme unités de récupération, sans fenêtrage 50/25, car les passages sont déjà l'unité annotée dans TREC-DL. Les labels TREC-DL `0,1,2,3` sont convertis en catégories proches de l'article : non pertinent, acceptable, excellent, perfect. Pour l'évaluation binaire principale, `relevance >= 2` est considéré pertinent fort, par analogie avec l'utilisation des labels perfect/excellent dans l'article.
#
# Limites :
# - TREC-DL n'est pas GOV2 ;
# - les passages sont des passages MS MARCO, pas des passages annotés librement dans des documents GOV2 ;
# - la catégorie `good` de l'article n'a pas d'équivalent direct séparé dans TREC-DL ;
# - mais les jugements sont humains, gradués et beaucoup plus denses que MS MARCO train.
