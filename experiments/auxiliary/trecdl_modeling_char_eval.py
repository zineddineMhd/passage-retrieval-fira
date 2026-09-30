# Auxiliary IR experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # Notebook 2 — Modélisation et évaluation sur TREC-DL Passage, sans fenêtres + évaluation caractère
#
# Ce notebook est adapté au notebook de données `01_build_trec_dl_passage_dataset_no_windows.ipynb`.
#
# Différence principale avec l'ancienne version MS MARCO :
# - l'unité de retrieval est directement le **passage TREC-DL** ;
# - il n'y a plus de `windows.tsv` ;
# - les labels viennent directement des qrels TREC-DL : `0, 1, 2, 3` ;
# - l'évaluation principale utilise `relevance >= 2`, qui approxime `excellent/perfect` dans l'article.
#
# Modèles évalués :
# - QL / DirichletLM ;
# - SDM ;
# - une interpolation optionnelle entre score QL et score initial SDM, nommée explicitement `QL_INITIAL_INTERP`.
#
# Correction importante dans cette version :
# - l'index PyTerrier des passages est reconstruit avec `fields=True` et `blocks=True`, ce qui est nécessaire pour éviter les erreurs `Index must have fields` et `This index does not support blocks` avec QL/SDM.
#
# Important : `QL_INITIAL_INTERP` n'est **pas exactement** l'interpolation document/passage de l'article, car ici nous travaillons directement au niveau passage.
#
#
# Cette version ajoute une évaluation **character-level / Passage2-like** : les passages pertinents sont pondérés par leur longueur en caractères, afin de se rapprocher de l'esprit de l'évaluation de l'article.

# %%
# =====================================================================
# 1) Installation des dépendances
# =====================================================================
# NOTEBOOK: !pip -q install python-terrier ir_datasets pandas numpy tqdm ranx

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# =====================================================================
# 2) Imports, paramètres et chemins
# =====================================================================
import os
import re
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import pyterrier as pt

# Initialisation PyTerrier
# Remarque : dans les versions récentes, Java démarre souvent automatiquement.
if hasattr(pt, "java"):
    if not pt.java.started():
        pt.java.init()
else:
    if not pt.started():
        pt.init()

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------
# Paramètres principaux
# ---------------------------------------------------------------------
TOPK_RETRIEVAL = 1000      # nombre large pour pouvoir filtrer ensuite sur les candidats top-50
TOPK_EVAL = 50             # on évalue principalement le top-50 re-ranké
MIN_STRONG_RELEVANCE = 2   # 2/3 approxime excellent/perfect
INTERP_ALPHA = 0.75        # poids QL dans QL_INITIAL_INTERP

# Important : l'index candidat est petit, donc on force sa reconstruction.
# Cela évite de réutiliser un ancien index sans fields/blocks incompatible avec SDM.
FORCE_REBUILD_INDEX = True

WORKDIR = Path.cwd()

# Dossier d'entrée par défaut si tu exécutes Notebook 2 juste après Notebook 1.
# Modifie ce chemin si ton backup Notebook 1 a un autre nom.
LOCAL_ARTIFACT_DIR = Path("/content/drive/MyDrive/IR_project_backup/trec_dl_no_windows_backup_20260428_124738")

# Dossier de sortie Notebook 2
ARTIFACT_DIR_M2 = WORKDIR / "artifacts_trec_dl_notebook2_no_windows"
ARTIFACT_DIR_M2.mkdir(parents=True, exist_ok=True)

PASSAGE_INDEX_DIR = ARTIFACT_DIR_M2 / "pt_candidate_passage_index"

# Exports Notebook 2
PATH_RUN_QL = ARTIFACT_DIR_M2 / "run_ql.tsv"
PATH_RUN_SDM = ARTIFACT_DIR_M2 / "run_sdm.tsv"
PATH_RUN_INTERP = ARTIFACT_DIR_M2 / "run_ql_initial_interp.tsv"
PATH_METRICS_SUMMARY = ARTIFACT_DIR_M2 / "metrics_summary.tsv"
PATH_PER_QUERY_METRICS = ARTIFACT_DIR_M2 / "per_query_metrics.tsv"
PATH_CHAR_METRICS_SUMMARY = ARTIFACT_DIR_M2 / "char_metrics_summary.tsv"
PATH_CHAR_PER_QUERY_METRICS = ARTIFACT_DIR_M2 / "char_per_query_metrics.tsv"
PATH_METADATA_M2 = ARTIFACT_DIR_M2 / "modeling_metadata.json"

print("Dossier de travail :", WORKDIR)
print("Dossier sortie Notebook 2 :", ARTIFACT_DIR_M2)

# %%
# =====================================================================
# 3) Trouver automatiquement les artefacts du Notebook 1
# =====================================================================
# Le notebook cherche d'abord le dossier local.
# Si tu as sauvegardé dans Google Drive, il cherche le backup le plus récent
# qui contient les fichiers attendus.

REQUIRED_NB1_FILES = [
    "queries_final.tsv",
    "trec_dl_qrels_passages.tsv",
    "top_passages_top50.tsv",
    "passage_labels.tsv",
    "summary_by_qid.tsv",
    "dataset_metadata.json",
]

# Sécurité : convertir en Path même si la variable a été modifiée en str.
LOCAL_ARTIFACT_DIR = Path(LOCAL_ARTIFACT_DIR)

def is_valid_artifact_dir(path) -> bool:
    path = Path(path)
    return path.exists() and path.is_dir() and all((path / f).exists() for f in REQUIRED_NB1_FILES)

def find_nb1_artifacts():
    # 1) local/direct
    if is_valid_artifact_dir(LOCAL_ARTIFACT_DIR):
        return Path(LOCAL_ARTIFACT_DIR)

    # 2) chercher dans Drive si disponible
    drive_root = Path("/content/drive/MyDrive/IR_project_backup")
    if drive_root.exists():
        candidates = [p for p in drive_root.iterdir() if p.is_dir()]
        candidates = sorted(candidates, key=lambda p: p.name, reverse=True)
        for c in candidates:
            if is_valid_artifact_dir(c):
                return c

    raise FileNotFoundError(
        "Impossible de trouver les artefacts du Notebook 1. "
        "Exécute d'abord Notebook 1 ou monte Google Drive avec un backup valide."
    )

ARTIFACT_DIR_NB1 = find_nb1_artifacts()
print("Artefacts Notebook 1 utilisés :", ARTIFACT_DIR_NB1)

PATH_QUERIES = ARTIFACT_DIR_NB1 / "queries_final.tsv"
PATH_QRELS = ARTIFACT_DIR_NB1 / "trec_dl_qrels_passages.tsv"
PATH_TOP_PASSAGES = ARTIFACT_DIR_NB1 / "top_passages_top50.tsv"
PATH_PASSAGE_LABELS = ARTIFACT_DIR_NB1 / "passage_labels.tsv"
PATH_QID_SUMMARY = ARTIFACT_DIR_NB1 / "summary_by_qid.tsv"
PATH_METADATA_NB1 = ARTIFACT_DIR_NB1 / "dataset_metadata.json"

# %%
# =====================================================================
# 4) Helpers généraux
# =====================================================================

def norm_qid(x):
    return str(x)

def norm_pid(x):
    return str(x)

def safe_minmax_normalize(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

def trec_dl_to_article_label(rel):
    rel = int(rel)
    return {
        0: "non_relevant",
        1: "acceptable",
        2: "excellent",
        3: "perfect",
    }.get(rel, "unknown")

# %%
# =====================================================================
# 5) Chargement des artefacts du Notebook 1
# =====================================================================

topics = pd.read_csv(PATH_QUERIES, sep="	")
df_qrels = pd.read_csv(PATH_QRELS, sep="	")
df_top_passages = pd.read_csv(PATH_TOP_PASSAGES, sep="	")
passage_labels = pd.read_csv(PATH_PASSAGE_LABELS, sep="	")
summary_by_qid = pd.read_csv(PATH_QID_SUMMARY, sep="	")
metadata_nb1 = json.loads(PATH_METADATA_NB1.read_text(encoding="utf-8"))

# Normalisation types
for df in [topics, df_qrels, df_top_passages, passage_labels, summary_by_qid]:
    if "qid" in df.columns:
        df["qid"] = df["qid"].map(norm_qid)

for df in [df_qrels, df_top_passages, passage_labels]:
    if "passage_id" in df.columns:
        df["passage_id"] = df["passage_id"].map(norm_pid)

# S'assurer que le texte existe dans df_top_passages
if "passage_text" not in df_top_passages.columns:
    raise ValueError("df_top_passages doit contenir passage_text. Relance Notebook 1 no-windows.")

print("Nb requêtes :", topics["qid"].nunique())
print("Nb qrels sélectionnés :", len(df_qrels))
print("Nb top passages :", len(df_top_passages))
print("Nb labels passages :", len(passage_labels))
print("Distribution relevance dans top_passages :")
display(df_top_passages["relevance"].value_counts().sort_index())
print("Résumé Notebook 1 :")
display(summary_by_qid.head())

# %% [markdown]
# ## 6) Index des passages candidats
#
# On indexe les passages candidats récupérés par le Notebook 1. Cela permet de re-scorrer les mêmes unités avec QL et SDM.
#
# Important : comme on travaille sans fenêtres, `docno = passage_id`.

# %%
# =====================================================================
# 6) Construction de l'index des passages candidats compatible QL + SDM
# =====================================================================
# SDM a besoin des positions/proximités des termes => blocks=True.
# DirichletLM/Hiemstra_LM dans Terrier peuvent exiger des champs => fields=True.
# On reconstruit donc l'index avec fields=True ET blocks=True.

# Éviter les doublons : un passage peut apparaître pour plusieurs requêtes.
candidate_passages = (
    df_top_passages[["passage_id", "passage_text"]]
    .drop_duplicates(subset=["passage_id"])
    .rename(columns={"passage_id": "docno", "passage_text": "text"})
    .copy()
)

candidate_passages["docno"] = candidate_passages["docno"].astype(str)
candidate_passages["text"] = candidate_passages["text"].fillna("").astype(str)

print("Nombre de passages candidats uniques :", len(candidate_passages))
display(candidate_passages.head())

if FORCE_REBUILD_INDEX and PASSAGE_INDEX_DIR.exists():
    print("Suppression de l'ancien index candidat :", PASSAGE_INDEX_DIR)
    shutil.rmtree(PASSAGE_INDEX_DIR)

if PASSAGE_INDEX_DIR.exists():
    print("Index existant détecté :", PASSAGE_INDEX_DIR)
    index_ref = pt.IndexRef.of(str(PASSAGE_INDEX_DIR / "data.properties"))
else:
    print("Construction de l'index avec fields=True et blocks=True...")
    indexer = pt.IterDictIndexer(
        str(PASSAGE_INDEX_DIR),
        overwrite=True,
        meta={"docno": 128},
        fields=True,
        blocks=True,
    )
    index_ref = indexer.index(candidate_passages.to_dict(orient="records"))

print("Index prêt :", index_ref)

# Diagnostic rapide de l'index.
index = pt.IndexFactory.of(index_ref)
print(index.getCollectionStatistics())

# %%
# =====================================================================
# 7) Définir les requêtes PyTerrier
# =====================================================================

topics_pt = topics[["qid", "query"]].copy()
topics_pt["qid"] = topics_pt["qid"].astype(str)
topics_pt["query"] = topics_pt["query"].astype(str)

display(topics_pt.head())

# %%
# =====================================================================
# 7bis) Vérification rapide de compatibilité de l'index
# =====================================================================
# Cette cellule est un diagnostic. Elle ne remplace pas l'index.
# Si SDM échoue ensuite avec une erreur de blocks, relance la cellule 6
# avec FORCE_REBUILD_INDEX=True.

index = pt.IndexFactory.of(index_ref)
print("Statistiques de collection :")
print(index.getCollectionStatistics())
print("Index utilisé :", index_ref)

# %% [markdown]
# ## 8) Scoring QL et SDM
#
# On récupère largement (`TOPK_RETRIEVAL`) puis on filtre sur les passages candidats top-50 issus du Notebook 1 pour rester dans le même espace candidat.

# %%
# =====================================================================
# 8) Lancer QL et SDM
# =====================================================================

def run_retriever(model_name: str, pipeline, topics_df: pd.DataFrame) -> pd.DataFrame:
    print(f"Lancement {model_name}...")
    run = pipeline.transform(topics_df).copy()
    run["qid"] = run["qid"].astype(str)
    run["passage_id"] = run["docno"].astype(str)

    keep_cols = [c for c in ["qid", "docno", "passage_id", "rank", "score"] if c in run.columns]
    run = run[keep_cols].copy()
    run["model"] = model_name
    return run

# QL avec DirichletLM.
# Si cette ligne échoue avec un problème de fields, relance la cellule 6
# pour reconstruire l'index avec fields=True.
ql_model_used = "DirichletLM"
ql_pipe = pt.terrier.Retriever(
    index_ref,
    wmodel=ql_model_used,
    num_results=TOPK_RETRIEVAL
)
run_ql_raw = run_retriever("QL_DirichletLM", ql_pipe, topics_pt)

# SDM : rewrite SDM puis scoring avec le même modèle QL.
# SDM nécessite un index avec blocks=True.
try:
    sdm_pipe = pt.rewrite.SDM() >> pt.terrier.Retriever(
        index_ref,
        wmodel=ql_model_used,
        num_results=TOPK_RETRIEVAL
    )
    run_sdm_raw = run_retriever("SDM", sdm_pipe, topics_pt)
    sdm_model_used = f"SDM + {ql_model_used}"
except Exception as e:
    print("[ERREUR] SDM a échoué.")
    print("Cause probable : l'index n'a pas été reconstruit avec blocks=True.")
    print("Solution : relance la cellule 6 avec FORCE_REBUILD_INDEX=True, puis relance cette cellule.")
    raise e

print("QL utilisé :", ql_model_used)
print("SDM utilisé :", sdm_model_used)
print("run_ql_raw:", run_ql_raw.shape)
print("run_sdm_raw:", run_sdm_raw.shape)
print("Requêtes QL raw :", run_ql_raw["qid"].nunique())
print("Requêtes SDM raw :", run_sdm_raw["qid"].nunique())

# %%
# =====================================================================
# 9) Filtrer les runs sur les candidats top-50 par requête
# =====================================================================

candidate_pairs = df_top_passages[["qid", "passage_id"]].drop_duplicates().copy()
candidate_pairs["qid"] = candidate_pairs["qid"].astype(str)
candidate_pairs["passage_id"] = candidate_pairs["passage_id"].astype(str)

passage_meta = df_top_passages[[
    "qid", "passage_id", "passage_text", "passage_len_tokens",
    "relevance", "article_label_name", "label_any", "label_strong",
    "is_judged"
]].drop_duplicates(subset=["qid", "passage_id"]).copy()

# score initial du Notebook 1, utile pour l'interpolation optionnelle
initial_scores = df_top_passages[["qid", "passage_id", "score", "rank"]].rename(
    columns={"score": "initial_score", "rank": "initial_rank"}
).copy()

def restrict_and_enrich(run_raw: pd.DataFrame, model_name: str) -> pd.DataFrame:
    run = run_raw.merge(candidate_pairs, on=["qid", "passage_id"], how="inner")
    run = run.merge(passage_meta, on=["qid", "passage_id"], how="left", validate="many_to_one")
    run = run.merge(initial_scores, on=["qid", "passage_id"], how="left", validate="many_to_one")

    # Re-rank après filtrage sur candidats
    run = run.sort_values(["qid", "score"], ascending=[True, False]).copy()
    run["rank"] = run.groupby("qid").cumcount() + 1
    run = run[run["rank"] <= TOPK_EVAL].copy()
    run["model"] = model_name
    return run.reset_index(drop=True)

run_ql = restrict_and_enrich(run_ql_raw, "QL")
run_sdm = restrict_and_enrich(run_sdm_raw, "SDM")

print("Run QL filtré :", run_ql.shape)
print("Run SDM filtré :", run_sdm.shape)
print("Requêtes QL :", run_ql["qid"].nunique())
print("Requêtes SDM :", run_sdm["qid"].nunique())
display(run_ql.head())

# %%
# =====================================================================
# 10) Interpolation optionnelle QL + score initial
# =====================================================================
# Cette interpolation n'est PAS celle de l'article.
# Elle est seulement un diagnostic : on combine le score QL re-ranké avec
# le score initial SDM utilisé pour créer les candidats.

run_interp = run_ql.copy()

# Normalisation par requête pour éviter les échelles incompatibles
run_interp["ql_norm"] = run_interp.groupby("qid")["score"].transform(safe_minmax_normalize)
run_interp["initial_norm"] = run_interp.groupby("qid")["initial_score"].transform(safe_minmax_normalize)

run_interp["score"] = INTERP_ALPHA * run_interp["ql_norm"] + (1 - INTERP_ALPHA) * run_interp["initial_norm"]
run_interp = run_interp.sort_values(["qid", "score"], ascending=[True, False]).copy()
run_interp["rank"] = run_interp.groupby("qid").cumcount() + 1
run_interp["model"] = "QL_INITIAL_INTERP"

print("Run interpolation :", run_interp.shape)
display(run_interp.head())

# %% [markdown]
# ## 11) Évaluation passage-level
#
# On utilise les passages TREC-DL directement. Deux évaluations sont calculées :
#
# - **binaire forte** : `relevance >= 2`, proche de `excellent/perfect` ;
# - **graduée** : nDCG@10 avec les labels `0,1,2,3`.
#
# Les métriques principales proches de l'article sont : MAP, P@1, P@10.
#
#
# En plus, on ajoute une évaluation **caractère** : au lieu de compter chaque passage pertinent comme une unité, on compte ses caractères pertinents. C'est une approximation de Passage2 adaptée à TREC-DL Passage, où le passage entier est l'unité annotée.

# %%
# =====================================================================
# 11) Fonctions d'évaluation
# =====================================================================

def average_precision_binary(g: pd.DataFrame, rel_col="label_strong") -> float:
    g = g.sort_values("rank")
    num_rel_total = int(g[rel_col].sum())
    if num_rel_total == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, (_, row) in enumerate(g.iterrows(), start=1):
        if int(row[rel_col]) > 0:
            hits += 1
            precision_sum += hits / i
    return precision_sum / num_rel_total

def precision_at_k(g: pd.DataFrame, k: int, rel_col="label_strong") -> float:
    g = g.sort_values("rank").head(k)
    if len(g) == 0:
        return 0.0
    return float(g[rel_col].sum() / k)

def dcg_at_k(rels, k):
    rels = list(rels)[:k]
    return sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(rels))

def ndcg_at_k(g: pd.DataFrame, k=10, rel_col="relevance") -> float:
    g = g.sort_values("rank")
    dcg = dcg_at_k(g[rel_col].astype(int).tolist(), k)
    ideal = sorted(g[rel_col].astype(int).tolist(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)

def evaluate_run(run: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, dict]:
    per_q = []
    for qid, g in run.groupby("qid"):
        g = g.sort_values("rank").copy()
        per_q.append({
            "model": model_name,
            "qid": str(qid),
            "AP": average_precision_binary(g, "label_strong"),
            "P@1": precision_at_k(g, 1, "label_strong"),
            "P@10": precision_at_k(g, 10, "label_strong"),
            "nDCG@10": ndcg_at_k(g, 10, "relevance"),
            "num_retrieved": len(g),
            "num_rel_strong_in_run": int(g["label_strong"].sum()),
            "max_relevance_in_run": int(g["relevance"].max()) if len(g) else 0,
        })

    per_q_df = pd.DataFrame(per_q).sort_values(["model", "qid"]).reset_index(drop=True)
    summary = {
        "model": model_name,
        "MAP": float(per_q_df["AP"].mean()) if len(per_q_df) else 0.0,
        "P@1": float(per_q_df["P@1"].mean()) if len(per_q_df) else 0.0,
        "P@10": float(per_q_df["P@10"].mean()) if len(per_q_df) else 0.0,
        "nDCG@10": float(per_q_df["nDCG@10"].mean()) if len(per_q_df) else 0.0,
        "num_queries": int(per_q_df["qid"].nunique()) if len(per_q_df) else 0,
        "num_queries_with_rel_strong": int((per_q_df["num_rel_strong_in_run"] > 0).sum()) if len(per_q_df) else 0,
    }
    return per_q_df, summary

# %%
# =====================================================================
# 11bis) Évaluation character-level / Passage2-like
# =====================================================================
# Objectif : se rapprocher de l'article, où l'évaluation est faite au niveau caractère.
# Ici, TREC-DL annote des passages entiers. On approxime donc :
# - un passage fortement pertinent (relevance >= 2) contribue par tous ses caractères ;
# - un passage non pertinent contribue 0 caractère pertinent ;
# - les caractères d'un passage déjà récupéré ne sont pas recomptés.
#
# Cette métrique est plus stricte qu'une évaluation passage-level simple :
# elle pondère les passages par leur longueur, comme une version simplifiée de Passage2.


def _char_len(text) -> int:
    """Longueur caractère robuste. On garde les espaces, comme une approximation simple."""
    if pd.isna(text):
        return 0
    return len(str(text))


def prepare_gold_char_table(df_candidates: pd.DataFrame, rel_col="label_strong") -> pd.DataFrame:
    """
    Prépare la vérité terrain caractère au niveau des candidats top-50.
    On utilise les candidats top-50 du Notebook 1 comme espace évalué.
    """
    cols = ["qid", "passage_id", "passage_text", "relevance", rel_col]
    gold = df_candidates[cols].drop_duplicates(subset=["qid", "passage_id"]).copy()
    gold["qid"] = gold["qid"].astype(str)
    gold["passage_id"] = gold["passage_id"].astype(str)
    gold["char_len"] = gold["passage_text"].map(_char_len).astype(int)
    gold["rel_char_len"] = np.where(gold[rel_col].astype(int) > 0, gold["char_len"], 0)
    return gold


def character_average_precision(run_q: pd.DataFrame, gold_q: pd.DataFrame) -> float:
    """
    Average Precision au niveau caractère.

    Pour chaque caractère pertinent récupéré, on ajoute la précision caractère courante.
    Formule intuitive :
      AP_char = somme_precision_aux_caracteres_pertinents_recuperes / nb_caracteres_pertinents_total

    Contrairement à une AP passage-level, un passage long fortement pertinent pèse plus.
    """
    gold_meta = gold_q.set_index("passage_id")[["char_len", "rel_char_len"]].to_dict("index")
    total_rel_chars = int(gold_q["rel_char_len"].sum())
    if total_rel_chars <= 0:
        return 0.0

    retrieved_chars = 0
    retrieved_rel_chars = 0
    precision_sum = 0.0
    seen = set()

    run_q = run_q.sort_values("rank")
    for _, row in run_q.iterrows():
        pid = str(row["passage_id"])
        if pid in seen:
            continue
        seen.add(pid)

        meta = gold_meta.get(pid)
        if meta is None:
            # Passage hors espace gold : on le compte comme non pertinent avec longueur inconnue.
            # En pratique, cela ne devrait pas arriver après filtrage sur candidats.
            L = _char_len(row.get("passage_text", ""))
            rel_L = 0
        else:
            L = int(meta["char_len"])
            rel_L = int(meta["rel_char_len"])

        if L <= 0:
            continue

        # Si le passage est pertinent, chacun de ses rel_L caractères est un succès.
        # On calcule la somme exacte des précisions caractère par caractère.
        if rel_L > 0:
            # Dans notre approximation, si passage pertinent fort => tous ses caractères sont pertinents.
            for j in range(1, rel_L + 1):
                precision_sum += (retrieved_rel_chars + j) / (retrieved_chars + j)
            retrieved_rel_chars += rel_L

        retrieved_chars += L

    return float(precision_sum / total_rel_chars)


def character_precision_at_k(run_q: pd.DataFrame, gold_q: pd.DataFrame, k: int) -> float:
    """
    Précision caractère dans les k premiers passages :
      chars pertinents récupérés / chars totaux récupérés
    """
    gold_meta = gold_q.set_index("passage_id")[["char_len", "rel_char_len"]].to_dict("index")
    run_top = run_q.sort_values("rank").head(k)

    retrieved_chars = 0
    retrieved_rel_chars = 0
    seen = set()

    for _, row in run_top.iterrows():
        pid = str(row["passage_id"])
        if pid in seen:
            continue
        seen.add(pid)

        meta = gold_meta.get(pid)
        if meta is None:
            L = _char_len(row.get("passage_text", ""))
            rel_L = 0
        else:
            L = int(meta["char_len"])
            rel_L = int(meta["rel_char_len"])

        retrieved_chars += max(0, L)
        retrieved_rel_chars += max(0, rel_L)

    if retrieved_chars <= 0:
        return 0.0
    return float(retrieved_rel_chars / retrieved_chars)


def evaluate_run_character_level(run: pd.DataFrame, model_name: str, gold_char: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    per_q = []
    all_qids = sorted(gold_char["qid"].astype(str).unique())

    for qid in all_qids:
        g_run = run[run["qid"].astype(str) == qid].copy()
        g_gold = gold_char[gold_char["qid"].astype(str) == qid].copy()

        ap_char = character_average_precision(g_run, g_gold)
        p1_char = character_precision_at_k(g_run, g_gold, 1)
        p10_char = character_precision_at_k(g_run, g_gold, 10)

        per_q.append({
            "model": model_name,
            "qid": qid,
            "AP_char": ap_char,
            "CharP@1": p1_char,
            "CharP@10": p10_char,
            "num_gold_rel_chars": int(g_gold["rel_char_len"].sum()),
            "num_gold_rel_passages": int((g_gold["rel_char_len"] > 0).sum()),
            "num_retrieved": int(len(g_run)),
        })

    per_q_df = pd.DataFrame(per_q).sort_values(["model", "qid"]).reset_index(drop=True)
    summary = {
        "model": model_name,
        "MAP_char": float(per_q_df["AP_char"].mean()) if len(per_q_df) else 0.0,
        "CharP@1": float(per_q_df["CharP@1"].mean()) if len(per_q_df) else 0.0,
        "CharP@10": float(per_q_df["CharP@10"].mean()) if len(per_q_df) else 0.0,
        "num_queries": int(per_q_df["qid"].nunique()) if len(per_q_df) else 0,
        "num_queries_with_gold_rel_chars": int((per_q_df["num_gold_rel_chars"] > 0).sum()) if len(per_q_df) else 0,
    }
    return per_q_df, summary

# %%
# =====================================================================
# 12) Calcul des métriques passage-level + character-level
# =====================================================================

# 1) Métriques passage-level classiques
perq_ql, sum_ql = evaluate_run(run_ql, "QL")
perq_sdm, sum_sdm = evaluate_run(run_sdm, "SDM")
perq_interp, sum_interp = evaluate_run(run_interp, "QL_INITIAL_INTERP")

metrics_summary = pd.DataFrame([sum_ql, sum_sdm, sum_interp])
per_query_metrics = pd.concat([perq_ql, perq_sdm, perq_interp], ignore_index=True)

print("Résumé métriques passage-level :")
display(metrics_summary)

# 2) Métriques character-level / Passage2-like
# Gold sur les candidats top-50, avec pertinence forte relevance >= MIN_STRONG_RELEVANCE.
gold_char = prepare_gold_char_table(passage_meta, rel_col="label_strong")

perq_ql_char, sum_ql_char = evaluate_run_character_level(run_ql, "QL", gold_char)
perq_sdm_char, sum_sdm_char = evaluate_run_character_level(run_sdm, "SDM", gold_char)
perq_interp_char, sum_interp_char = evaluate_run_character_level(run_interp, "QL_INITIAL_INTERP", gold_char)

char_metrics_summary = pd.DataFrame([sum_ql_char, sum_sdm_char, sum_interp_char])
char_per_query_metrics = pd.concat([perq_ql_char, perq_sdm_char, perq_interp_char], ignore_index=True)

print("Résumé métriques character-level / Passage2-like :")
display(char_metrics_summary)

# 3) Tableau combiné pour lecture rapide
combined_metrics_summary = metrics_summary.merge(
    char_metrics_summary,
    on=["model", "num_queries"],
    how="left"
)

print("Résumé combiné :")
display(combined_metrics_summary)

print("Métriques par requête passage-level :")
display(per_query_metrics.head())

print("Métriques par requête character-level :")
display(char_per_query_metrics.head())

# %%
# =====================================================================
# 13) Diagnostics rapides
# =====================================================================

print("Distribution des labels dans QL top-50 :")
display(run_ql["relevance"].value_counts().sort_index())

print("Distribution des labels dans SDM top-50 :")
display(run_sdm["relevance"].value_counts().sort_index())

print("Requêtes sans pertinent fort dans le run QL :")
display(perq_ql[perq_ql["num_rel_strong_in_run"] == 0].head(20))

print("Top requêtes selon AP QL :")
display(perq_ql.sort_values("AP", ascending=False).head(10))

# %%
# =====================================================================
# 14) Exports
# =====================================================================

run_ql.to_csv(PATH_RUN_QL, sep="	", index=False)
run_sdm.to_csv(PATH_RUN_SDM, sep="	", index=False)
run_interp.to_csv(PATH_RUN_INTERP, sep="	", index=False)
metrics_summary.to_csv(PATH_METRICS_SUMMARY, sep="	", index=False)
per_query_metrics.to_csv(PATH_PER_QUERY_METRICS, sep="	", index=False)
char_metrics_summary.to_csv(PATH_CHAR_METRICS_SUMMARY, sep="	", index=False)
char_per_query_metrics.to_csv(PATH_CHAR_PER_QUERY_METRICS, sep="	", index=False)

metadata_m2 = {
    "project": "IR answer passage retrieval replication/adaptation",
    "notebook": "02_modeling_trec_dl_no_windows",
    "input_artifact_dir": str(ARTIFACT_DIR_NB1),
    "output_artifact_dir": str(ARTIFACT_DIR_M2),
    "unit_of_retrieval": "passage",
    "uses_windows": False,
    "index_options": {"fields": True, "blocks": True},
    "models": ["QL", "SDM", "QL_INITIAL_INTERP"],
    "ql_model_used": ql_model_used,
    "sdm_model_used": sdm_model_used,
    "topk_eval": TOPK_EVAL,
    "min_strong_relevance": MIN_STRONG_RELEVANCE,
    "label_mapping": {
        "0": "non_relevant",
        "1": "acceptable",
        "2": "excellent",
        "3": "perfect",
    },
    "metrics_passage_level": metrics_summary.to_dict(orient="records"),
    "metrics_character_level": char_metrics_summary.to_dict(orient="records"),
    "character_level_evaluation": "Passage2-like approximation: relevant passages contribute by all their characters; AP_char and CharP@k are computed over characters.",
    "note": (
        "This notebook evaluates direct TREC-DL passages and adds a character-level Passage2-like proxy. "
        "QL_INITIAL_INTERP is an optional diagnostic interpolation with the initial score, "
        "not the document/passage interpolation from the original paper."
    ),
}

PATH_METADATA_M2.write_text(json.dumps(metadata_m2, indent=2, ensure_ascii=False), encoding="utf-8")

print("Exports terminés dans :", ARTIFACT_DIR_M2)
for p in [PATH_RUN_QL, PATH_RUN_SDM, PATH_RUN_INTERP, PATH_METRICS_SUMMARY, PATH_PER_QUERY_METRICS, PATH_CHAR_METRICS_SUMMARY, PATH_CHAR_PER_QUERY_METRICS, PATH_METADATA_M2]:
    print("-", p.name, "OK" if p.exists() else "MANQUANT")

# %%
# =====================================================================
# 15) Backup Google Drive sécurisé
# =====================================================================
# À exécuter uniquement sur Colab si tu veux sauvegarder les résultats.

from google.colab import drive
from datetime import datetime

drive.mount('/content/drive')

DRIVE_BASE_DIR = Path("/content/drive/MyDrive/IR_project_backup")
RUN_NAME = f"trec_dl_no_windows_modeling_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DRIVE_RUN_DIR = DRIVE_BASE_DIR / RUN_NAME

expected = [
    PATH_RUN_QL,
    PATH_RUN_SDM,
    PATH_RUN_INTERP,
    PATH_METRICS_SUMMARY,
    PATH_PER_QUERY_METRICS,
    PATH_METADATA_M2,
]

missing = [str(p) for p in expected if not p.exists()]
if missing:
    raise FileNotFoundError("Backup refusé. Fichiers manquants :" + "".join(missing))

DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)

for item in ARTIFACT_DIR_M2.iterdir():
    src = item
    dst = DRIVE_RUN_DIR / item.name
    if item.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Sauvegarde terminée.")
print("Dossier Drive :", DRIVE_RUN_DIR)
for p in sorted(DRIVE_RUN_DIR.iterdir()):
    print(" -", p.name)