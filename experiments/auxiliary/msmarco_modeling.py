# Auxiliary IR experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # Notebook 2 — Modélisation et évaluation (version corrigée)
#
# Ce notebook charge les artefacts exportés par le **Notebook 1 corrigé** et exécute la partie **modelling** de façon plus propre et plus proche du papier.
#
# ## Ce que fait ce notebook
# 1. charge `queries_final.tsv`, `topdocs_top50.tsv`, `windows.tsv`, `silver_window_labels.tsv` et `silver_spans.tsv` ;
# 2. construit un **index PyTerrier sur les fenêtres** ;
# 3. exécute trois baselines :
#    - **QL** (DirichletLM sur les fenêtres),
#    - **SDM** (Sequential Dependence Model sur les fenêtres),
#    - **QL + interpolation doc/passage** avec un poids dépendant de la longueur du document ;
# 4. calcule une évaluation :
#    - **au niveau fenêtre** : MAP, P@1, P@10 ;
#    - **au niveau token/span** : approximation de Passage2 pour se rapprocher du papier ;
# 5. exporte des tableaux de résultats et des runs.
#
# ## Important
# - Les labels sont des **silver labels** issus du Notebook 1, pas des annotations humaines.
# - Les résultats sont donc **comparables qualitativement** au papier, mais pas comme réplication exacte.
# - L'évaluation *token/span* est une **approximation utile** du Passage2 MAP de l'article, pas une reproduction parfaite.

# %%
# =====================================================================
# 1) Installation des dépendances
# =====================================================================
# Décommente si besoin en Colab / Jupyter.
# NOTEBOOK: !pip -q install python-terrier ir_datasets pandas numpy tqdm

# %%
# =====================================================================
# 2) Imports, paramètres et chemins
# =====================================================================
import os
import re
import json
import math
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import pyterrier as pt

# Init PyTerrier
if hasattr(pt, "java"):
    if not pt.java.started():
        pt.init()
else:
    if not pt.started():
        pt.init()

# ---------------------------------------------------------------------
# Paramètres principaux
# ---------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# Dossier des artefacts issus du Notebook 1
WORKDIR = Path("/content/drive/MyDrive/IR_project_backup")
ARTIFACT_DIR = Path("/content/drive/MyDrive/IR_project_backup/notebook1_backup_20260425_201003")

# Fichiers d'entrée
PATH_QUERIES = ARTIFACT_DIR / "queries_final.tsv"
PATH_TOPDOCS = ARTIFACT_DIR / "topdocs_top50.tsv"
PATH_SPANS = ARTIFACT_DIR / "silver_spans.tsv"
PATH_WINDOWS = ARTIFACT_DIR / "windows.tsv"
PATH_WINDOW_LABELS = ARTIFACT_DIR / "silver_window_labels.tsv"
PATH_METADATA = ARTIFACT_DIR / "dataset_metadata.json"

# Dossier de sortie pour ce notebook
ARTIFACT_DIR_M2 = WORKDIR / "artifacts_notebook2"
ARTIFACT_DIR_M2.mkdir(parents=True, exist_ok=True)

# Index des fenêtres
WINDOW_INDEX_PATH = ARTIFACT_DIR_M2 / "pt_windows_index"

# Hyperparamètres de ranking
MU_DIRICHLET = 1000
TOPK_WINDOWS = 100
DOC_INTERP_C = 150.0   # contrôle le poids document dans l'interpolation dépendante de la longueur

# Seuil de pertinence binaire pour l'évaluation fenêtre.
# Dans le papier, seuls perfect/excellent sont considérés pertinents.
# Ici, on suit la même logique avec silver_label >= 3.
MIN_RELEVANT_LABEL = 3

# Fichiers de sortie
PATH_RUN_QL = ARTIFACT_DIR_M2 / "run_ql.tsv"
PATH_RUN_SDM = ARTIFACT_DIR_M2 / "run_sdm.tsv"
PATH_RUN_INTERP = ARTIFACT_DIR_M2 / "run_interp.tsv"
PATH_METRICS_SUMMARY = ARTIFACT_DIR_M2 / "metrics_summary.tsv"
PATH_PER_QUERY_METRICS = ARTIFACT_DIR_M2 / "per_query_metrics.tsv"
PATH_METADATA_M2 = ARTIFACT_DIR_M2 / "modeling_metadata.json"

print("Notebook 2 - Dossier d'entrée :", ARTIFACT_DIR)
print("Notebook 2 - Dossier de sortie :", ARTIFACT_DIR_M2)

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# =====================================================================
# 3) Helpers généraux
# =====================================================================
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

def norm_qid(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def clean_query(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def make_docno(qid, doc_id, window_id):
    return f"{qid}__{doc_id}__w{window_id}"

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def binary_ap(y_true_ranked):
    """Average Precision binaire sur une liste déjà ordonnée du meilleur au pire."""
    num_rel = sum(1 for y in y_true_ranked if y > 0)
    if num_rel == 0:
        return 0.0
    hit = 0
    ap = 0.0
    for i, y in enumerate(y_true_ranked, start=1):
        if y > 0:
            hit += 1
            ap += hit / i
    return ap / num_rel

def precision_at_k(y_true_ranked, k):
    if k <= 0:
        return 0.0
    prefix = y_true_ranked[:k]
    if len(prefix) == 0:
        return 0.0
    return sum(1 for y in prefix if y > 0) / min(k, len(prefix))

def build_char_intervals(df_labels_one_qid):
    """Retourne les intervalles (span_start, span_end) gold pour un qid.
    On travaille au niveau token, ce qui donne une approximation du Passage2 MAP.
    """
    intervals = []
    seen = set()
    for _, r in df_labels_one_qid.iterrows():
        if pd.isna(r["matched_span_start"]) or pd.isna(r["matched_span_end"]):
            continue
        a = int(r["matched_span_start"])
        b = int(r["matched_span_end"])
        if b <= a:
            continue
        key = (a, b)
        if key not in seen:
            intervals.append(key)
            seen.add(key)
    return intervals

def interval_overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))

def token_level_ap(run_qid_df, gold_intervals):
    """Approximation token/span de Passage2 MAP.
    Chaque token pertinent est compté au plus une fois, même s'il est récupéré plusieurs fois.
    """
    total_rel = sum(max(0, b - a) for a, b in gold_intervals)
    if total_rel == 0 or len(run_qid_df) == 0:
        return 0.0

    covered = set()
    ap_sum = 0.0
    seen_relevant = 0

    for rank_idx, (_, row) in enumerate(run_qid_df.iterrows(), start=1):
        w_start = int(row["window_start_token"])
        w_end = int(row["window_end_token"])

        newly_covered = 0
        for a, b in gold_intervals:
            ov_start = max(w_start, a)
            ov_end = min(w_end, b)
            if ov_end > ov_start:
                for tok in range(ov_start, ov_end):
                    if tok not in covered:
                        covered.add(tok)
                        newly_covered += 1

        if newly_covered > 0:
            seen_relevant += newly_covered
            precision_here = seen_relevant / rank_idx
            ap_sum += newly_covered * precision_here

    return ap_sum / total_rel if total_rel > 0 else 0.0

def token_level_precision_at_k(run_qid_df, gold_intervals, k):
    if len(run_qid_df) == 0 or k <= 0:
        return 0.0

    covered = set()
    prefix = run_qid_df.head(k)
    for _, row in prefix.iterrows():
        w_start = int(row["window_start_token"])
        w_end = int(row["window_end_token"])
        for a, b in gold_intervals:
            ov_start = max(w_start, a)
            ov_end = min(w_end, b)
            if ov_end > ov_start:
                for tok in range(ov_start, ov_end):
                    covered.add(tok)

    total_retrieved_tokens = int((prefix["window_end_token"] - prefix["window_start_token"]).sum())
    return len(covered) / total_retrieved_tokens if total_retrieved_tokens > 0 else 0.0

def qrels_for_pyterrier(df_labels, min_rel_label=3):
    df = df_labels.copy()
    df["qid"] = df["qid"].map(norm_qid).astype(str)
    df["docno"] = df.apply(lambda r: make_docno(r["qid"], r["doc_id"], r["window_id"]), axis=1)
    df["label_bin"] = (df["silver_label"] >= min_rel_label).astype(int)
    qrels = df.loc[df["label_bin"] > 0, ["qid", "docno", "label_bin"]].rename(columns={"label_bin": "label"})
    return qrels

# %%
# =====================================================================
# 4) Chargement des artefacts du Notebook 1
# =====================================================================
required_files = [
    PATH_QUERIES, PATH_TOPDOCS, PATH_SPANS, PATH_WINDOWS, PATH_WINDOW_LABELS, PATH_METADATA
]
missing = [str(p) for p in required_files if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Fichiers manquants. Exécute d'abord le Notebook 1 corrigé.\n" + "\n".join(missing)
    )

topics = pd.read_csv(PATH_QUERIES, sep="\t")
df_topdocs = pd.read_csv(PATH_TOPDOCS, sep="\t")
df_spans = pd.read_csv(PATH_SPANS, sep="\t")
df_windows = pd.read_csv(PATH_WINDOWS, sep="\t")
df_window_labels = pd.read_csv(PATH_WINDOW_LABELS, sep="\t")
metadata_nb1 = json.loads(PATH_METADATA.read_text())

# Normalisation légère
topics["qid"] = topics["qid"].map(norm_qid).astype(str)
df_topdocs["qid"] = df_topdocs["qid"].map(norm_qid).astype(str)
df_windows["qid"] = df_windows["qid"].map(norm_qid).astype(str)
df_window_labels["qid"] = df_window_labels["qid"].map(norm_qid).astype(str)
df_spans["qid"] = df_spans["qid"].map(norm_qid).astype(str)

df_topdocs["doc_id"] = df_topdocs["doc_id"].astype(str)
df_windows["doc_id"] = df_windows["doc_id"].astype(str)
df_window_labels["doc_id"] = df_window_labels["doc_id"].astype(str)
df_spans["doc_id"] = df_spans["doc_id"].astype(str)

print("Nb requêtes :", topics['qid'].nunique())
print("Nb topdocs :", len(df_topdocs))
print("Nb fenêtres :", len(df_windows))
print("Nb labels fenêtres :", len(df_window_labels))
print("Distribution silver_label :")
display(df_window_labels["silver_label"].value_counts().sort_index())

# Fusionner les labels dans les fenêtres
df_data = df_windows.merge(
    df_window_labels,
    on=["qid", "doc_id", "window_id"],
    how="left",
    validate="one_to_one"
)

# Ajout docno, longueur doc estimée et binaire de pertinence
df_data["docno"] = df_data.apply(lambda r: make_docno(r["qid"], r["doc_id"], r["window_id"]), axis=1)
doc_len_by_qid_doc = (
    df_data.groupby(["qid", "doc_id"])["window_end_token"].max().reset_index(name="doc_len_tokens")
)
df_data = df_data.merge(doc_len_by_qid_doc, on=["qid", "doc_id"], how="left")
df_data["label_bin"] = (df_data["silver_label"] >= MIN_RELEVANT_LABEL).astype(int)

print("Nb fenêtres binaires positives :", int(df_data["label_bin"].sum()))
display(df_data.head())

# %%
# =====================================================================
# 5) Construction / chargement de l'index PyTerrier des fenêtres
# =====================================================================
# On indexe toutes les fenêtres une seule fois.
# Le docno contient qid + doc_id + window_id pour pouvoir filtrer strictement par requête ensuite.

def iter_windows_for_index(df):
    for _, r in df.iterrows():
        yield {
            "docno": str(r["docno"]),
            "text": str(r["window_text"]),
            "qid_meta": str(r["qid"]),
            "doc_id_meta": str(r["doc_id"]),
            "window_id_meta": str(r["window_id"]),
        }

if not WINDOW_INDEX_PATH.exists():
    WINDOW_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    print("Construction de l'index fenêtres...")
    indexer = pt.IterDictIndexer(
        str(WINDOW_INDEX_PATH),
        meta={"docno": 128, "qid_meta": 32, "doc_id_meta": 64, "window_id_meta": 16},
        overwrite=True,
        blocks=True,
        threads=1,
    )
    indexref_windows = indexer.index(iter_windows_for_index(df_data))
else:
    print("Index fenêtres déjà présent :", WINDOW_INDEX_PATH)
    indexref_windows = str(WINDOW_INDEX_PATH)

windows_index = pt.IndexFactory.of(indexref_windows)
print("Index fenêtres prêt.")

# %%
# =====================================================================
# 6) Préparer les topics et les qrels PyTerrier
# =====================================================================
# On préfère utiliser la requête brute issue du topdocs quand disponible ; sinon queries_final.
if "query_raw" in df_topdocs.columns:
    qid2query = (
        df_topdocs[["qid", "query_raw"]]
        .drop_duplicates("qid")
        .rename(columns={"query_raw": "query"})
        .set_index("qid")["query"]
        .to_dict()
    )
else:
    qid2query = topics.set_index("qid")["query"].to_dict()

topics_pt = pd.DataFrame({
    "qid": sorted(df_data["qid"].unique()),
    "query": [clean_query(qid2query[qid]) for qid in sorted(df_data["qid"].unique())]
})

# Qrels binaires au niveau fenêtre pour l'évaluation classique
qrels_pt = qrels_for_pyterrier(df_window_labels, min_rel_label=MIN_RELEVANT_LABEL)

print("Nb topics pour le ranking :", len(topics_pt))
print("Nb fenêtres pertinentes dans les qrels binaires :", len(qrels_pt))
display(topics_pt.head())
display(qrels_pt.head())

# %%
# =====================================================================
# 7) Ranking baseline 1 : Query Likelihood (DirichletLM)
# =====================================================================
# On utilise un filtre strict sur le préfixe qid__ pour éviter qu'une requête puisse scorer
# des fenêtres construites pour une autre requête.
def filter_run_to_own_qid(run_df):
    run_df = run_df.copy()
    run_df["qid"] = run_df["qid"].astype(str)
    run_df["docno"] = run_df["docno"].astype(str)
    run_df = run_df[run_df.apply(lambda r: str(r["docno"]).startswith(str(r["qid"]) + "__"), axis=1)].copy()
    return run_df

ql_retr = pt.terrier.Retriever(
    windows_index,
    wmodel="DirichletLM",
    controls={"mu": str(MU_DIRICHLET)},
    num_results=TOPK_WINDOWS,
)

run_ql = ql_retr.transform(topics_pt.copy())
run_ql = filter_run_to_own_qid(run_ql)
run_ql = run_ql.sort_values(["qid", "score"], ascending=[True, False]).groupby("qid").head(TOPK_WINDOWS).reset_index(drop=True)

print("Run QL :", len(run_ql))
display(run_ql.head())

# %%
# =====================================================================
# 8) Ranking baseline 2 : SDM sur les fenêtres
# =====================================================================
# SDM peut ne pas être disponible selon l'environnement PyTerrier/Terrier.
# On garde un fallback explicite.
sdm_mode_used = None

try:
    sdm_pipe = pt.rewrite.SDM() >> pt.terrier.Retriever(
        windows_index,
        wmodel="DirichletLM",
        controls={"mu": str(MU_DIRICHLET)},
        num_results=TOPK_WINDOWS,
    )
    run_sdm = sdm_pipe.transform(topics_pt.copy())
    sdm_mode_used = "SDM + DirichletLM"
except Exception as e:
    print("[WARN] SDM indisponible. Fallback vers DPH.")
    print("Erreur SDM :", repr(e))
    sdm_pipe = pt.terrier.Retriever(
        windows_index,
        wmodel="DPH",
        num_results=TOPK_WINDOWS,
    )
    run_sdm = sdm_pipe.transform(topics_pt.copy())
    sdm_mode_used = "DPH fallback"

run_sdm = filter_run_to_own_qid(run_sdm)
run_sdm = run_sdm.sort_values(["qid", "score"], ascending=[True, False]).groupby("qid").head(TOPK_WINDOWS).reset_index(drop=True)

print("Run SDM :", len(run_sdm))
print("Mode utilisé :", sdm_mode_used)
display(run_sdm.head())

# %%
# =====================================================================
# 9) Ranking baseline 3 : QL + interpolation doc/passage plus fidèle au papier
# =====================================================================
# Idée :
# score_interp = (1 - lambda_d) * score_passage + lambda_d * score_document
# avec lambda_d qui diminue lorsque le document est long.
#
# Cela est plus proche du papier que l'ancienne interpolation min-max à alpha fixe.

doc_scores = (
    df_topdocs[["qid", "doc_id", "score"]]
    .rename(columns={"score": "doc_score"})
    .drop_duplicates(["qid", "doc_id"])
)

run_interp = run_ql.copy()
run_interp = run_interp.merge(
    df_data[["qid", "docno", "doc_id", "doc_len_tokens", "window_start_token", "window_end_token"]],
    on=["qid", "docno"],
    how="left",
    validate="one_to_one"
)
run_interp = run_interp.merge(
    doc_scores,
    on=["qid", "doc_id"],
    how="left",
    validate="many_to_one"
)

# Normalisation par requête des scores passage et document
def minmax_by_group(series):
    mn = series.min()
    mx = series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)

run_interp["passage_score_norm"] = run_interp.groupby("qid")["score"].transform(minmax_by_group)
run_interp["doc_score_norm"] = run_interp.groupby("qid")["doc_score"].transform(minmax_by_group)

# Poids document dépendant de la longueur du document
run_interp["lambda_doc"] = DOC_INTERP_C / (DOC_INTERP_C + run_interp["doc_len_tokens"].clip(lower=1))
run_interp["score_interp"] = (
    (1.0 - run_interp["lambda_doc"]) * run_interp["passage_score_norm"] +
    run_interp["lambda_doc"] * run_interp["doc_score_norm"]
)

run_interp = run_interp.sort_values(["qid", "score_interp"], ascending=[True, False]).reset_index(drop=True)
run_interp["rank"] = run_interp.groupby("qid").cumcount()
run_interp["score"] = run_interp["score_interp"]

print("Run interpolé :", len(run_interp))
display(run_interp.head())

# %%
# =====================================================================
# 10) Enrichir les runs avec les labels et métadonnées nécessaires
# =====================================================================
# Cette cellule prépare les runs pour l'évaluation classique et l'évaluation token/span.

keep_cols = [
    "qid", "docno", "score", "rank",
]
meta_cols = [
    "qid", "docno", "doc_id", "window_id",
    "window_start_token", "window_end_token", "window_len",
    "silver_label", "label_bin",
    "matched_span_start", "matched_span_end"
]

meta_lookup = df_data[meta_cols].drop_duplicates(["qid", "docno"])

def attach_meta(run_df):
    out = run_df.merge(meta_lookup, on=["qid", "docno"], how="left")
    out = out.sort_values(["qid", "score"], ascending=[True, False]).reset_index(drop=True)
    out["rank"] = out.groupby("qid").cumcount() + 1
    return out

run_ql_eval = attach_meta(run_ql)
run_sdm_eval = attach_meta(run_sdm)
run_interp_eval = attach_meta(run_interp)

display(run_ql_eval.head())

# %%
# =====================================================================
# 11) Évaluation au niveau fenêtre (MAP, P@1, P@10)
# =====================================================================
# On calcule les métriques à la main pour garder un contrôle total sur le protocole.
# Cela évite aussi les ambiguïtés liées aux runs filtrés par qid.

def evaluate_window_run(run_eval_df, qrels_bin_df):
    gold_by_qid = defaultdict(set)
    for _, r in qrels_bin_df.iterrows():
        gold_by_qid[str(r["qid"])].add(str(r["docno"]))

    per_q = []
    for qid, g in run_eval_df.groupby("qid"):
        ranked_docnos = list(g.sort_values("rank")["docno"].astype(str))
        y_true = [1 if d in gold_by_qid[str(qid)] else 0 for d in ranked_docnos]
        ap = binary_ap(y_true)
        p1 = precision_at_k(y_true, 1)
        p10 = precision_at_k(y_true, 10)
        per_q.append({
            "qid": str(qid),
            "AP_window": ap,
            "P@1_window": p1,
            "P@10_window": p10,
            "num_retrieved": len(y_true),
            "num_gold_window": len(gold_by_qid[str(qid)]),
        })

    per_q_df = pd.DataFrame(per_q).sort_values("qid").reset_index(drop=True)
    summary = {
        "MAP_window": float(per_q_df["AP_window"].mean()) if len(per_q_df) else 0.0,
        "P@1_window": float(per_q_df["P@1_window"].mean()) if len(per_q_df) else 0.0,
        "P@10_window": float(per_q_df["P@10_window"].mean()) if len(per_q_df) else 0.0,
    }
    return per_q_df, summary

perq_ql_win, sum_ql_win = evaluate_window_run(run_ql_eval, qrels_pt)
perq_sdm_win, sum_sdm_win = evaluate_window_run(run_sdm_eval, qrels_pt)
perq_interp_win, sum_interp_win = evaluate_window_run(run_interp_eval, qrels_pt)

print("QL window metrics:", sum_ql_win)
print("SDM window metrics:", sum_sdm_win)
print("INTERP window metrics:", sum_interp_win)

# %%
# =====================================================================
# Recréer proprement les runs d'évaluation avec les métadonnées fenêtres
# =====================================================================

window_meta_cols = [
    "qid",
    "docno",
    "doc_id",
    "window_id",
    "window_start_token",
    "window_end_token",
]
window_meta = df_data[window_meta_cols].drop_duplicates().copy()

# On repart des runs bruts, pas des *_eval déjà modifiés
run_ql_eval = run_ql.merge(
    window_meta,
    on=["qid", "docno"],
    how="left",
    validate="many_to_one"
)

run_sdm_eval = run_sdm.merge(
    window_meta,
    on=["qid", "docno"],
    how="left",
    validate="many_to_one"
)

run_interp_eval = run_interp.merge(
    window_meta,
    on=["qid", "docno"],
    how="left",
    validate="many_to_one"
)

for name, df_run in [
    ("QL", run_ql_eval),
    ("SDM", run_sdm_eval),
    ("INTERP", run_interp_eval),
]:
    print(f"\n{name}")
    print("Colonnes clés présentes :")
    for c in ["qid", "docno", "doc_id", "window_id", "window_start_token", "window_end_token"]:
        print(f" - {c}: {c in df_run.columns}")
    if "window_start_token" in df_run.columns:
        print("Lignes sans métadonnées fenêtre :", int(df_run["window_start_token"].isna().sum()))

# %%
# =====================================================================
# Réparer run_interp_eval si les colonnes sont suffixées (_x / _y)
# =====================================================================

print("Colonnes actuelles de run_interp_eval :")
print(run_interp_eval.columns.tolist())

def coalesce_columns(df, target, candidates):
    """
    Crée/écrase la colonne `target` en prenant la première colonne existante
    parmi `candidates`.
    """
    for c in candidates:
        if c in df.columns:
            df[target] = df[c]
            return df
    return df

# On reconstruit les colonnes standard si elles existent sous forme suffixée
run_interp_eval = coalesce_columns(run_interp_eval, "doc_id", [
    "doc_id", "doc_id_y", "doc_id_x"
])

run_interp_eval = coalesce_columns(run_interp_eval, "window_start_token", [
    "window_start_token", "window_start_token_y", "window_start_token_x"
])

run_interp_eval = coalesce_columns(run_interp_eval, "window_end_token", [
    "window_end_token", "window_end_token_y", "window_end_token_x"
])

run_interp_eval = coalesce_columns(run_interp_eval, "window_id", [
    "window_id", "window_id_y", "window_id_x"
])

print("\nAprès réparation :")
for c in ["qid", "docno", "doc_id", "window_id", "window_start_token", "window_end_token"]:
    print(f"{c:20s} -> {c in run_interp_eval.columns}")

if "window_start_token" in run_interp_eval.columns:
    print("\nLignes sans window_start_token :", int(run_interp_eval["window_start_token"].isna().sum()))
if "window_end_token" in run_interp_eval.columns:
    print("Lignes sans window_end_token   :", int(run_interp_eval["window_end_token"].isna().sum()))

# %%
# =====================================================================
# 12) Évaluation token/span (approximation de Passage2)
# =====================================================================
# On approxime Passage2 en travaillant au niveau token :
# - les spans gold proviennent des silver spans associés aux fenêtres pertinentes ;
# - un token pertinent n'est compté qu'une seule fois même si plusieurs fenêtres le couvrent.

def evaluate_token_run(run_eval_df, df_labels):
    # On ne garde pour la vérité terrain que les fenêtres "pertinentes" au sens fort.
    rel_labels = df_labels[df_labels["silver_label"] >= MIN_RELEVANT_LABEL].copy()

    per_q = []
    for qid, g_run in run_eval_df.groupby("qid"):
        qid = str(qid)
        g_run = g_run.sort_values("rank").copy()
        g_gold = rel_labels[rel_labels["qid"].astype(str) == qid].copy()

        gold_intervals = build_char_intervals(g_gold)
        ap_tok = token_level_ap(g_run, gold_intervals)
        p1_tok = token_level_precision_at_k(g_run, gold_intervals, 1)
        p10_tok = token_level_precision_at_k(g_run, gold_intervals, 10)

        per_q.append({
            "qid": qid,
            "AP_token": ap_tok,
            "P@1_token": p1_tok,
            "P@10_token": p10_tok,
            "num_gold_intervals": len(gold_intervals),
            "num_gold_tokens": int(sum(max(0, b - a) for a, b in gold_intervals)),
        })

    per_q_df = pd.DataFrame(per_q).sort_values("qid").reset_index(drop=True)
    summary = {
        "MAP_token": float(per_q_df["AP_token"].mean()) if len(per_q_df) else 0.0,
        "P@1_token": float(per_q_df["P@1_token"].mean()) if len(per_q_df) else 0.0,
        "P@10_token": float(per_q_df["P@10_token"].mean()) if len(per_q_df) else 0.0,
    }
    return per_q_df, summary

perq_ql_tok, sum_ql_tok = evaluate_token_run(run_ql_eval, df_window_labels)
perq_sdm_tok, sum_sdm_tok = evaluate_token_run(run_sdm_eval, df_window_labels)
perq_interp_tok, sum_interp_tok = evaluate_token_run(run_interp_eval, df_window_labels)

print("QL token metrics:", sum_ql_tok)
print("SDM token metrics:", sum_sdm_tok)
print("INTERP token metrics:", sum_interp_tok)

# %%
# =====================================================================
# 13) Tableau final et exports
# =====================================================================
summary_rows = [
    {
        "model": "QL",
        **sum_ql_win,
        **sum_ql_tok,
    },
    {
        "model": "SDM",
        **sum_sdm_win,
        **sum_sdm_tok,
    },
    {
        "model": "QL_doc_length_interpolated",
        **sum_interp_win,
        **sum_interp_tok,
    },
]
df_summary = pd.DataFrame(summary_rows)

per_query_merged = (
    perq_ql_win.merge(perq_ql_tok, on="qid", how="outer", suffixes=("_ql_win", "_ql_tok"))
    .merge(perq_sdm_win, on="qid", how="outer", suffixes=("", "_sdm_win"))
    .merge(perq_sdm_tok, on="qid", how="outer", suffixes=("", "_sdm_tok"))
    .merge(perq_interp_win, on="qid", how="outer", suffixes=("", "_interp_win"))
    .merge(perq_interp_tok, on="qid", how="outer", suffixes=("", "_interp_tok"))
)

# Sauvegarde des runs
run_ql_eval.to_csv(PATH_RUN_QL, sep="\t", index=False)
run_sdm_eval.to_csv(PATH_RUN_SDM, sep="\t", index=False)
run_interp_eval.to_csv(PATH_RUN_INTERP, sep="\t", index=False)

# Sauvegarde des métriques
df_summary.to_csv(PATH_METRICS_SUMMARY, sep="\t", index=False)
per_query_merged.to_csv(PATH_PER_QUERY_METRICS, sep="\t", index=False)

metadata_m2 = {
    "seed": SEED,
    "mu_dirichlet": MU_DIRICHLET,
    "topk_windows": TOPK_WINDOWS,
    "doc_interp_c": DOC_INTERP_C,
    "min_relevant_label": MIN_RELEVANT_LABEL,
    "sdm_mode_used": sdm_mode_used,
    "notes": [
        "QL = DirichletLM sur les fenêtres.",
        "SDM appliqué sur l'index de fenêtres ; fallback explicite si indisponible.",
        "Interpolation doc/passage avec poids dépendant de la longueur du document.",
        "Évaluation fenêtre binaire : silver_label >= 3.",
        "Évaluation token/span : approximation de Passage2 MAP."
    ],
    "inputs": {
        "queries": str(PATH_QUERIES),
        "topdocs": str(PATH_TOPDOCS),
        "spans": str(PATH_SPANS),
        "windows": str(PATH_WINDOWS),
        "window_labels": str(PATH_WINDOW_LABELS),
        "nb1_metadata": str(PATH_METADATA),
    },
    "outputs": {
        "run_ql": str(PATH_RUN_QL),
        "run_sdm": str(PATH_RUN_SDM),
        "run_interp": str(PATH_RUN_INTERP),
        "metrics_summary": str(PATH_METRICS_SUMMARY),
        "per_query_metrics": str(PATH_PER_QUERY_METRICS),
    },
}
PATH_METADATA_M2.write_text(json.dumps(metadata_m2, ensure_ascii=False, indent=2))

print("Exports enregistrés dans :", ARTIFACT_DIR_M2)
display(df_summary)

# %%
Measure MAP P@1 P@10
QL 0.021 0.148 0.057
SDM 0.020 0.107 0.060
QL-Interpolated 0.022 0.073 0.062

# %% [markdown]
# ## Interprétation recommandée
#
# - **MAP/P@1/P@10 au niveau fenêtre** : utiles pour comparer rapidement les baselines entre elles.
# - **MAP/P@1/P@10 au niveau token/span** : plus proches de l'esprit du papier, car moins sensibles au découpage arbitraire en fenêtres.
# - **QL_doc_length_interpolated** : ce n'est pas une copie exacte de la formule de l'article, mais c'est une version **beaucoup plus fidèle** que l'ancienne interpolation min-max à alpha fixe.
#
# ## Fichiers produits
# - `run_ql.tsv`
# - `run_sdm.tsv`
# - `run_interp.tsv`
# - `metrics_summary.tsv`
# - `per_query_metrics.tsv`
# - `modeling_metadata.json`