# Auxiliary IR experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # 02 — Modeling FiRA TREC-19 avec fenêtres + évaluation caractère
#
# Ce notebook est adapté aux données que nous avons créées dans Drive :
#
# - dossier de données : `/content/drive/MyDrive/IR_project_backup/fira_trec19_article_like_with_windows`
# - dossier de sortie modeling : `/content/drive/MyDrive/IR_project_backup/fira_trec19_modeling_char_eval_with_windows`
# - unité évaluée : **fenêtre 50/25**
# - labels : **labels FiRA directs** (`fira_label` si présent, sinon alias compatible `passage_relevance`).
#
# La version ne dépend plus de `reconstructed_docs.tsv`. Pour l'évaluation character-level, on utilise directement la longueur en caractères de `unit_text` / `window_text` / `passage_text`.

# %%
# =====================================================================
# 1) Installation des dépendances
# =====================================================================
# NOTEBOOK: !pip -q install python-terrier pandas numpy tqdm

# %%
# =====================================================================
# 2) Monter Google Drive
# =====================================================================
from google.colab import drive
drive.mount('/content/drive')

# %%
# =====================================================================
# 3) Imports, paramètres et chemins
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

# Initialisation PyTerrier robuste selon la version.
if hasattr(pt, 'java'):
    if not pt.java.started():
        pt.java.init()
else:
    if not pt.started():
        pt.init()

SEED = 42
np.random.seed(SEED)

DATA_VARIANT = 'with_windows'
USE_WINDOWS = True

TOPK_RETRIEVAL = 1000
TOPK_EVAL = 50
MIN_STRONG_RELEVANCE = 2
INTERP_ALPHA = 0.75
FORCE_REBUILD_INDEX = True

DRIVE_ROOT = Path('/content/drive/MyDrive/IR_project_backup')
DATA_DIR = DRIVE_ROOT / 'fira_trec19_article_like_with_windows'
MODEL_OUT_DIR = DRIVE_ROOT / 'fira_trec19_modeling_char_eval_with_windows'
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIT_INDEX_DIR = MODEL_OUT_DIR / 'pt_fira_unit_index'

PATH_QUERIES = DATA_DIR / 'queries_final.tsv'
PATH_TOPDOCS = DATA_DIR / 'topdocs_top50.tsv'
PATH_RELEVANT_TOPDOCS = DATA_DIR / 'relevant_topdocs.tsv'
PATH_METADATA_DATA = DATA_DIR / 'dataset_metadata.json'
PATH_RECON_DOCS = DATA_DIR / 'fira_reconstructed_documents.tsv'  # seulement pour information, pas obligatoire

# Fichiers spécifiques selon la variante.
PATH_WINDOWS = DATA_DIR / 'windows.tsv'
PATH_WINDOW_LABELS = DATA_DIR / 'window_labels.tsv'
PATH_PASSAGE_UNITS = DATA_DIR / 'passage_units.tsv'

# Exports modeling.
PATH_MODELING_UNITS = MODEL_OUT_DIR / 'modeling_units.tsv'
PATH_RUN_QL = MODEL_OUT_DIR / 'run_ql.tsv'
PATH_RUN_SDM = MODEL_OUT_DIR / 'run_sdm.tsv'
PATH_RUN_INTERP = MODEL_OUT_DIR / 'run_ql_initial_interp.tsv'
PATH_METRICS_SUMMARY = MODEL_OUT_DIR / 'metrics_summary.tsv'
PATH_PER_QUERY_METRICS = MODEL_OUT_DIR / 'per_query_metrics.tsv'
PATH_CHAR_METRICS_SUMMARY = MODEL_OUT_DIR / 'char_metrics_summary.tsv'
PATH_CHAR_PER_QUERY_METRICS = MODEL_OUT_DIR / 'char_per_query_metrics.tsv'
PATH_METADATA_MODEL = MODEL_OUT_DIR / 'modeling_metadata.json'

print('[DATA] Dossier utilisé :', DATA_DIR)
print('[OUT ] Dossier sortie  :', MODEL_OUT_DIR)
print('[LINKS] Données créées avant :')
for p in [PATH_QUERIES, PATH_WINDOWS, PATH_WINDOW_LABELS, PATH_PASSAGE_UNITS, PATH_TOPDOCS, PATH_RELEVANT_TOPDOCS, PATH_RECON_DOCS, PATH_METADATA_DATA]:
    if p.exists():
        print(' -', p)

# %%
# =====================================================================
# 4) Helpers
# =====================================================================

def read_tsv(path, **kwargs):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')
    return pd.read_csv(path, sep='	', dtype=str, keep_default_na=False, **kwargs)


def save_tsv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep='	', index=False)


def pick_col(df, candidates, required=True, name='column'):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f'Impossible de trouver {name}. Colonnes disponibles: {list(df.columns)}')
    return None


def to_int_series(s, default=0):
    return pd.to_numeric(s, errors='coerce').fillna(default).astype(int)


def to_float_series(s, default=0.0):
    return pd.to_numeric(s, errors='coerce').fillna(default).astype(float)


def safe_minmax_normalize(s):
    s = pd.to_numeric(s, errors='coerce').fillna(0.0).astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def relevance_name(rel):
    rel = int(rel)
    return {
        0: 'non_relevant',
        1: 'fair_or_topic_relevant',
        2: 'good_answer',
        3: 'perfect_answer',
    }.get(rel, f'label_{rel}')


def clean_query(q):
    q = str(q)
    q = re.sub(r'[^A-Za-z0-9_\-\s]', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()
    return q

# %%
# =====================================================================
# 5) Vérification des fichiers d'entrée
# =====================================================================
required = [PATH_QUERIES]
if USE_WINDOWS:
    required += [PATH_WINDOWS, PATH_WINDOW_LABELS]
else:
    required += [PATH_PASSAGE_UNITS]

rows = []
for p in required + [PATH_TOPDOCS, PATH_RELEVANT_TOPDOCS, PATH_RECON_DOCS, PATH_METADATA_DATA]:
    rows.append({
        'file': p.name,
        'exists': p.exists(),
        'size_MB': round(p.stat().st_size / (1024 * 1024), 3) if p.exists() else 0,
        'path': str(p),
    })
check_df = pd.DataFrame(rows)
display(check_df)

missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError('Fichiers obligatoires manquants:' + ''.join(missing))

# %%
# =====================================================================
# 6) Construire modeling_units à partir des fenêtres + labels FiRA
# =====================================================================
topics = read_tsv(PATH_QUERIES)
windows = read_tsv(PATH_WINDOWS)
window_labels = read_tsv(PATH_WINDOW_LABELS)

for df in [topics, windows, window_labels]:
    if 'qid' in df.columns:
        df['qid'] = df['qid'].astype(str)
    if 'doc_id' in df.columns:
        df['doc_id'] = df['doc_id'].astype(str)
    if 'window_id' in df.columns:
        df['window_id'] = to_int_series(df['window_id'])

label_col = pick_col(window_labels, ['fira_label', 'passage_relevance', 'silver_label'], required=True, name='label FiRA/fallback')
if label_col == 'silver_label':
    print('[WARN] Ancien cache détecté : silver_label utilisé en fallback. Idéalement relancer le notebook data corrigé FiRA labels.')
else:
    print('[OK] label utilisé :', label_col)

units = windows.merge(
    window_labels,
    on=['qid', 'doc_id', 'window_id'],
    how='left',
    suffixes=('', '_label')
)

text_col = pick_col(units, ['unit_text', 'window_text', 'text'], required=True, name='texte fenêtre')
units['unit_text'] = units[text_col].fillna('').astype(str)
units['unit_id'] = units.apply(lambda r: f"{r['qid']}__{r['doc_id']}__w{int(r['window_id'])}", axis=1)
units['unit_type'] = 'window_50_25'
units['source_unit_id'] = units['window_id'].astype(str)
units['label_relevance'] = to_int_series(units[label_col], default=0)
units['label_name'] = units['label_relevance'].apply(relevance_name)
units['label_strong'] = (units['label_relevance'] >= MIN_STRONG_RELEVANCE).astype(int)

# Métadonnées utiles si disponibles.
for src, dst in [
    ('doc_rank', 'initial_rank'),
    ('doc_score', 'initial_score'),
    ('window_start_token', 'unit_start_token'),
    ('window_end_token', 'unit_end_token'),
]:
    if src in units.columns:
        units[dst] = units[src]

if 'initial_score' not in units.columns:
    units['initial_score'] = 0.0
if 'initial_rank' not in units.columns:
    units['initial_rank'] = 0

query_col = pick_col(topics, ['query_clean', 'query', 'text'], required=True, name='requête')
query_map = dict(zip(topics['qid'].astype(str), topics[query_col].astype(str)))
units['query'] = units['qid'].map(query_map)
units['query_clean'] = units['query'].map(clean_query)

keep_cols = [
    'unit_id', 'unit_type', 'qid', 'query', 'query_clean', 'doc_id', 'source_unit_id',
    'unit_text', 'label_relevance', 'label_name', 'label_strong',
    'initial_rank', 'initial_score', 'unit_start_token', 'unit_end_token'
]
keep_cols = [c for c in keep_cols if c in units.columns]
units = units[keep_cols].drop_duplicates(['qid', 'unit_id']).copy()
units = units[units['unit_text'].str.len() > 0].reset_index(drop=True)
units['unit_len_chars'] = units['unit_text'].str.len().astype(int)
units['unit_len_tokens'] = units['unit_text'].str.split().str.len().astype(int)

save_tsv(units, PATH_MODELING_UNITS)

print('[OK] modeling_units sauvegardé :', PATH_MODELING_UNITS)
print('Shape :', units.shape)
print('Requêtes :', units['qid'].nunique())
print('Distribution labels :')
display(units['label_relevance'].value_counts().sort_index())
display(units.head())

# %%
# =====================================================================
# 7) Indexer les unités candidates avec PyTerrier
# =====================================================================
candidate_units = units[['unit_id', 'unit_text']].drop_duplicates('unit_id').rename(
    columns={'unit_id': 'docno', 'unit_text': 'text'}
).copy()
candidate_units['docno'] = candidate_units['docno'].astype(str)
candidate_units['text'] = candidate_units['text'].fillna('').astype(str)

print('Unités candidates uniques :', len(candidate_units))
display(candidate_units.head())

if FORCE_REBUILD_INDEX and UNIT_INDEX_DIR.exists():
    print('[INDEX] Suppression ancien index :', UNIT_INDEX_DIR)
    shutil.rmtree(UNIT_INDEX_DIR)

if UNIT_INDEX_DIR.exists():
    index_ref = pt.IndexRef.of(str(UNIT_INDEX_DIR / 'data.properties'))
else:
    print('[INDEX] Construction index fields=True, blocks=True...')
    indexer = pt.IterDictIndexer(
        str(UNIT_INDEX_DIR),
        overwrite=True,
        meta={'docno': 512},
        fields=True,
        blocks=True,
    )
    index_ref = indexer.index(candidate_units.to_dict(orient='records'))

index = pt.IndexFactory.of(index_ref)
print('[OK] Index prêt :', index_ref)
print(index.getCollectionStatistics())

# %%
# =====================================================================
# 8) Préparer les requêtes
# =====================================================================
topics_pt = units[['qid', 'query_clean']].drop_duplicates('qid').rename(columns={'query_clean': 'query'}).copy()
topics_pt['qid'] = topics_pt['qid'].astype(str)
topics_pt['query'] = topics_pt['query'].fillna('').astype(str).map(clean_query)
topics_pt = topics_pt[topics_pt['query'].str.len() > 0].reset_index(drop=True)

print('Nombre de requêtes pour retrieval :', len(topics_pt))
display(topics_pt.head())

# %%
# =====================================================================
# 9) Lancer QL et SDM puis filtrer sur les candidats de chaque requête
# =====================================================================

def run_retriever(model_name, pipeline, topics_df):
    print(f'[RUN] {model_name}...')
    run = pipeline.transform(topics_df).copy()
    run['qid'] = run['qid'].astype(str)
    run['unit_id'] = run['docno'].astype(str)
    keep = [c for c in ['qid', 'docno', 'unit_id', 'rank', 'score'] if c in run.columns]
    run = run[keep].copy()
    run['model'] = model_name
    return run

ql_model_used = 'DirichletLM'
ql_pipe = pt.terrier.Retriever(index_ref, wmodel=ql_model_used, num_results=TOPK_RETRIEVAL)
run_ql_raw = run_retriever('QL', ql_pipe, topics_pt)

try:
    sdm_pipe = pt.rewrite.SDM() >> pt.terrier.Retriever(index_ref, wmodel=ql_model_used, num_results=TOPK_RETRIEVAL)
    run_sdm_raw = run_retriever('SDM', sdm_pipe, topics_pt)
    sdm_model_used = f'SDM + {ql_model_used}'
except Exception as e:
    print('[ERROR] SDM a échoué. Vérifie que l’index a bien blocks=True.')
    raise e

candidate_pairs = units[['qid', 'unit_id']].drop_duplicates().copy()
unit_meta = units.drop_duplicates(['qid', 'unit_id']).copy()
unit_meta['initial_score'] = to_float_series(unit_meta['initial_score'], default=0.0)
unit_meta['initial_rank'] = to_int_series(unit_meta['initial_rank'], default=0)


def restrict_and_enrich(run_raw, model_name):
    run = run_raw.merge(candidate_pairs, on=['qid', 'unit_id'], how='inner')
    run = run.merge(unit_meta, on=['qid', 'unit_id'], how='left', validate='many_to_one')
    run = run.sort_values(['qid', 'score'], ascending=[True, False]).copy()
    run['rank'] = run.groupby('qid').cumcount() + 1
    run = run[run['rank'] <= TOPK_EVAL].copy()
    run['model'] = model_name
    return run.reset_index(drop=True)

run_ql = restrict_and_enrich(run_ql_raw, 'QL')
run_sdm = restrict_and_enrich(run_sdm_raw, 'SDM')

print('run_ql :', run_ql.shape, 'requêtes=', run_ql['qid'].nunique())
print('run_sdm:', run_sdm.shape, 'requêtes=', run_sdm['qid'].nunique())
display(run_ql.head())

# %%
# =====================================================================
# 10) Interpolation optionnelle : QL + score initial des candidats
# =====================================================================
run_interp = run_ql.copy()
run_interp['ql_norm'] = run_interp.groupby('qid')['score'].transform(safe_minmax_normalize)
run_interp['initial_norm'] = run_interp.groupby('qid')['initial_score'].transform(safe_minmax_normalize)
run_interp['score'] = INTERP_ALPHA * run_interp['ql_norm'] + (1 - INTERP_ALPHA) * run_interp['initial_norm']
run_interp = run_interp.sort_values(['qid', 'score'], ascending=[True, False]).copy()
run_interp['rank'] = run_interp.groupby('qid').cumcount() + 1
run_interp['model'] = 'QL_INITIAL_INTERP'

print('run_interp:', run_interp.shape)
display(run_interp.head())

# %%
# =====================================================================
# 11) Métriques passage/window-level
# =====================================================================

def average_precision_binary(g, rel_col='label_strong'):
    g = g.sort_values('rank')
    total_rel = int(g[rel_col].sum())
    if total_rel == 0:
        return 0.0
    hits, s = 0, 0.0
    for i, (_, row) in enumerate(g.iterrows(), start=1):
        if int(row[rel_col]) > 0:
            hits += 1
            s += hits / i
    return s / total_rel


def precision_at_k(g, k, rel_col='label_strong'):
    g = g.sort_values('rank').head(k)
    if len(g) == 0:
        return 0.0
    return float(g[rel_col].sum() / k)


def dcg_at_k(rels, k):
    rels = list(rels)[:k]
    return sum((2**int(rel) - 1) / np.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(g, k=10, rel_col='label_relevance'):
    g = g.sort_values('rank')
    dcg = dcg_at_k(g[rel_col].astype(int).tolist(), k)
    ideal = sorted(g[rel_col].astype(int).tolist(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else float(dcg / idcg)


def evaluate_run(run, model_name):
    per_q = []
    for qid, g in run.groupby('qid'):
        g = g.sort_values('rank')
        per_q.append({
            'model': model_name,
            'qid': str(qid),
            'AP': average_precision_binary(g),
            'P@1': precision_at_k(g, 1),
            'P@10': precision_at_k(g, 10),
            'nDCG@10': ndcg_at_k(g, 10),
            'num_retrieved': len(g),
            'num_rel_strong_in_run': int(g['label_strong'].sum()),
            'max_label_in_run': int(g['label_relevance'].max()) if len(g) else 0,
        })
    per_q_df = pd.DataFrame(per_q).sort_values(['model', 'qid']).reset_index(drop=True)
    summary = {
        'model': model_name,
        'num_queries': int(per_q_df['qid'].nunique()) if len(per_q_df) else 0,
        'MAP': float(per_q_df['AP'].mean()) if len(per_q_df) else 0.0,
        'P@1': float(per_q_df['P@1'].mean()) if len(per_q_df) else 0.0,
        'P@10': float(per_q_df['P@10'].mean()) if len(per_q_df) else 0.0,
        'nDCG@10': float(per_q_df['nDCG@10'].mean()) if len(per_q_df) else 0.0,
    }
    return per_q_df, summary

perq_ql, sum_ql = evaluate_run(run_ql, 'QL')
perq_sdm, sum_sdm = evaluate_run(run_sdm, 'SDM')
perq_interp, sum_interp = evaluate_run(run_interp, 'QL_INITIAL_INTERP')

metrics_summary = pd.DataFrame([sum_ql, sum_sdm, sum_interp])
per_query_metrics = pd.concat([perq_ql, perq_sdm, perq_interp], ignore_index=True)

display(metrics_summary)

# %%
# =====================================================================
# 12) Évaluation character-level / Passage2-like simplifiée
# =====================================================================
# Ici on ne cherche plus reconstructed_docs.tsv : chaque unité contribue par len(unit_text).
# Si label_strong=1, tous les caractères de l'unité sont considérés pertinents.


def character_average_precision(run_q, gold_q):
    total_rel_chars = int(gold_q['rel_char_len'].sum())
    if total_rel_chars <= 0:
        return 0.0
    seen = set()
    retrieved_chars = 0
    relevant_chars = 0
    weighted_precision_sum = 0.0
    for _, row in run_q.sort_values('rank').iterrows():
        uid = str(row['unit_id'])
        if uid in seen:
            continue
        seen.add(uid)
        unit_chars = int(row.get('unit_len_chars', 0))
        rel_chars = int(row.get('unit_len_chars', 0)) if int(row.get('label_strong', 0)) > 0 else 0
        retrieved_chars += max(unit_chars, 0)
        if rel_chars > 0 and retrieved_chars > 0:
            relevant_chars += rel_chars
            weighted_precision_sum += rel_chars * (relevant_chars / retrieved_chars)
    return float(weighted_precision_sum / total_rel_chars)


def char_precision_at_k(run_q, k):
    g = run_q.sort_values('rank').head(k)
    denom = int(g['unit_len_chars'].sum())
    if denom <= 0:
        return 0.0
    num = int(g.loc[g['label_strong'].astype(int) > 0, 'unit_len_chars'].sum())
    return float(num / denom)


def evaluate_run_character_level(run, model_name):
    gold = unit_meta[['qid', 'unit_id', 'unit_len_chars', 'label_strong']].drop_duplicates(['qid', 'unit_id']).copy()
    gold['rel_char_len'] = np.where(gold['label_strong'].astype(int) > 0, gold['unit_len_chars'].astype(int), 0)
    per_q = []
    for qid, g in run.groupby('qid'):
        gold_q = gold[gold['qid'].astype(str) == str(qid)]
        per_q.append({
            'model': model_name,
            'qid': str(qid),
            'CharAP': character_average_precision(g, gold_q),
            'CharP@1': char_precision_at_k(g, 1),
            'CharP@10': char_precision_at_k(g, 10),
            'gold_rel_chars': int(gold_q['rel_char_len'].sum()),
            'retrieved_chars_top50': int(g.sort_values('rank').head(TOPK_EVAL)['unit_len_chars'].sum()),
        })
    per_q_df = pd.DataFrame(per_q).sort_values(['model', 'qid']).reset_index(drop=True)
    summary = {
        'model': model_name,
        'num_queries': int(per_q_df['qid'].nunique()) if len(per_q_df) else 0,
        'CharMAP': float(per_q_df['CharAP'].mean()) if len(per_q_df) else 0.0,
        'CharP@1': float(per_q_df['CharP@1'].mean()) if len(per_q_df) else 0.0,
        'CharP@10': float(per_q_df['CharP@10'].mean()) if len(per_q_df) else 0.0,
    }
    return per_q_df, summary

perq_ql_char, sum_ql_char = evaluate_run_character_level(run_ql, 'QL')
perq_sdm_char, sum_sdm_char = evaluate_run_character_level(run_sdm, 'SDM')
perq_interp_char, sum_interp_char = evaluate_run_character_level(run_interp, 'QL_INITIAL_INTERP')

char_metrics_summary = pd.DataFrame([sum_ql_char, sum_sdm_char, sum_interp_char])
char_per_query_metrics = pd.concat([perq_ql_char, perq_sdm_char, perq_interp_char], ignore_index=True)

combined_metrics_summary = metrics_summary.merge(char_metrics_summary, on=['model', 'num_queries'], how='left')
print('Résumé combiné :')
display(combined_metrics_summary)

# %%
# =====================================================================
# 13) Diagnostics rapides
# =====================================================================
print('Distribution labels dans toutes les unités :')
display(units['label_relevance'].value_counts().sort_index())

print('Distribution labels dans QL top-50 :')
display(run_ql['label_relevance'].value_counts().sort_index())

print('Distribution labels dans SDM top-50 :')
display(run_sdm['label_relevance'].value_counts().sort_index())

print('Top requêtes selon AP QL :')
display(perq_ql.sort_values('AP', ascending=False).head(10))

print('Requêtes faibles selon AP QL :')
display(perq_ql.sort_values('AP', ascending=True).head(10))

# %%
# =====================================================================
# 14) Exports dans Drive
# =====================================================================
save_tsv(run_ql, PATH_RUN_QL)
save_tsv(run_sdm, PATH_RUN_SDM)
save_tsv(run_interp, PATH_RUN_INTERP)
save_tsv(metrics_summary, PATH_METRICS_SUMMARY)
save_tsv(per_query_metrics, PATH_PER_QUERY_METRICS)
save_tsv(char_metrics_summary, PATH_CHAR_METRICS_SUMMARY)
save_tsv(char_per_query_metrics, PATH_CHAR_PER_QUERY_METRICS)

metadata_model = {
    'project': 'FiRA TREC-19 article-like modeling',
    'data_variant': DATA_VARIANT,
    'uses_windows': USE_WINDOWS,
    'data_dir': str(DATA_DIR),
    'output_dir': str(MODEL_OUT_DIR),
    'modeling_units_path': str(PATH_MODELING_UNITS),
    'unit_type': 'fenêtre 50/25',
    'labels': 'FiRA direct labels: 0 non_relevant, 1 fair/topic, 2 good, 3 perfect',
    'min_strong_relevance': MIN_STRONG_RELEVANCE,
    'topk_retrieval': TOPK_RETRIEVAL,
    'topk_eval': TOPK_EVAL,
    'models': ['QL', 'SDM', 'QL_INITIAL_INTERP'],
    'ql_model_used': ql_model_used,
    'sdm_model_used': sdm_model_used,
    'metrics_summary': metrics_summary.to_dict(orient='records'),
    'char_metrics_summary': char_metrics_summary.to_dict(orient='records'),
    'important_note': 'Character-level evaluation uses len(unit_text), so it does not require reconstructed_docs.tsv.',
}
PATH_METADATA_MODEL.write_text(json.dumps(metadata_model, indent=2, ensure_ascii=False), encoding='utf-8')

print('[OK] Exports terminés dans :', MODEL_OUT_DIR)
for p in sorted(MODEL_OUT_DIR.iterdir()):
    print(' -', p)