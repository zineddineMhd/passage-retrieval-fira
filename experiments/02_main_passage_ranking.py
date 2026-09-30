# FiRA/TREC-DL passage retrieval experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # 02 — Modeling FiRA TREC-19 top-50-only avec fenêtres + évaluation unifiée + tables article
#
# Ce notebook couvre les étapes 2 et 3, puis ajoute la suite expérimentale de l'article :
#
# - Table 1-like : QL, SDM, QL-Interpolated ;
# - Table 2-like : PM-TFIDF, PM-Dirichlet, PM-SkewedGaussian ;
# - Table 3-like : Query Expansion / Relevance Model sur documents et passages ;
# - Table 4-like : effet de la qualité des documents d'entrée Top 5 / Top 10 / Top 25 / Top 50 / Only Relevant ;
# - Figure 1-like : distribution des scores des passages récupérés vs passages-réponses.
#
# Entrée attendue : le dataset corrigé de l'étape 1, c'est-à-dire `windows_top50_labeled_main.tsv`.

# %%
# =====================================================================
# 1) Installation des dépendances
# =====================================================================
# NOTEBOOK: !pip -q install python-terrier pandas numpy tqdm scipy

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
import math
import shutil
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scipy.special import erf as scipy_erf
except Exception:
    scipy_erf = None

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

# ---------------------------------------------------------------------
# Paramètres article-like
# ---------------------------------------------------------------------
MIN_STRONG_RELEVANCE = 2          # Perfect/Excellent-like : labels FiRA >= 2
TOPK_EVAL = 50                   # nombre de fenêtres évaluées par requête
TOPK_RETRIEVAL_GLOBAL = 200000   # fallback si le scoring candidat PyTerrier ne marche pas
FORCE_REBUILD_INDEX = True

# QL / interpolation
QL_WMODEL = 'DirichletLM'
INTERP_ALPHA = 0.75              # score_final = alpha*QL + (1-alpha)*score_document_initial

# PM article-like
PM_MU = 2500.0
PM_SIGMA = 2000.0
PM_SKEW_ALPHA = 1.0

DRIVE_ROOT = Path('/content/drive/MyDrive/IR_project_backup')

# Entrée corrigée de l'étape 1.
DATA_DIR = DRIVE_ROOT / 'fira_trec19_article_like_with_windows_top50_only'

# Sortie séparée pour éviter de mélanger avec l'ancien notebook.
MODEL_OUT_DIR = DRIVE_ROOT / 'fira_trec19_modeling_article_like_top50_only_full_article_eval'
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIT_INDEX_DIR = MODEL_OUT_DIR / 'pt_window_index'

# Artefacts d'entrée produits par le notebook 01 corrigé.
PATH_MAIN_DATASET = DATA_DIR / 'windows_top50_labeled_main.tsv'
PATH_QUERIES = DATA_DIR / 'queries_final.tsv'
PATH_TOPDOCS = DATA_DIR / 'topdocs_top50.tsv'
PATH_CANDIDATE_TOPDOCS = DATA_DIR / 'candidate_topdocs_top50.tsv'
PATH_RECON_DOCS = DATA_DIR / 'fira_reconstructed_documents.tsv'
PATH_GOLD_SPANS_TOKEN = DATA_DIR / 'fira_aligned_snippet_spans_in_top50.tsv'
PATH_METADATA_DATA = DATA_DIR / 'dataset_metadata.json'

# Exports.
PATH_MODELING_UNITS = MODEL_OUT_DIR / 'modeling_units_windows.tsv'
PATH_GOLD_CHAR_SPANS = MODEL_OUT_DIR / 'gold_char_spans.tsv'
PATH_RUN_STANDARD_ALL = MODEL_OUT_DIR / 'runs_standard_all_models.tsv'
PATH_METRICS_ARTICLE = MODEL_OUT_DIR / 'metrics_article_like_char.tsv'
PATH_PER_QUERY_ARTICLE = MODEL_OUT_DIR / 'per_query_article_like_char.tsv'
PATH_METRICS_WINDOW = MODEL_OUT_DIR / 'metrics_secondary_window.tsv'
PATH_PER_QUERY_WINDOW = MODEL_OUT_DIR / 'per_query_secondary_window.tsv'
PATH_METRICS_COMBINED = MODEL_OUT_DIR / 'metrics_combined.tsv'
PATH_ARTICLE_TABLE = MODEL_OUT_DIR / 'table_main_article_like_MAP_P1_P10.tsv'
PATH_METADATA_MODEL = MODEL_OUT_DIR / 'modeling_metadata.json'

print('[DATA] ', DATA_DIR)
print('[OUT ] ', MODEL_OUT_DIR)
print('[MAIN]', PATH_MAIN_DATASET)

# %%
# =====================================================================
# 4) Helpers
# =====================================================================

def read_tsv(path, **kwargs):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Fichier manquant : {path}')
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, **kwargs)


def save_tsv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep='\t', index=False)
    print(f'[SAVE] {path} ({len(df)} lignes)')


def to_int_series(s, default=0):
    return pd.to_numeric(s, errors='coerce').fillna(default).astype(int)


def to_float_series(s, default=0.0):
    return pd.to_numeric(s, errors='coerce').fillna(default).astype(float)


def clean_query(q):
    q = str(q).lower()
    q = re.sub(r'[_]+', ' ', q)
    q = re.sub(r'[^a-z0-9\s]', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()
    return q


def tok(text):
    return re.findall(r'\w+', str(text).lower())


def safe_minmax_normalize(s):
    s = pd.to_numeric(s, errors='coerce').fillna(0.0).astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def relevance_name(rel):
    rel = int(rel)
    return {
        0: 'non_relevant_or_no_overlap',
        1: 'fair_or_topic_relevant',
        2: 'good_answer',
        3: 'perfect_answer',
    }.get(rel, f'label_{rel}')


def tokenize_with_char_offsets(text):
    """
    Tokenisation compatible avec tok() : \\w+ en lowercase.
    Retourne les tokens et leurs offsets caractère dans le document reconstruit.
    """
    tokens = []
    starts = []
    ends = []
    for m in re.finditer(r'\w+', str(text).lower()):
        tokens.append(m.group(0))
        starts.append(int(m.start()))
        ends.append(int(m.end()))
    return tokens, starts, ends


def token_span_to_char_span(start_token, end_token, char_starts, char_ends):
    """
    Convertit un intervalle token [start_token, end_token) en intervalle caractère.
    Si l'intervalle dépasse le document, il est tronqué.
    """
    n = len(char_starts)
    start_token = max(0, int(start_token))
    end_token = min(n, int(end_token))
    if n == 0 or end_token <= start_token:
        return None
    return int(char_starts[start_token]), int(char_ends[end_token - 1])


def stable_doc_key(doc_id, char_pos):
    return (str(doc_id), int(char_pos))

# %%
# =====================================================================
# 5) Vérification des fichiers d'entrée
# =====================================================================
required = {
    'main_dataset': PATH_MAIN_DATASET,
    'queries': PATH_QUERIES,
    'reconstructed_docs': PATH_RECON_DOCS,
    'gold_token_spans': PATH_GOLD_SPANS_TOKEN,
}
optional = {
    'topdocs': PATH_TOPDOCS,
    'candidate_topdocs': PATH_CANDIDATE_TOPDOCS,
    'metadata': PATH_METADATA_DATA,
}

rows = []
for name, p in {**required, **optional}.items():
    p = Path(p)
    rows.append({
        'artifact': name,
        'exists': p.exists(),
        'size_MB': round(p.stat().st_size / (1024 * 1024), 3) if p.exists() else 0,
        'path': str(p),
    })
check_df = pd.DataFrame(rows)
display(check_df)

missing = [str(p) for p in required.values() if not Path(p).exists()]
if missing:
    raise FileNotFoundError('Fichiers obligatoires manquants :\n' + '\n'.join(missing))

# %%
# =====================================================================
# 6) Charger le dataset principal et construire modeling_units
# =====================================================================
main = read_tsv(PATH_MAIN_DATASET)
topics = read_tsv(PATH_QUERIES)

# Normalisation des types.
for df in [main, topics]:
    if 'qid' in df.columns:
        df['qid'] = df['qid'].astype(str)
    if 'doc_id' in df.columns:
        df['doc_id'] = df['doc_id'].astype(str)

for c in ['doc_rank', 'window_id', 'window_start', 'window_end', 'fira_label', 'is_relevant']:
    if c in main.columns:
        main[c] = to_int_series(main[c])

if 'doc_score' in main.columns:
    main['doc_score'] = to_float_series(main['doc_score'])
else:
    main['doc_score'] = 0.0

expected = {'qid', 'doc_id', 'doc_rank', 'window_id', 'window_start', 'window_end', 'window_text', 'fira_label', 'is_relevant'}
missing_cols = expected - set(main.columns)
if missing_cols:
    raise KeyError(f'Colonnes manquantes dans windows_top50_labeled_main.tsv : {missing_cols}')

# Requête.
query_col = None
for c in ['query_clean', 'query', 'text']:
    if c in topics.columns:
        query_col = c
        break
if query_col is None:
    raise KeyError(f'Impossible de trouver la colonne requête dans {PATH_QUERIES}. Colonnes = {list(topics.columns)}')

query_map = dict(zip(topics['qid'].astype(str), topics[query_col].astype(str)))

units = main.copy()
units['unit_id'] = units.apply(lambda r: f"{r['qid']}__{r['doc_id']}__w{int(r['window_id'])}", axis=1)
units['unit_type'] = 'window_50_25'
units['unit_text'] = units['window_text'].fillna('').astype(str)
units['query'] = units['qid'].map(query_map).fillna('')
units['query_clean'] = units['query'].map(clean_query)

units['label_relevance'] = to_int_series(units['fira_label'])
units['label_strong'] = (units['label_relevance'] >= MIN_STRONG_RELEVANCE).astype(int)
# Sécurité : si le fichier fournit is_relevant, on vérifie qu'il est cohérent.
units['is_relevant_from_file'] = to_int_series(units['is_relevant'])
mismatch = int((units['is_relevant_from_file'] != units['label_strong']).sum())
if mismatch:
    print(f'[WARN] {mismatch} lignes ont is_relevant différent de int(fira_label >= {MIN_STRONG_RELEVANCE}). On utilise label_strong recalculé.')

units['initial_rank'] = to_int_series(units['doc_rank'])
units['initial_score'] = to_float_series(units['doc_score'])
units['unit_start_token'] = to_int_series(units['window_start'])
units['unit_end_token'] = to_int_series(units['window_end'])
units['unit_len_tokens'] = (units['unit_end_token'] - units['unit_start_token']).clip(lower=0).astype(int)
units['unit_len_chars_approx'] = units['unit_text'].str.len().astype(int)

# On enlève seulement les fenêtres vides.
units = units[units['unit_text'].str.len() > 0].drop_duplicates(['qid', 'unit_id']).reset_index(drop=True)

keep_cols = [
    'qid', 'query', 'query_clean', 'doc_id', 'unit_id', 'unit_type',
    'window_id', 'unit_start_token', 'unit_end_token', 'unit_text',
    'label_relevance', 'label_strong', 'initial_rank', 'initial_score',
    'unit_len_tokens', 'unit_len_chars_approx'
]
units = units[keep_cols].copy()

save_tsv(units, PATH_MODELING_UNITS)

print('[OK] modeling_units')
print('shape:', units.shape)
print('requêtes:', units.qid.nunique())
print('fenêtres positives label>=2:', int(units.label_strong.sum()))
display(units.head())
display(units['label_relevance'].value_counts().sort_index())

# %%
# =====================================================================
# 7) Construire les offsets caractère pour fenêtres et gold spans
# =====================================================================
# Cette cellule produit le format commun :
# run standard : qid | doc_id | unit_id | rank | score | start_char | end_char
# gold standard : qid | doc_id | gold_start_char | gold_end_char | label

docs = read_tsv(PATH_RECON_DOCS)
docs['doc_id'] = docs['doc_id'].astype(str)

doc_text_col = None
for c in ['doc_text', 'text', 'body']:
    if c in docs.columns:
        doc_text_col = c
        break
if doc_text_col is None:
    raise KeyError(f'Impossible de trouver le texte document dans {PATH_RECON_DOCS}. Colonnes={list(docs.columns)}')

doc_text = dict(zip(docs['doc_id'].astype(str), docs[doc_text_col].astype(str)))

# Token offsets par document.
doc_offsets = {}
for did, text in tqdm(doc_text.items(), desc='Token offsets docs'):
    tokens_doc, starts, ends = tokenize_with_char_offsets(text)
    doc_offsets[str(did)] = {
        'tokens': tokens_doc,
        'starts': starts,
        'ends': ends,
        'num_tokens': len(tokens_doc),
    }

# Offsets caractère pour chaque fenêtre.
start_chars = []
end_chars = []
bad_windows = 0
for r in tqdm(units.itertuples(index=False), total=len(units), desc='Window char offsets'):
    off = doc_offsets.get(str(r.doc_id))
    if not off:
        start_chars.append(-1)
        end_chars.append(-1)
        bad_windows += 1
        continue
    span = token_span_to_char_span(r.unit_start_token, r.unit_end_token, off['starts'], off['ends'])
    if span is None:
        start_chars.append(-1)
        end_chars.append(-1)
        bad_windows += 1
    else:
        start_chars.append(span[0])
        end_chars.append(span[1])

units['start_char'] = start_chars
units['end_char'] = end_chars
units['unit_len_chars'] = (units['end_char'] - units['start_char']).clip(lower=0).astype(int)

if bad_windows:
    print(f'[WARN] {bad_windows} fenêtres sans offset caractère valide. Elles seront ignorées dans l’évaluation char-level.')

# Gold spans caractère depuis les annotations FiRA alignées.
gold_token = read_tsv(PATH_GOLD_SPANS_TOKEN)
for c in ['qid', 'doc_id']:
    gold_token[c] = gold_token[c].astype(str)
for c in ['start_token', 'end_token', 'passage_relevance']:
    if c in gold_token.columns:
        gold_token[c] = to_int_series(gold_token[c])

gold_rows = []
bad_spans = 0
for r in tqdm(gold_token.itertuples(index=False), total=len(gold_token), desc='Gold char spans'):
    off = doc_offsets.get(str(r.doc_id))
    if not off:
        bad_spans += 1
        continue
    span = token_span_to_char_span(int(r.start_token), int(r.end_token), off['starts'], off['ends'])
    if span is None:
        bad_spans += 1
        continue
    gold_rows.append({
        'qid': str(r.qid),
        'doc_id': str(r.doc_id),
        'gold_start_char': int(span[0]),
        'gold_end_char': int(span[1]),
        'label': int(r.passage_relevance),
        'passage_id': str(getattr(r, 'passage_id', '')),
        'gold_start_token': int(r.start_token),
        'gold_end_token': int(r.end_token),
    })

gold_char = pd.DataFrame(gold_rows)
gold_char['is_relevant'] = (gold_char['label'].astype(int) >= MIN_STRONG_RELEVANCE).astype(int)

save_tsv(gold_char, PATH_GOLD_CHAR_SPANS)

print('[OK] gold_char_spans:', gold_char.shape)
print('bad_spans:', bad_spans)
display(gold_char.head())
display(gold_char['label'].value_counts().sort_index())

# %%
# =====================================================================
# 8) Indexer les fenêtres avec PyTerrier
# =====================================================================
candidate_units = units[['unit_id', 'unit_text']].drop_duplicates('unit_id').rename(
    columns={'unit_id': 'docno', 'unit_text': 'text'}
).copy()
candidate_units['docno'] = candidate_units['docno'].astype(str)
candidate_units['text'] = candidate_units['text'].fillna('').astype(str)

if FORCE_REBUILD_INDEX and UNIT_INDEX_DIR.exists():
    print('[INDEX] Suppression ancien index :', UNIT_INDEX_DIR)
    shutil.rmtree(UNIT_INDEX_DIR)

if UNIT_INDEX_DIR.exists():
    index_ref = pt.IndexRef.of(str(UNIT_INDEX_DIR / 'data.properties'))
else:
    print('[INDEX] Construction index fenêtres fields=True, blocks=True...')
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
# 9) QL / SDM sur les fenêtres candidates de chaque requête
# =====================================================================
topics_pt = units[['qid', 'query_clean']].drop_duplicates('qid').rename(columns={'query_clean': 'query'}).copy()
topics_pt['qid'] = topics_pt['qid'].astype(str)
topics_pt['query'] = topics_pt['query'].fillna('').astype(str).map(clean_query)
topics_pt = topics_pt[topics_pt['query'].str.len() > 0].reset_index(drop=True)

candidate_pairs = units[['qid', 'query_clean', 'unit_id']].drop_duplicates().rename(
    columns={'query_clean': 'query', 'unit_id': 'docno'}
).copy()
candidate_pairs['qid'] = candidate_pairs['qid'].astype(str)
candidate_pairs['query'] = candidate_pairs['query'].astype(str)
candidate_pairs['docno'] = candidate_pairs['docno'].astype(str)

unit_meta = units.copy()
unit_meta['qid'] = unit_meta['qid'].astype(str)
unit_meta['unit_id'] = unit_meta['unit_id'].astype(str)


def _enrich_rank(run, model_name, topk=TOPK_EVAL):
    run = run.copy()
    run['qid'] = run['qid'].astype(str)
    if 'docno' in run.columns and 'unit_id' not in run.columns:
        run = run.rename(columns={'docno': 'unit_id'})
    run['unit_id'] = run['unit_id'].astype(str)
    run['score'] = to_float_series(run['score'])

    # Filtre fort : on ne garde que les fenêtres candidates du même qid.
    valid_pairs = units[['qid', 'unit_id']].drop_duplicates()
    run = run.merge(valid_pairs, on=['qid', 'unit_id'], how='inner')
    run = run.merge(unit_meta, on=['qid', 'unit_id'], how='left', validate='many_to_one')

    run = run.sort_values(['qid', 'score', 'initial_rank', 'window_id'], ascending=[True, False, True, True]).copy()
    run['rank'] = run.groupby('qid').cumcount() + 1
    run = run[run['rank'] <= topk].copy()
    run['model'] = model_name

    cols = [
        'model', 'qid', 'doc_id', 'unit_id', 'rank', 'score',
        'start_char', 'end_char', 'unit_start_token', 'unit_end_token',
        'label_relevance', 'label_strong', 'initial_rank', 'initial_score',
        'unit_text'
    ]
    cols = [c for c in cols if c in run.columns]
    return run[cols].reset_index(drop=True)


def score_with_pyterrier_candidates(model_name, pipe):
    """
    Essaye d'abord de scorer explicitement les candidats (qid, query, docno).
    Si la version PyTerrier/Terrier installée ne supporte pas ce mode, fallback :
    retrieval global avec num_results très grand, puis filtrage sur les fenêtres candidates qid.
    """
    print(f'[RUN] {model_name} candidate scoring...')
    try:
        out = pipe.transform(candidate_pairs[['qid', 'query', 'docno']].copy())
        if not {'qid', 'docno', 'score'}.issubset(out.columns):
            raise RuntimeError(f'Sortie PyTerrier inattendue: {out.columns}')

        scored_pairs = out[['qid', 'docno']].drop_duplicates()
        coverage = len(scored_pairs.merge(candidate_pairs[['qid', 'docno']], on=['qid', 'docno'], how='inner')) / max(1, len(candidate_pairs))
        print(f'[RUN] {model_name} candidate coverage = {coverage:.3f}')

        # Si PyTerrier a vraiment scoré les candidats, la couverture doit être élevée.
        if coverage >= 0.90:
            return _enrich_rank(out.rename(columns={'docno': 'unit_id'}), model_name)

        print(f'[WARN] Couverture candidate trop faible pour {model_name}. Fallback global high-k.')
    except Exception as e:
        print(f'[WARN] Candidate scoring PyTerrier indisponible pour {model_name}: {repr(e)}')
        print('[WARN] Fallback global high-k + filtrage par qid/unit_id.')

    out = pipe.transform(topics_pt.copy())
    if 'docno' not in out.columns:
        raise RuntimeError(f'Sortie retrieval sans docno pour {model_name}. Colonnes={out.columns}')
    return _enrich_rank(out.rename(columns={'docno': 'unit_id'}), model_name)


ql_pipe = pt.terrier.Retriever(index_ref, wmodel=QL_WMODEL, num_results=TOPK_RETRIEVAL_GLOBAL)
run_ql = score_with_pyterrier_candidates('QL', ql_pipe)

try:
    sdm_pipe = pt.rewrite.SDM() >> pt.terrier.Retriever(index_ref, wmodel=QL_WMODEL, num_results=TOPK_RETRIEVAL_GLOBAL)
    run_sdm = score_with_pyterrier_candidates('SDM', sdm_pipe)
    sdm_model_used = f'SDM + {QL_WMODEL}'
except Exception as e:
    print('[ERROR] SDM a échoué. Vérifie que l’index est bien créé avec blocks=True.')
    raise e

print('run_ql :', run_ql.shape, 'requêtes=', run_ql.qid.nunique())
print('run_sdm:', run_sdm.shape, 'requêtes=', run_sdm.qid.nunique())
display(run_ql.head())

# %%
# =====================================================================
# 10) QL-Interpolated = QL fenêtre + score document initial
# =====================================================================
run_interp = run_ql.copy()
run_interp['ql_norm'] = run_interp.groupby('qid')['score'].transform(safe_minmax_normalize)
run_interp['initial_norm'] = run_interp.groupby('qid')['initial_score'].transform(safe_minmax_normalize)
run_interp['score'] = INTERP_ALPHA * run_interp['ql_norm'] + (1.0 - INTERP_ALPHA) * run_interp['initial_norm']
run_interp = run_interp.sort_values(['qid', 'score', 'initial_rank'], ascending=[True, False, True]).copy()
run_interp['rank'] = run_interp.groupby('qid').cumcount() + 1
run_interp['model'] = 'QL-Interpolated'
run_interp = run_interp[run_interp['rank'] <= TOPK_EVAL].reset_index(drop=True)

display(run_interp.head())

# %%
# =====================================================================
# 11) PM-TFIDF, PM-Dirichlet, PM-SkewedGaussian
# =====================================================================
# Implémentation article-like :
# - PM-TFIDF : score = somme pseudo_tf(t, fenêtre) * idf(t)
# - PM-Dirichlet : QL Dirichlet avec pseudo-tf positionnel symétrique
# - PM-SkewedGaussian : QL Dirichlet avec noyau Gaussian asymétrique
#
# Correction importante par rapport à une version simplifiée :
# PM-TFIDF utilise maintenant la pseudo-fréquence positionnelle, pas seulement
# le TF direct dans la fenêtre.

def erf_vec(x):
    x = np.asarray(x, dtype=float)
    if scipy_erf is not None:
        return scipy_erf(x)
    return np.vectorize(math.erf)(x)


def gaussian_kernel(query_pos, positions_i, sigma=PM_SIGMA):
    x = np.asarray(positions_i, dtype=float) - float(query_pos)
    return np.exp(-(x ** 2) / (2.0 * sigma ** 2))


def skewed_gaussian_kernel(query_pos, positions_i, sigma=PM_SIGMA, alpha=PM_SKEW_ALPHA):
    x = np.asarray(positions_i, dtype=float) - float(query_pos)
    return np.exp(-(x ** 2) / (2.0 * sigma ** 2)) * (1.0 + erf_vec(alpha * x / math.sqrt(2.0)))


# Statistiques IDF / collection sur les fenêtres.
all_window_tokens = units['unit_text'].map(tok).tolist()
N_units = len(all_window_tokens)
df_counts = Counter()
cf_counts = Counter()
total_terms = 0
for toks in all_window_tokens:
    total_terms += len(toks)
    cf_counts.update(toks)
    df_counts.update(set(toks))

def idf(term):
    return math.log((N_units + 1.0) / (df_counts.get(term, 0) + 0.5))

def p_collection(term):
    return (cf_counts.get(term, 0) + 1.0) / (total_terms + max(1, len(cf_counts)))


def build_positions_for_terms(doc_tokens, query_terms):
    query_set = set(query_terms)
    pos = {t: [] for t in query_set}
    for i, t in enumerate(doc_tokens):
        if t in query_set:
            pos[t].append(i)
    return pos


def pseudo_tf_for_positions(positions, start, end, kernel='gaussian'):
    if not positions:
        return 0.0
    positions_i = np.arange(int(start), int(end), dtype=float)
    if len(positions_i) == 0:
        return 0.0

    val = 0.0
    for pos in positions:
        if kernel == 'skewed':
            val += float(skewed_gaussian_kernel(pos, positions_i).sum())
        elif kernel == 'gaussian':
            val += float(gaussian_kernel(pos, positions_i).sum())
        else:
            raise ValueError(kernel)
    return val


def normalize_query_weights(query_terms):
    counts = Counter(query_terms)
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {t: float(c) / float(total) for t, c in counts.items()}


def pm_score_weighted(query_weights, positions_by_term, start, end, scoring_kind='skewed'):
    """
    Score PM avec poids de requête optionnels.
    Pour les modèles sans expansion, query_weights peut simplement contenir les termes originaux.
    """
    length = max(1, int(end) - int(start))
    score = 0.0

    for t, w in query_weights.items():
        if w <= 0:
            continue

        if scoring_kind == 'tfidf':
            # PM-TFIDF de l'article : tf pseudo-positionnel * idf.
            tf = pseudo_tf_for_positions(positions_by_term.get(t, []), start, end, kernel='gaussian')
            score += float(w) * tf * idf(t)

        elif scoring_kind == 'dirichlet':
            tf = pseudo_tf_for_positions(positions_by_term.get(t, []), start, end, kernel='gaussian')
            pc = p_collection(t)
            score += float(w) * math.log((tf + PM_MU * pc) / (length + PM_MU))

        elif scoring_kind == 'skewed':
            tf = pseudo_tf_for_positions(positions_by_term.get(t, []), start, end, kernel='skewed')
            pc = p_collection(t)
            score += float(w) * math.log((tf + PM_MU * pc) / (length + PM_MU))

        else:
            raise ValueError(scoring_kind)

    return float(score)


def score_pm_raw(units_subset, scoring_kind, query_weights_by_qid=None, desc=None):
    """
    Score toutes les fenêtres de units_subset et retourne un DataFrame brut :
    qid | unit_id | score

    query_weights_by_qid :
    - None : requête originale, poids uniformes par terme.
    - dict[qid] -> dict[term] -> weight : utile pour RM.
    """
    rows = []
    units_subset = units_subset.copy()
    units_subset['qid'] = units_subset['qid'].astype(str)
    units_subset['doc_id'] = units_subset['doc_id'].astype(str)
    units_subset['unit_id'] = units_subset['unit_id'].astype(str)

    iterator = units_subset.groupby('qid')
    if desc is None:
        desc = f'Scoring PM-{scoring_kind}'

    for qid, g in tqdm(iterator, desc=desc):
        qid = str(qid)

        if query_weights_by_qid is None:
            query_terms = tok(g['query_clean'].iloc[0])
            query_weights = normalize_query_weights(query_terms)
        else:
            query_weights = {str(k): float(v) for k, v in query_weights_by_qid.get(qid, {}).items() if float(v) > 0}

        if not query_weights:
            continue

        query_terms = list(query_weights.keys())

        for doc_id, gd in g.groupby('doc_id'):
            off = doc_offsets.get(str(doc_id))
            if not off:
                continue

            doc_tokens = off['tokens']
            positions_by_term = build_positions_for_terms(doc_tokens, query_terms)

            for r in gd.itertuples(index=False):
                start = int(r.unit_start_token)
                end = int(r.unit_end_token)
                score = pm_score_weighted(
                    query_weights=query_weights,
                    positions_by_term=positions_by_term,
                    start=start,
                    end=end,
                    scoring_kind=scoring_kind,
                )

                rows.append({
                    'qid': qid,
                    'unit_id': str(r.unit_id),
                    'score': float(score),
                })

    return pd.DataFrame(rows, columns=['qid', 'unit_id', 'score'])


def build_pm_run_for_units(units_subset, model_name, scoring_kind='skewed', query_weights_by_qid=None):
    raw = score_pm_raw(
        units_subset=units_subset,
        scoring_kind=scoring_kind,
        query_weights_by_qid=query_weights_by_qid,
        desc=f'Scoring {model_name}',
    )
    return _enrich_rank(raw, model_name)


# Runs principaux Table 2-like.
raw_pm_tfidf = score_pm_raw(units, scoring_kind='tfidf', desc='Scoring PM-TFIDF')
raw_pm_dirichlet = score_pm_raw(units, scoring_kind='dirichlet', desc='Scoring PM-Dirichlet')
raw_pm_skewed = score_pm_raw(units, scoring_kind='skewed', desc='Scoring PM-SkewedGaussian')

run_pm_tfidf = _enrich_rank(raw_pm_tfidf, 'PM-TFIDF')
run_pm_dirichlet = _enrich_rank(raw_pm_dirichlet, 'PM-Dirichlet')
run_pm_skewed = _enrich_rank(raw_pm_skewed, 'PM-SkewedGaussian')

print(run_pm_tfidf.shape, run_pm_dirichlet.shape, run_pm_skewed.shape)
display(run_pm_skewed.head())

# %%
# =====================================================================
# 12) Format run standard commun pour ton notebook et celui du binôme
# =====================================================================
model_runs = {
    'QL': run_ql,
    'SDM': run_sdm,
    'QL-Interpolated': run_interp,
    'PM-TFIDF': run_pm_tfidf,
    'PM-Dirichlet': run_pm_dirichlet,
    'PM-SkewedGaussian': run_pm_skewed,
}

def to_standard_run(run):
    cols = ['model', 'qid', 'doc_id', 'unit_id', 'rank', 'score', 'start_char', 'end_char']
    out = run[cols].copy()
    out['qid'] = out['qid'].astype(str)
    out['doc_id'] = out['doc_id'].astype(str)
    out['unit_id'] = out['unit_id'].astype(str)
    out['rank'] = to_int_series(out['rank'])
    out['score'] = to_float_series(out['score'])
    out['start_char'] = to_int_series(out['start_char'], default=-1)
    out['end_char'] = to_int_series(out['end_char'], default=-1)
    out = out[(out['start_char'] >= 0) & (out['end_char'] > out['start_char'])].copy()
    return out.sort_values(['model', 'qid', 'rank']).reset_index(drop=True)

runs_standard_all = pd.concat([to_standard_run(r) for r in model_runs.values()], ignore_index=True)
save_tsv(runs_standard_all, PATH_RUN_STANDARD_ALL)

print('[OK] run standard exporté :')
display(runs_standard_all.head())

print('Colonnes gold standard :')
display(gold_char[['qid', 'doc_id', 'gold_start_char', 'gold_end_char', 'label']].head())

# %%
# =====================================================================
# 13) Évaluation principale article-like : Passage2-like au niveau caractère
# =====================================================================

def build_gold_char_sets(gold_df, qid):
    g = gold_df[(gold_df['qid'].astype(str) == str(qid)) & (gold_df['label'].astype(int) >= MIN_STRONG_RELEVANCE)]
    rel_chars = set()
    for r in g.itertuples(index=False):
        # Union des caractères pertinents : évite de compter deux fois des spans gold qui se chevauchent.
        for pos in range(int(r.gold_start_char), int(r.gold_end_char)):
            rel_chars.add(stable_doc_key(r.doc_id, pos))
    return rel_chars


_gold_cache = {}

def get_gold_chars(qid):
    qid = str(qid)
    if qid not in _gold_cache:
        _gold_cache[qid] = build_gold_char_sets(gold_char, qid)
    return _gold_cache[qid]


def char_average_precision_passage2(run_q, qid):
    gold_chars = get_gold_chars(qid)
    total_rel = len(gold_chars)
    if total_rel == 0:
        return 0.0

    seen_retrieved = set()
    seen_relevant = set()
    retrieved_count = 0
    weighted_precision_sum = 0.0

    for r in run_q.sort_values('rank').itertuples(index=False):
        span_chars = {
            stable_doc_key(r.doc_id, pos)
            for pos in range(int(r.start_char), int(r.end_char))
        }
        new_chars = span_chars - seen_retrieved
        if not new_chars:
            continue

        seen_retrieved.update(new_chars)
        retrieved_count += len(new_chars)

        new_relevant = new_chars & gold_chars
        if new_relevant:
            seen_relevant.update(new_relevant)
            precision_now = len(seen_relevant) / max(1, retrieved_count)
            weighted_precision_sum += len(new_relevant) * precision_now

    return float(weighted_precision_sum / total_rel)


def char_precision_at_k_passage2(run_q, k, qid):
    gold_chars = get_gold_chars(qid)
    g = run_q.sort_values('rank').head(k)
    retrieved = set()
    for r in g.itertuples(index=False):
        for pos in range(int(r.start_char), int(r.end_char)):
            retrieved.add(stable_doc_key(r.doc_id, pos))

    if not retrieved:
        return 0.0
    return float(len(retrieved & gold_chars) / len(retrieved))


def evaluate_article_like_char(run_standard, model_name):
    per_q = []
    for qid, g in run_standard.groupby('qid'):
        qid = str(qid)
        gold_chars = get_gold_chars(qid)
        per_q.append({
            'model': model_name,
            'qid': qid,
            'CharAP': char_average_precision_passage2(g, qid),
            'CharP@1': char_precision_at_k_passage2(g, 1, qid),
            'CharP@10': char_precision_at_k_passage2(g, 10, qid),
            'gold_rel_chars': int(len(gold_chars)),
            'num_retrieved_windows': int(len(g)),
        })

    perq = pd.DataFrame(per_q).sort_values(['model', 'qid']).reset_index(drop=True)
    summary = {
        'model': model_name,
        'num_queries': int(perq['qid'].nunique()) if len(perq) else 0,
        'CharMAP': float(perq['CharAP'].mean()) if len(perq) else 0.0,
        'CharP@1': float(perq['CharP@1'].mean()) if len(perq) else 0.0,
        'CharP@10': float(perq['CharP@10'].mean()) if len(perq) else 0.0,
    }
    return perq, summary


perq_char_list = []
summary_char_list = []
for model_name, run in model_runs.items():
    std = to_standard_run(run)
    perq, summ = evaluate_article_like_char(std, model_name)
    perq_char_list.append(perq)
    summary_char_list.append(summ)

per_query_article = pd.concat(perq_char_list, ignore_index=True)
metrics_article = pd.DataFrame(summary_char_list)

save_tsv(per_query_article, PATH_PER_QUERY_ARTICLE)
save_tsv(metrics_article, PATH_METRICS_ARTICLE)

display(metrics_article)

# %%
# =====================================================================
# 14) Évaluation secondaire : window-level binaire
# =====================================================================

# Gold binaire complet par requête, basé sur toutes les fenêtres candidates, pas seulement les fenêtres retrouvées.
gold_window = units[['qid', 'unit_id', 'label_relevance', 'label_strong']].drop_duplicates(['qid', 'unit_id']).copy()
gold_total_relevant_by_qid = gold_window.groupby('qid')['label_strong'].sum().astype(int).to_dict()


def average_precision_window(run_q, qid):
    total_rel = int(gold_total_relevant_by_qid.get(str(qid), 0))
    if total_rel <= 0:
        return 0.0

    hits = 0
    ap_sum = 0.0
    for i, r in enumerate(run_q.sort_values('rank').itertuples(index=False), start=1):
        if int(r.label_strong) > 0:
            hits += 1
            ap_sum += hits / i
    return float(ap_sum / total_rel)


def precision_at_k_window(run_q, k):
    g = run_q.sort_values('rank').head(k)
    if len(g) == 0:
        return 0.0
    return float(g['label_strong'].astype(int).sum() / k)


def dcg_at_k(rels, k):
    rels = list(rels)[:k]
    return sum((2 ** int(rel) - 1) / np.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k_window(run_q, k=10):
    g = run_q.sort_values('rank').head(k)
    dcg = dcg_at_k(g['label_relevance'].astype(int).tolist(), k)

    # IDCG basé sur les labels de toutes les fenêtres candidates de la requête.
    qid = str(g['qid'].iloc[0]) if len(g) else None
    if qid is None:
        return 0.0
    ideal_rels = gold_window[gold_window['qid'].astype(str) == qid]['label_relevance'].astype(int).sort_values(ascending=False).tolist()
    idcg = dcg_at_k(ideal_rels, k)
    return 0.0 if idcg == 0 else float(dcg / idcg)


def evaluate_window_level(run, model_name):
    per_q = []
    for qid, g in run.groupby('qid'):
        qid = str(qid)
        per_q.append({
            'model': model_name,
            'qid': qid,
            'AP': average_precision_window(g, qid),
            'P@1': precision_at_k_window(g, 1),
            'P@10': precision_at_k_window(g, 10),
            'nDCG@10': ndcg_at_k_window(g, 10),
            'num_retrieved': int(len(g)),
            'gold_relevant_windows': int(gold_total_relevant_by_qid.get(qid, 0)),
            'num_rel_in_run': int(g['label_strong'].astype(int).sum()),
        })

    perq = pd.DataFrame(per_q).sort_values(['model', 'qid']).reset_index(drop=True)
    summary = {
        'model': model_name,
        'num_queries': int(perq['qid'].nunique()) if len(perq) else 0,
        'MAP': float(perq['AP'].mean()) if len(perq) else 0.0,
        'P@1': float(perq['P@1'].mean()) if len(perq) else 0.0,
        'P@10': float(perq['P@10'].mean()) if len(perq) else 0.0,
        'nDCG@10': float(perq['nDCG@10'].mean()) if len(perq) else 0.0,
    }
    return perq, summary


perq_win_list = []
summary_win_list = []
for model_name, run in model_runs.items():
    perq, summ = evaluate_window_level(run, model_name)
    perq_win_list.append(perq)
    summary_win_list.append(summ)

per_query_window = pd.concat(perq_win_list, ignore_index=True)
metrics_window = pd.DataFrame(summary_win_list)

save_tsv(per_query_window, PATH_PER_QUERY_WINDOW)
save_tsv(metrics_window, PATH_METRICS_WINDOW)

display(metrics_window)

# %%
# =====================================================================
# 15) Table finale principale + exports# =====================================================================
# Table demandée dans le rapport : on renomme CharMAP/CharP@k en MAP/P@k
# parce que l'évaluation principale est article-like au niveau caractère.

model_order = ['QL', 'SDM', 'QL-Interpolated', 'PM-TFIDF', 'PM-Dirichlet', 'PM-SkewedGaussian']

article_table = metrics_article.copy()
article_table['model'] = pd.Categorical(article_table['model'], categories=model_order, ordered=True)
article_table = article_table.sort_values('model').reset_index(drop=True)
article_table = article_table.rename(columns={
    'CharMAP': 'MAP',
    'CharP@1': 'P@1',
    'CharP@10': 'P@10',
})
article_table = article_table[['model', 'MAP', 'P@1', 'P@10']]

combined = metrics_article.merge(metrics_window, on=['model', 'num_queries'], how='outer', suffixes=('_char', '_window'))
combined['model'] = pd.Categorical(combined['model'], categories=model_order, ordered=True)
combined = combined.sort_values('model').reset_index(drop=True)

save_tsv(article_table, PATH_ARTICLE_TABLE)
save_tsv(combined, PATH_METRICS_COMBINED)

metadata_model = {
    'project': 'FiRA TREC-19 article-like modeling',
    'data_dir': str(DATA_DIR),
    'output_dir': str(MODEL_OUT_DIR),
    'main_dataset_path': str(PATH_MAIN_DATASET),
    'modeling_units_path': str(PATH_MODELING_UNITS),
    'gold_char_spans_path': str(PATH_GOLD_CHAR_SPANS),
    'runs_standard_all_path': str(PATH_RUN_STANDARD_ALL),
    'candidate_document_policy': 'top-50 retrieved documents only, inherited from notebook 01 corrected',
    'unit_type': 'window_50_25',
    'min_strong_relevance': MIN_STRONG_RELEVANCE,
    'topk_eval': TOPK_EVAL,
    'models': model_order,
    'ql_wmodel': QL_WMODEL,
    'sdm_model_used': sdm_model_used,
    'interpolation_alpha': INTERP_ALPHA,
    'pm_mu': PM_MU,
    'pm_sigma': PM_SIGMA,
    'pm_skew_alpha': PM_SKEW_ALPHA,
    'main_evaluation': 'Passage2-like character-level evaluation with duplicate retrieved characters counted once',
    'secondary_evaluation': 'window-level binary MAP/P@1/P@10/nDCG@10',
    'important_note': 'Main table MAP/P@1/P@10 corresponds to CharMAP/CharP@1/CharP@10.',
    'article_like_table': article_table.to_dict(orient='records'),
    'window_metrics': metrics_window.to_dict(orient='records'),
}
PATH_METADATA_MODEL.write_text(json.dumps(metadata_model, indent=2, ensure_ascii=False), encoding='utf-8')

print('=== TABLE PRINCIPALE ARTICLE-LIKE / CHAR-LEVEL ===')
display(article_table)

print('=== TABLE COMBINÉE ===')
display(combined)

print('[OK] Exports terminés dans :', MODEL_OUT_DIR)
for p in sorted(MODEL_OUT_DIR.iterdir()):
    print(' -', p)

# %% [markdown]
# ## 16) Lecture rapide des résultats de base
#
# Cette cellule aide à commenter les résultats des Tables 1/2-like après exécution. Elle ne change pas l'évaluation : elle lit simplement `article_table` et `combined`.

# %%
# =====================================================================
# 16) Lecture rapide des résultats de base
# =====================================================================
comment_rows = []

best_map = article_table.sort_values('MAP', ascending=False).iloc[0]
best_p1 = article_table.sort_values('P@1', ascending=False).iloc[0]
best_p10 = article_table.sort_values('P@10', ascending=False).iloc[0]

comment_rows.append({
    'observation': 'Meilleur MAP char-level',
    'model': best_map['model'],
    'value': float(best_map['MAP'])
})
comment_rows.append({
    'observation': 'Meilleur P@1 char-level',
    'model': best_p1['model'],
    'value': float(best_p1['P@1'])
})
comment_rows.append({
    'observation': 'Meilleur P@10 char-level',
    'model': best_p10['model'],
    'value': float(best_p10['P@10'])
})

if {'QL', 'SDM', 'QL-Interpolated'}.issubset(set(article_table['model'].astype(str))):
    ql_map = float(article_table.loc[article_table['model'].astype(str) == 'QL', 'MAP'].iloc[0])
    sdm_map = float(article_table.loc[article_table['model'].astype(str) == 'SDM', 'MAP'].iloc[0])
    interp_map = float(article_table.loc[article_table['model'].astype(str) == 'QL-Interpolated', 'MAP'].iloc[0])
    comment_rows.append({
        'observation': 'SDM - QL en MAP',
        'model': 'SDM vs QL',
        'value': sdm_map - ql_map
    })
    comment_rows.append({
        'observation': 'QL-Interpolated - QL en MAP',
        'model': 'QL-Interpolated vs QL',
        'value': interp_map - ql_map
    })

comments_base = pd.DataFrame(comment_rows)
display(comments_base)

print("""
Commentaire type pour le rapport :
- Les scores sont nettement plus élevés que ceux de l'article original, ce qui est attendu : FiRA/TREC-DL n'est pas GOV2 et les annotations proviennent de passages plus structurés.
- La comparaison interne est plus importante que la comparaison absolue avec les valeurs de l'article.
- Si PM-SkewedGaussian n'est pas meilleur que QL/SDM, il faut l'interpréter comme une différence de collection + d'alignement FiRA, pas forcément comme une erreur.
- La table suivante à ajouter est la Table 3-like : RM on documents / RM on passages.
""")

# %% [markdown]
# ## 17) Table 3-like — Query Expansion / Relevance Model
#
# L'article teste deux sources d'expansion :
#
# - **RM on documents** : termes pris depuis les 25 documents de feedback.
# - **RM on passages** : termes pris depuis les 25 passages/fenêtres de feedback.
#
# Ici, les requêtes étendues sont scorées avec **PM-SkewedGaussian**, comme dans l'article. Les labels FiRA ne sont jamais utilisés pour choisir les termes d'expansion.

# %%
# =====================================================================
# 17) Query Expansion Methods — Table 3-like
# =====================================================================
# Paramètres article-like.
RM_FEEDBACK_ITEMS = 25
RM_FEEDBACK_TERMS = 25
RM_ORIGINAL_WEIGHT = 0.85
RM_EXPANSION_WEIGHT = 0.15

PATH_RM_TERMS = MODEL_OUT_DIR / 'query_expansion_terms_rm.tsv'
PATH_RM_RUNS = MODEL_OUT_DIR / 'runs_query_expansion_standard.tsv'
PATH_RM_TABLE_CHAR = MODEL_OUT_DIR / 'table3_query_expansion_article_like_char.tsv'
PATH_RM_TABLE_WINDOW = MODEL_OUT_DIR / 'table3_query_expansion_secondary_window.tsv'

STOPWORDS = set("""
a an the and or of in on for to from with without by as is are was were be been being
this that these those it its into about what who whom whose which when where why how
do does did can could should would may might will shall not no yes
un une des le la les de du au aux et ou en dans sur pour par avec sans est sont
""".split())


def valid_expansion_term(t, original_terms):
    t = str(t).lower()
    if len(t) <= 2:
        return False
    if t in STOPWORDS:
        return False
    if t in original_terms:
        return False
    if t.isdigit():
        return False
    return True


def get_original_query_weights(qid):
    q = units.loc[units['qid'].astype(str) == str(qid), 'query_clean']
    if len(q) == 0:
        return {}
    return normalize_query_weights(tok(q.iloc[0]))


def extract_rm_terms_from_tokens(tokens, original_terms, top_terms=RM_FEEDBACK_TERMS):
    """
    Approximation légère du RM : termes fréquents dans le feedback, pondérés par TF * IDF.
    C'est volontairement non supervisé : aucun label n'est utilisé.
    """
    original_terms = set(original_terms)
    counts = Counter(t for t in tokens if valid_expansion_term(t, original_terms))
    if not counts:
        return {}

    scored = {}
    for t, c in counts.items():
        scored[t] = float(c) * idf(t)

    top = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_terms]
    total = sum(max(0.0, w) for _, w in top)
    if total <= 0:
        return {}
    return {t: max(0.0, w) / total for t, w in top}


def interpolate_original_and_expansion(qid, expansion_weights):
    original = get_original_query_weights(qid)
    if not original:
        return {}

    out = defaultdict(float)
    for t, w in original.items():
        out[t] += RM_ORIGINAL_WEIGHT * float(w)

    if expansion_weights:
        for t, w in expansion_weights.items():
            out[t] += RM_EXPANSION_WEIGHT * float(w)
    else:
        # Si aucun terme d'expansion n'est trouvé, on garde la requête originale à poids total 1.
        out = defaultdict(float)
        for t, w in original.items():
            out[t] += float(w)

    return dict(out)


def build_rm_document_query_weights():
    """
    RM on documents : top 25 documents SDM initiaux par requête.
    On utilise initial_rank hérité du notebook 01.
    """
    qweights = {}
    term_rows = []

    doc_candidates = (
        units[['qid', 'doc_id', 'initial_rank']]
        .drop_duplicates(['qid', 'doc_id'])
        .sort_values(['qid', 'initial_rank'])
    )

    for qid, g in tqdm(doc_candidates.groupby('qid'), desc='RM document terms'):
        qid = str(qid)
        original_terms = set(get_original_query_weights(qid).keys())
        top_docs = g[g['initial_rank'].astype(int) <= RM_FEEDBACK_ITEMS]['doc_id'].astype(str).tolist()

        feedback_tokens = []
        for did in top_docs:
            off = doc_offsets.get(str(did))
            if off:
                feedback_tokens.extend(off['tokens'])

        exp_weights = extract_rm_terms_from_tokens(feedback_tokens, original_terms, RM_FEEDBACK_TERMS)
        qweights[qid] = interpolate_original_and_expansion(qid, exp_weights)

        for term, weight in exp_weights.items():
            term_rows.append({'qid': qid, 'source': 'documents', 'term': term, 'rm_weight': weight})

    return qweights, term_rows


def build_rm_passage_query_weights(feedback_run):
    """
    RM on passages : top 25 fenêtres récupérées par PM-SkewedGaussian.
    """
    qweights = {}
    term_rows = []

    unit_text_map = dict(zip(units['unit_id'].astype(str), units['unit_text'].astype(str)))

    for qid, g in tqdm(feedback_run.groupby('qid'), desc='RM passage terms'):
        qid = str(qid)
        original_terms = set(get_original_query_weights(qid).keys())
        top_units = g.sort_values('rank').head(RM_FEEDBACK_ITEMS)['unit_id'].astype(str).tolist()

        feedback_tokens = []
        for uid in top_units:
            feedback_tokens.extend(tok(unit_text_map.get(uid, '')))

        exp_weights = extract_rm_terms_from_tokens(feedback_tokens, original_terms, RM_FEEDBACK_TERMS)
        qweights[qid] = interpolate_original_and_expansion(qid, exp_weights)

        for term, weight in exp_weights.items():
            term_rows.append({'qid': qid, 'source': 'passages', 'term': term, 'rm_weight': weight})

    return qweights, term_rows


# Construction des requêtes étendues.
rm_doc_weights, rm_doc_terms = build_rm_document_query_weights()
rm_passage_weights, rm_passage_terms = build_rm_passage_query_weights(run_pm_skewed)

rm_terms = pd.DataFrame(rm_doc_terms + rm_passage_terms)
save_tsv(rm_terms, PATH_RM_TERMS)

print('[INFO] Exemples de termes RM :')
display(rm_terms.head(20))

# Scoring PM-SkewedGaussian avec les requêtes étendues.
run_rm_docs = build_pm_run_for_units(
    units_subset=units,
    model_name='RM on documents',
    scoring_kind='skewed',
    query_weights_by_qid=rm_doc_weights,
)

run_rm_passages = build_pm_run_for_units(
    units_subset=units,
    model_name='RM on passages',
    scoring_kind='skewed',
    query_weights_by_qid=rm_passage_weights,
)

# Runs standardisés.
runs_rm_standard = pd.concat([
    to_standard_run(run_pm_skewed.assign(model='PM-SkewedGaussian')),
    to_standard_run(run_rm_docs),
    to_standard_run(run_rm_passages),
], ignore_index=True)
save_tsv(runs_rm_standard, PATH_RM_RUNS)

# Évaluation char-level principale.
rm_char_rows = []
for model_name, run in [
    ('PM-SkewedGaussian', run_pm_skewed),
    ('RM on documents', run_rm_docs),
    ('RM on passages', run_rm_passages),
]:
    perq, summ = evaluate_article_like_char(to_standard_run(run), model_name)
    rm_char_rows.append({
        'model': model_name,
        'MAP': summ['CharMAP'],
        'P@1': summ['CharP@1'],
        'P@10': summ['CharP@10'],
    })

table3_char = pd.DataFrame(rm_char_rows)
save_tsv(table3_char, PATH_RM_TABLE_CHAR)

# Évaluation secondaire window-level.
rm_window_rows = []
for model_name, run in [
    ('PM-SkewedGaussian', run_pm_skewed),
    ('RM on documents', run_rm_docs),
    ('RM on passages', run_rm_passages),
]:
    perq, summ = evaluate_window_level(run, model_name)
    rm_window_rows.append({
        'model': model_name,
        'MAP': summ['MAP'],
        'P@1': summ['P@1'],
        'P@10': summ['P@10'],
        'nDCG@10': summ['nDCG@10'],
    })

table3_window = pd.DataFrame(rm_window_rows)
save_tsv(table3_window, PATH_RM_TABLE_WINDOW)

print('=== TABLE 3-LIKE : QUERY EXPANSION / CHAR-LEVEL ===')
display(table3_char)

print('=== TABLE 3-LIKE : QUERY EXPANSION / WINDOW-LEVEL ===')
display(table3_window)

# %% [markdown]
# ## 18) Table 4-like — Effect of input document quality
#
# On garde le modèle principal **PM-SkewedGaussian** et on change uniquement les documents candidats :
#
# - Top 5 documents initiaux ;
# - Top 10 ;
# - Top 25 ;
# - Top 50 ;
# - Only Relevant : oracle limité aux documents du top-50 qui contiennent au moins une fenêtre fortement pertinente.
#
# Le gold standard reste identique : on ne réduit pas le gold quand on réduit les candidats.

# %%
# =====================================================================
# 18) Effect of Input Document Quality — Table 4-like
# =====================================================================
PATH_INPUT_QUALITY_TABLE_CHAR = MODEL_OUT_DIR / 'table4_input_document_quality_article_like_char.tsv'
PATH_INPUT_QUALITY_TABLE_WINDOW = MODEL_OUT_DIR / 'table4_input_document_quality_secondary_window.tsv'
PATH_INPUT_QUALITY_RUNS = MODEL_OUT_DIR / 'runs_input_document_quality_standard.tsv'

ALL_QIDS = sorted(units['qid'].astype(str).unique())


def evaluate_article_like_char_over_all_qids(run_standard, model_name, all_qids=ALL_QIDS):
    """
    Variante robuste : les qids absents du run comptent 0.
    Utile pour Top5 ou Only Relevant si une requête n'a aucun candidat.
    """
    run_standard = run_standard.copy()
    if len(run_standard):
        run_standard['qid'] = run_standard['qid'].astype(str)

    per_q = []
    for qid in all_qids:
        g = run_standard[run_standard['qid'].astype(str) == str(qid)] if len(run_standard) else pd.DataFrame()
        gold_chars = get_gold_chars(qid)

        if len(g) == 0:
            char_ap = 0.0
            char_p1 = 0.0
            char_p10 = 0.0
        else:
            char_ap = char_average_precision_passage2(g, qid)
            char_p1 = char_precision_at_k_passage2(g, 1, qid)
            char_p10 = char_precision_at_k_passage2(g, 10, qid)

        per_q.append({
            'model': model_name,
            'qid': str(qid),
            'CharAP': char_ap,
            'CharP@1': char_p1,
            'CharP@10': char_p10,
            'gold_rel_chars': int(len(gold_chars)),
            'num_retrieved_windows': int(len(g)),
        })

    perq = pd.DataFrame(per_q)
    summary = {
        'model': model_name,
        'num_queries': int(len(all_qids)),
        'CharMAP': float(perq['CharAP'].mean()),
        'CharP@1': float(perq['CharP@1'].mean()),
        'CharP@10': float(perq['CharP@10'].mean()),
    }
    return perq, summary


def evaluate_window_level_over_all_qids(run, model_name, all_qids=ALL_QIDS):
    """
    Variante robuste : les qids absents du run comptent 0.
    """
    run = run.copy()
    if len(run):
        run['qid'] = run['qid'].astype(str)

    per_q = []
    for qid in all_qids:
        g = run[run['qid'].astype(str) == str(qid)] if len(run) else pd.DataFrame()

        if len(g) == 0:
            ap = 0.0
            p1 = 0.0
            p10 = 0.0
            ndcg10 = 0.0
            num_rel_in_run = 0
        else:
            ap = average_precision_window(g, qid)
            p1 = precision_at_k_window(g, 1)
            p10 = precision_at_k_window(g, 10)
            ndcg10 = ndcg_at_k_window(g, 10)
            num_rel_in_run = int(g['label_strong'].astype(int).sum())

        per_q.append({
            'model': model_name,
            'qid': str(qid),
            'AP': ap,
            'P@1': p1,
            'P@10': p10,
            'nDCG@10': ndcg10,
            'num_retrieved': int(len(g)),
            'gold_relevant_windows': int(gold_total_relevant_by_qid.get(str(qid), 0)),
            'num_rel_in_run': num_rel_in_run,
        })

    perq = pd.DataFrame(per_q)
    summary = {
        'model': model_name,
        'num_queries': int(len(all_qids)),
        'MAP': float(perq['AP'].mean()),
        'P@1': float(perq['P@1'].mean()),
        'P@10': float(perq['P@10'].mean()),
        'nDCG@10': float(perq['nDCG@10'].mean()),
    }
    return perq, summary


# Documents oracle : documents du top-50 qui contiennent au moins une fenêtre label>=2.
relevant_doc_keys = (
    units[units['label_strong'].astype(int) == 1][['qid', 'doc_id']]
    .drop_duplicates()
    .assign(_keep=1)
)

units_only_relevant_docs = units.merge(relevant_doc_keys, on=['qid', 'doc_id'], how='inner').drop(columns=['_keep'])

input_configs = [
    ('Top 5', units[units['initial_rank'].astype(int) <= 5].copy()),
    ('Top 10', units[units['initial_rank'].astype(int) <= 10].copy()),
    ('Top 25', units[units['initial_rank'].astype(int) <= 25].copy()),
    ('Top 50', units.copy()),
    ('Only Relevant', units_only_relevant_docs.copy()),
]

table4_char_rows = []
table4_window_rows = []
input_quality_runs = []

for label, units_subset in input_configs:
    print(f'[INPUT QUALITY] {label}: {len(units_subset)} fenêtres, {units_subset.doc_id.nunique()} documents')

    if label == 'Top 50':
        run = run_pm_skewed.copy()
        run['model'] = label
    else:
        run = build_pm_run_for_units(
            units_subset=units_subset,
            model_name=label,
            scoring_kind='skewed',
            query_weights_by_qid=None,
        )

    std = to_standard_run(run)
    input_quality_runs.append(std.assign(input_setting=label))

    perq_char, summ_char = evaluate_article_like_char_over_all_qids(std, label)
    table4_char_rows.append({
        'Input docs': label,
        'MAP': summ_char['CharMAP'],
        'P@1': summ_char['CharP@1'],
        'P@10': summ_char['CharP@10'],
        'num_queries': summ_char['num_queries'],
        'num_windows': int(len(units_subset)),
        'num_docs': int(units_subset[['qid', 'doc_id']].drop_duplicates().shape[0]),
    })

    perq_win, summ_win = evaluate_window_level_over_all_qids(run, label)
    table4_window_rows.append({
        'Input docs': label,
        'MAP': summ_win['MAP'],
        'P@1': summ_win['P@1'],
        'P@10': summ_win['P@10'],
        'nDCG@10': summ_win['nDCG@10'],
        'num_queries': summ_win['num_queries'],
        'num_windows': int(len(units_subset)),
        'num_docs': int(units_subset[['qid', 'doc_id']].drop_duplicates().shape[0]),
    })

table4_char = pd.DataFrame(table4_char_rows)
table4_window = pd.DataFrame(table4_window_rows)
runs_input_quality_standard = pd.concat(input_quality_runs, ignore_index=True)

save_tsv(table4_char, PATH_INPUT_QUALITY_TABLE_CHAR)
save_tsv(table4_window, PATH_INPUT_QUALITY_TABLE_WINDOW)
save_tsv(runs_input_quality_standard, PATH_INPUT_QUALITY_RUNS)

print('=== TABLE 4-LIKE : INPUT DOCUMENT QUALITY / CHAR-LEVEL ===')
display(table4_char)

print('=== TABLE 4-LIKE : INPUT DOCUMENT QUALITY / WINDOW-LEVEL ===')
display(table4_window)

# %% [markdown]
# ## 19) Figure 1-like — Score distribution analysis
#
# L'objectif est de comparer les scores des fenêtres top-rankées avec les scores des fenêtres qui contiennent vraiment une réponse. Cela sert à la discussion : si les réponses ont souvent des scores plus faibles, les modèles basés sur les termes de requête ratent une partie des vrais passages-réponses.

# %%
# =====================================================================
# 19) Figure 1-like — Score distribution comme dans l'article
# =====================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

PATH_FIGURE1_LIKE = MODEL_OUT_DIR / "figure1_like_article_style.png"
PATH_FIGURE1_LEFT = MODEL_OUT_DIR / "figure1_left_score_distribution_article_style.png"
PATH_FIGURE1_RIGHT = MODEL_OUT_DIR / "figure1_right_mean_score_by_topic_article_style.png"
PATH_SCORE_ANALYSIS = MODEL_OUT_DIR / "figure1_like_score_analysis_data.tsv"

# ---------------------------------------------------------------------
# 1) Préparer les scores
# ---------------------------------------------------------------------
# On utilise PM-SkewedGaussian, comme dans la discussion de l'article.
# raw_pm_skewed contient normalement les scores pour toutes les fenêtres candidates.
raw_scores = raw_pm_skewed.copy()
raw_scores["qid"] = raw_scores["qid"].astype(str)
raw_scores["unit_id"] = raw_scores["unit_id"].astype(str)
raw_scores["score"] = pd.to_numeric(raw_scores["score"], errors="coerce").fillna(0.0)

score_meta = units[["qid", "unit_id", "doc_id", "label_strong", "label_relevance"]].copy()
score_meta["qid"] = score_meta["qid"].astype(str)
score_meta["unit_id"] = score_meta["unit_id"].astype(str)
score_meta["label_strong"] = pd.to_numeric(score_meta["label_strong"], errors="coerce").fillna(0).astype(int)

all_scored = raw_scores.merge(score_meta, on=["qid", "unit_id"], how="left")
all_scored["label_strong"] = all_scored["label_strong"].fillna(0).astype(int)

# Top 20 passages récupérés par requête
top20 = run_pm_skewed[run_pm_skewed["rank"].astype(int) <= 20][["qid", "unit_id", "score"]].copy()
top20["qid"] = top20["qid"].astype(str)
top20["unit_id"] = top20["unit_id"].astype(str)
top20["score"] = pd.to_numeric(top20["score"], errors="coerce").fillna(0.0)
top20["group"] = "Top 20 retrieved passages"

# Passages/fenêtres annotés comme réponses : label FiRA >= 2
answers = all_scored[all_scored["label_strong"] == 1][["qid", "unit_id", "score"]].copy()
answers["score"] = pd.to_numeric(answers["score"], errors="coerce").fillna(0.0)
answers["group"] = "Answer passages"

score_analysis_df = pd.concat([top20, answers], ignore_index=True)
score_analysis_df.to_csv(PATH_SCORE_ANALYSIS, sep="\t", index=False)

print("[INFO] Top20 scores:", len(top20))
print("[INFO] Answer passage scores:", len(answers))
print("[SAVE]", PATH_SCORE_ANALYSIS)

# ---------------------------------------------------------------------
# 2) Helper : KDE article-like
# ---------------------------------------------------------------------
def kde_curve(scores, x_grid):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    if len(scores) < 2 or np.std(scores) == 0:
        return np.zeros_like(x_grid)

    kde = gaussian_kde(scores)
    y = kde(x_grid)

    # Normalisation pour obtenir une courbe de "probability" entre 0 et 1,
    # proche du rendu visuel de l'article.
    if y.max() > 0:
        y = y / y.max()

    return y

top_scores = top20["score"].astype(float).to_numpy()
ans_scores = answers["score"].astype(float).to_numpy()

all_scores = np.concatenate([top_scores, ans_scores])
all_scores = all_scores[np.isfinite(all_scores)]

x_min = np.percentile(all_scores, 1)
x_max = np.percentile(all_scores, 99)

# marge visuelle
margin = 0.10 * (x_max - x_min) if x_max > x_min else 1.0
x_grid = np.linspace(x_min - margin, x_max + margin, 400)

y_top = kde_curve(top_scores, x_grid)
y_ans = kde_curve(ans_scores, x_grid)

# ---------------------------------------------------------------------
# 3) Figure gauche : Score distribution
# ---------------------------------------------------------------------
plt.figure(figsize=(6, 4))

plt.plot(
    x_grid,
    y_top,
    linestyle="-",
    linewidth=2,
    label="retrieved"
)

plt.plot(
    x_grid,
    y_ans,
    linestyle="--",
    linewidth=2,
    label="relevant"
)

plt.xlabel("score")
plt.ylabel("probability")
plt.title("Score distribution")
plt.legend(
    title="",
    frameon=False,
    loc="upper left"
)

# Style plus proche article : sobre, sans grille
plt.grid(False)
plt.tight_layout()
plt.savefig(PATH_FIGURE1_LEFT, dpi=300, bbox_inches="tight")
plt.show()

print("[SAVE]", PATH_FIGURE1_LEFT)

# ---------------------------------------------------------------------
# 4) Figure droite : Distribution of the mean of passage scores per topic
# ---------------------------------------------------------------------
top20_mean = (
    top20.groupby("qid")["score"]
    .mean()
    .rename("Top 20 retrieved passages")
)

answers_mean = (
    answers.groupby("qid")["score"]
    .mean()
    .rename("Answer passages")
)

mean_by_qid = pd.concat([top20_mean, answers_mean], axis=1).reset_index()

# On garde seulement les topics où les deux valeurs existent
mean_by_qid = mean_by_qid.dropna(subset=["Top 20 retrieved passages", "Answer passages"]).copy()

# Pour ressembler davantage à l'article :
# on trie par score moyen des top20 passages, au lieu de garder l'ordre arbitraire des qid.
mean_by_qid = mean_by_qid.sort_values("Top 20 retrieved passages").reset_index(drop=True)
mean_by_qid["topic_index"] = np.arange(len(mean_by_qid))

plt.figure(figsize=(6, 4))

plt.plot(
    mean_by_qid["topic_index"],
    mean_by_qid["Top 20 retrieved passages"],
    linestyle="-",
    linewidth=2,
    label="Top 20 retrieved passages"
)

plt.plot(
    mean_by_qid["topic_index"],
    mean_by_qid["Answer passages"],
    linestyle="--",
    linewidth=2,
    label="Answer passages"
)

plt.xlabel("topic")
plt.ylabel("mean score")
plt.title("Distribution of the mean of passage scores per topic")
plt.legend(frameon=False, loc="best")

plt.grid(False)
plt.tight_layout()
plt.savefig(PATH_FIGURE1_RIGHT, dpi=300, bbox_inches="tight")
plt.show()

print("[SAVE]", PATH_FIGURE1_RIGHT)

# ---------------------------------------------------------------------
# 5) Figure combinée comme dans l'article : deux graphes côte à côte
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gauche
axes[0].plot(x_grid, y_top, linestyle="-", linewidth=2, label="retrieved")
axes[0].plot(x_grid, y_ans, linestyle="--", linewidth=2, label="relevant")
axes[0].set_xlabel("score")
axes[0].set_ylabel("probability")
axes[0].set_title("Score distribution")
axes[0].legend(frameon=False, loc="upper left")
axes[0].grid(False)

# Droite
axes[1].plot(
    mean_by_qid["topic_index"],
    mean_by_qid["Top 20 retrieved passages"],
    linestyle="-",
    linewidth=2,
    label="Top 20 retrieved passages"
)

axes[1].plot(
    mean_by_qid["topic_index"],
    mean_by_qid["Answer passages"],
    linestyle="--",
    linewidth=2,
    label="Answer passages"
)

axes[1].set_xlabel("topic")
axes[1].set_ylabel("mean score")
axes[1].set_title("Distribution of the mean of passage scores per topic")
axes[1].legend(frameon=False, loc="best")
axes[1].grid(False)

plt.tight_layout()
plt.savefig(PATH_FIGURE1_LIKE, dpi=300, bbox_inches="tight")
plt.show()

print("[SAVE]", PATH_FIGURE1_LIKE)

# ---------------------------------------------------------------------
# 6) Résumé numérique pour la discussion
# ---------------------------------------------------------------------
print("Moyenne Top20 retrieved passages:", float(top20["score"].mean()))
print("Moyenne Answer passages:", float(answers["score"].mean()))

if float(answers["score"].mean()) < float(top20["score"].mean()):
    print("[DISCUSSION] Les passages-réponses ont en moyenne un score plus faible que les passages top-ranked.")
    print("[DISCUSSION] Conclusion proche de l'article : les scores lexicaux/positionnels ne suffisent pas à identifier les vraies réponses.")
else:
    print("[DISCUSSION] Les passages-réponses ont un score moyen comparable ou supérieur.")
    print("[DISCUSSION] Différence possible liée à FiRA/TREC-DL, aux documents reconstruits ou à la granularité des annotations.")

display(mean_by_qid.head(20))

# %%
# =====================================================================
# 21) Démo qualitative : requête + meilleur passage du meilleur modèle
# =====================================================================

PATH_DEMO_TOP_RESULTS = MODEL_OUT_DIR / "demo_best_model_top_results.tsv"
PATH_DEMO_MARKDOWN = MODEL_OUT_DIR / "demo_best_model_example.md"

# 1) Choisir automatiquement le meilleur modèle selon la table principale
metric_col = "CharMAP" if "CharMAP" in metrics_article.columns else "MAP"
best_model = metrics_article.sort_values(metric_col, ascending=False).iloc[0]["model"]

print("[INFO] Meilleur modèle détecté :", best_model)

best_run = model_runs[best_model].copy()
best_run["qid"] = best_run["qid"].astype(str)
best_run["unit_id"] = best_run["unit_id"].astype(str)

# 2) Choisir une requête intéressante :
# priorité à une requête où le meilleur modèle a au moins un passage pertinent dans le top 5
demo_candidates = best_run[best_run["rank"].astype(int) <= 5].copy()

if "label_strong" in demo_candidates.columns:
    demo_candidates = demo_candidates[demo_candidates["label_strong"].astype(int) == 1]

if len(demo_candidates) > 0:
    demo_qid = str(demo_candidates.iloc[0]["qid"])
else:
    demo_qid = str(best_run.iloc[0]["qid"])

print("[INFO] Requête démo qid =", demo_qid)

# 3) Récupérer top passages
demo = best_run[best_run["qid"] == demo_qid].sort_values("rank").head(5).copy()

meta_cols = [
    "qid", "query", "doc_id", "unit_id", "window_id",
    "unit_text", "label_relevance", "label_strong",
    "initial_rank", "start_char", "end_char"
]

demo = demo.merge(
    units[meta_cols].drop_duplicates(["qid", "unit_id"]),
    on=["qid", "unit_id"],
    how="left",
    suffixes=("", "_meta")
)

# éviter colonnes dupliquées si déjà présentes
for col in ["doc_id", "start_char", "end_char"]:
    alt = f"{col}_meta"
    if alt in demo.columns:
        demo[col] = demo[col].fillna(demo[alt])
        demo = demo.drop(columns=[alt])

demo["model"] = best_model
demo["rank"] = demo["rank"].astype(int)
demo["score"] = pd.to_numeric(demo["score"], errors="coerce")

demo_export = demo[
    [
        "model", "qid", "query", "rank", "score",
        "doc_id", "window_id", "label_relevance",
        "label_strong", "unit_text"
    ]
].copy()

save_tsv(demo_export, PATH_DEMO_TOP_RESULTS)

# 4) Affichage lisible
query_text = demo_export.iloc[0]["query"]
top_passage = demo_export.iloc[0]["unit_text"]
top_label = int(demo_export.iloc[0]["label_relevance"])
top_score = float(demo_export.iloc[0]["score"])

print("\n================ DÉMO QUALITATIVE ================")
print("Modèle :", best_model)
print("QID :", demo_qid)
print("Requête :", query_text)
print("Top passage score :", round(top_score, 4))
print("Label FiRA :", top_label)
print("\nTop passage :\n")
print(top_passage[:1200])

# 5) Markdown prêt pour slide/rapport
md = f"""# Démo qualitative — meilleur modèle

**Modèle :** {best_model}

**QID :** {demo_qid}

**Requête :** {query_text}

**Top passage — score :** {top_score:.4f}

**Label FiRA :** {top_label}

## Passage retourné

{top_passage}
"""

PATH_DEMO_MARKDOWN.write_text(md, encoding="utf-8")

print("\n[SAVE]", PATH_DEMO_TOP_RESULTS)
print("[SAVE]", PATH_DEMO_MARKDOWN)

display(demo_export)

# %% [markdown]
# ## 20) Exports finaux
#
# Cette cellule écrit un fichier de métadonnées global pour savoir quels artefacts correspondent aux tables/figures du rapport.

# %%
# =====================================================================
# 20) Metadata final
# =====================================================================
PATH_FINAL_METADATA = MODEL_OUT_DIR / 'full_article_eval_metadata.json'

final_metadata = {
    'project': 'FiRA TREC-19 article-like full evaluation',
    'data_dir': str(DATA_DIR),
    'output_dir': str(MODEL_OUT_DIR),
    'main_dataset_path': str(PATH_MAIN_DATASET),
    'unit_type': 'window_50_25',
    'candidate_document_policy': 'top-50 SDM documents only from corrected notebook 01',
    'main_evaluation': 'Passage2-like character-level evaluation',
    'secondary_evaluation': 'window-level binary evaluation',
    'tables': {
        'table1_table2_main_models': str(PATH_ARTICLE_TABLE),
        'table3_query_expansion_char': str(PATH_RM_TABLE_CHAR) if 'PATH_RM_TABLE_CHAR' in globals() else None,
        'table3_query_expansion_window': str(PATH_RM_TABLE_WINDOW) if 'PATH_RM_TABLE_WINDOW' in globals() else None,
        'table4_input_quality_char': str(PATH_INPUT_QUALITY_TABLE_CHAR) if 'PATH_INPUT_QUALITY_TABLE_CHAR' in globals() else None,
        'table4_input_quality_window': str(PATH_INPUT_QUALITY_TABLE_WINDOW) if 'PATH_INPUT_QUALITY_TABLE_WINDOW' in globals() else None,
    },
    'figures': {
        'figure1_score_distribution_global': str(PATH_FIG_SCORE_GLOBAL) if 'PATH_FIG_SCORE_GLOBAL' in globals() else None,
        'figure1_score_mean_by_query': str(PATH_FIG_SCORE_BY_QUERY) if 'PATH_FIG_SCORE_BY_QUERY' in globals() else None,
    },
    'important_notes': [
        'PM-TFIDF uses positional pseudo-frequency, not direct window TF.',
        'RM terms are selected without using relevance labels.',
        'Only Relevant is an oracle over relevant documents available inside the top-50 candidate pool.',
        'Do not compare absolute scores directly to GOV2; compare trends because the dataset is FiRA/TREC-DL.'
    ],
}

PATH_FINAL_METADATA.write_text(json.dumps(final_metadata, indent=2, ensure_ascii=False), encoding='utf-8')
print('[OK] Full article-like evaluation notebook finished.')
print('[SAVE]', PATH_FINAL_METADATA)
print('[OUT]', MODEL_OUT_DIR)