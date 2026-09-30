# FiRA/TREC-DL passage retrieval experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # FiRA / TREC-DL 2019 — dataset principal article-like avec fenêtres 50/25
#
# Ce notebook construit le **dataset principal** pour reproduire le protocole de l'article de façon plus fidèle.
#
# Objectif de cette version :
#
# 1. charger les queries + qrels FiRA/TREC-DL 2019 ;
# 2. reconstruire un petit corpus document-first à partir des snippets FiRA ;
# 3. indexer les documents reconstruits ;
# 4. récupérer les **top-50 documents** par requête ;
# 5. **ne pas ajouter automatiquement** les documents pertinents hors top-50 ;
# 6. garder aussi les documents top-50 non pertinents ou sans annotation ;
# 7. découper tous les documents candidats en fenêtres de **50 tokens avec overlap 25** ;
# 8. attribuer à chaque fenêtre un label FiRA :
#    - `fira_label = 0` si aucune annotation FiRA ne chevauche la fenêtre ;
#    - sinon le meilleur label FiRA qui chevauche la fenêtre ;
#    - `is_relevant = 1` seulement si `fira_label >= 2`.
#
# Le fichier principal produit est :
#
# ```text
# windows_top50_labeled_main.tsv
# ```
#
# avec les colonnes principales :
#
# ```text
# qid | doc_id | doc_rank | window_id | window_start | window_end | window_text | fira_label | is_relevant
# ```
#
# Note importante : FiRA ne donne pas le corpus GOV2 complet utilisé dans l'article. Ce notebook reste donc une **reproduction adaptée**, mais il respecte la logique importante : document-first, top-50 documents, fenêtres 50/25, et labels passage/window dérivés des annotations fines.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# # Version FiRA — avec fenêtres 50/25

# %%
# =====================================================================
# 1) Installation, imports, Drive, paramètres
# =====================================================================
# NOTEBOOK: !pip -q install python-terrier ir_datasets pandas numpy tqdm

import os, re, json, shutil, csv
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from tqdm import tqdm
import ir_datasets
import pyterrier as pt
from google.colab import drive

drive.mount('/content/drive')

if hasattr(pt, 'java'):
    if not pt.java.started():
        pt.java.init()
else:
    if not pt.started():
        pt.init()

SEED = 42
np.random.seed(SEED)

TOPK_DOCS = 50
N_FINAL_QUERIES = None          # None = toutes les requêtes disponibles dans FiRA/TREC-DL 2019
MIN_STRONG_RELEVANCE = 2        # article-like : seuls les labels >= 2 sont considérés pertinents
WINDOW_SIZE = 50
WINDOW_OVERLAP = 25

# Correction principale de l'étape 1 :
# False = on garde uniquement les documents récupérés dans le top-50 SDM/DirichletLM.
# On n'ajoute plus les documents pertinents FiRA qui ne sont pas dans le top-50.
INCLUDE_ALL_AVAILABLE_RELEVANT_DOCS = False

# Une fenêtre est annotée si elle chevauche au moins 1 token d'un passage FiRA annoté.
# Le label binaire final reste is_relevant = int(fira_label >= MIN_STRONG_RELEVANCE).
MIN_WINDOW_OVERLAP_RATIO = 0.0

DRIVE_ROOT = Path('/content/drive/MyDrive/IR_project_backup')

# Nouveau nom pour éviter de recharger les anciens caches biaisés.
EXPERIMENT_NAME = 'fira_trec19_article_like_with_windows_top50_only'
APPROACH_NAME = 'document_first_top50_with_windows_fira_small'
ARTIFACT_DIR = DRIVE_ROOT / EXPERIMENT_NAME
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

FIRA_REPO_DIR = ARTIFACT_DIR / 'fira-trec-19-dataset'
FIRA_GITHUB = 'https://github.com/sebastian-hofstaetter/fira-trec-19-dataset.git'

DOC_INDEX_DIR = ARTIFACT_DIR / 'pt_fira_doc_index'
PATH_QUERIES = ARTIFACT_DIR / 'queries_final.tsv'
PATH_DOC_QRELS = ARTIFACT_DIR / 'fira_doc_qrels.tsv'
PATH_PASS_QRELS = ARTIFACT_DIR / 'fira_snippet_qrels.tsv'
PATH_SNIPPETS = ARTIFACT_DIR / 'fira_snippets_standardized.tsv'
PATH_RECON_DOCS = ARTIFACT_DIR / 'fira_reconstructed_documents.tsv'
PATH_ALL_SNIPPET_SPANS = ARTIFACT_DIR / 'fira_all_snippet_spans.tsv'
PATH_TOPDOCS = ARTIFACT_DIR / f'topdocs_top{TOPK_DOCS}.tsv'
PATH_CANDIDATE_TOPDOCS = ARTIFACT_DIR / f'candidate_topdocs_top{TOPK_DOCS}.tsv'
PATH_RELEVANT_TOPDOCS = PATH_CANDIDATE_TOPDOCS  # alias pour compatibilité avec d'anciens notebooks
PATH_ALIGNED_SPANS = ARTIFACT_DIR / 'fira_aligned_snippet_spans_in_top50.tsv'
PATH_SUMMARY = ARTIFACT_DIR / 'summary_by_qid.tsv'
PATH_METADATA = ARTIFACT_DIR / 'dataset_metadata.json'
PATH_WINDOWS = ARTIFACT_DIR / 'windows.tsv'
PATH_WINDOW_LABELS = ARTIFACT_DIR / 'window_labels.tsv'
PATH_MAIN_DATASET = ARTIFACT_DIR / 'windows_top50_labeled_main.tsv'

print('Dossier Drive:', ARTIFACT_DIR)
print('SEED =', SEED)
print('IMPORTANT: INCLUDE_ALL_AVAILABLE_RELEVANT_DOCS =', INCLUDE_ALL_AVAILABLE_RELEVANT_DOCS)

# %%
# =====================================================================
# 2) Fonctions utilitaires générales
# =====================================================================
def clean_query_for_terrier(text):
    text = str(text).lower()
    text = re.sub(r'[_]+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tok(text):
    return re.findall(r'\w+', str(text).lower())

def save_tsv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep='\t', index=False)
    print(f'[SAVE] {path} ({len(df)} lignes)')

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[SAVE] {path}')

def relevance_name(rel):
    rel = int(rel)
    return {
        0: 'non_relevant',
        1: 'topic_relevant_does_not_answer',
        2: 'good_answer',
        3: 'perfect_answer',
    }.get(rel, f'rel_{rel}')

def get_doc_text(doc):
    parts = []
    for attr in ['title', 'body', 'text']:
        if hasattr(doc, attr):
            val = getattr(doc, attr)
            if val:
                parts.append(str(val))
    return '\n'.join(parts).strip()

def get_passage_text(p):
    if hasattr(p, 'text'):
        return str(p.text)
    if hasattr(p, 'body'):
        return str(p.body)
    return str(p)

def find_passage_span_in_doc(passage_text, doc_text, min_score=0.55):
    """Conservé depuis l'ancien notebook.
    Dans FiRA, on connaît normalement les offsets des snippets reconstruits,
    donc cette fonction sert seulement de fallback si nécessaire.
    """
    p_tokens = tok(passage_text)
    d_tokens = tok(doc_text)
    if not p_tokens or not d_tokens:
        return None

    p_norm = ' '.join(p_tokens)
    d_norm = ' '.join(d_tokens)
    idx = d_norm.find(p_norm)
    if idx >= 0:
        prefix = d_norm[:idx]
        start = len(prefix.split()) if prefix.strip() else 0
        return {'start_token': start, 'end_token': start + len(p_tokens), 'alignment_score': 1.0, 'alignment_method': 'exact_token_string'}

    p_set = set(p_tokens)
    w = len(p_tokens)
    step = max(1, min(10, w // 4 if w >= 4 else 1))
    best = None
    best_score = 0.0
    for start in range(0, max(1, len(d_tokens) - w + 1), step):
        window = d_tokens[start:start + w]
        w_set = set(window)
        union = len(p_set | w_set)
        jacc = len(p_set & w_set) / union if union else 0.0
        seq = SequenceMatcher(None, p_tokens, window).ratio()
        score = 0.7 * jacc + 0.3 * seq
        if score > best_score:
            best_score = score
            best = (start, start + w)
    if best and best_score >= min_score:
        return {'start_token': best[0], 'end_token': best[1], 'alignment_score': float(best_score), 'alignment_method': 'fuzzy_token_window'}
    return None


def read_trec_qrels(path, id_col):
    df = pd.read_csv(path, sep=r'\s+', names=['qid', 'Q0', id_col, 'relevance'], dtype={'qid': str, id_col: str}, engine='python')
    df['relevance'] = df['relevance'].astype(int)
    return df[['qid', id_col, 'relevance']]


def safe_display(df, n=5):
    try:
        display(df.head(n))
    except Exception:
        print(df.head(n))

# %%
# =====================================================================
# 3) Télécharger / vérifier le dépôt FiRA
# =====================================================================
def ensure_fira_repo(repo_dir=FIRA_REPO_DIR, github_url=FIRA_GITHUB):
    """Validate a locally prepared FiRA checkout.

    The original notebook could clone/download FiRA automatically. For this
    portfolio export, external process execution was intentionally removed.
    Follow docs/REPRODUCIBILITY.md and place the FiRA repository locally.
    """
    repo_dir = Path(repo_dir)
    expected = repo_dir / "data" / "input" / "documents.tsv"
    if not expected.exists():
        raise FileNotFoundError(
            f"FiRA dataset not found at {expected}. Prepare it manually before running this experiment."
        )
    print("[OK] Local FiRA repository:", repo_dir)
    return repo_dir

FIRA_REPO_DIR = ensure_fira_repo()
FIRA_DATA_DIR = FIRA_REPO_DIR / 'data'
FIRA_INPUT_DIR = FIRA_DATA_DIR / 'input'

print('Fichiers FiRA:')
for p in sorted(FIRA_DATA_DIR.glob('*')):
    print(' -', p.name)
print('Fichiers input:')
for p in sorted(FIRA_INPUT_DIR.glob('*')):
    print(' -', p.name)

# %%
# =====================================================================
# 4) Lecture robuste des queries, documents/snippets et qrels FiRA
# =====================================================================
def read_queries_fira_or_irds(path):
    """Lit data/input/queries.tsv. Si le format local est mal parsé,
    on retombe sur les queries officielles TREC-DL 2019 via ir_datasets.
    """
    path = Path(path)
    queries = None
    try:
        df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, engine='python')
        df.columns = [str(c).strip() for c in df.columns]
        lower = {c.lower(): c for c in df.columns}
        qid_col = lower.get('query_id') or lower.get('qid') or lower.get('queryid')
        text_col = lower.get('query_text') or lower.get('query') or lower.get('text')
        if qid_col and text_col:
            queries = df[[qid_col, text_col]].rename(columns={qid_col: 'qid', text_col: 'query'})
    except Exception as e:
        print('[WARN] Lecture standard queries.tsv échouée:', repr(e))

    if queries is None or len(queries) < 10:
        rows = []
        try:
            for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
                line = line.strip()
                if not line or line.lower().startswith('query_id'):
                    continue
                parts = line.split('\t', 1) if '\t' in line else re.split(r'\s+', line, maxsplit=1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    rows.append({'qid': parts[0].strip(), 'query': parts[1].strip()})
            if len(rows) >= 10:
                queries = pd.DataFrame(rows)
        except Exception as e:
            print('[WARN] Fallback parsing queries.tsv échoué:', repr(e))

    if queries is None or len(queries) < 10:
        print('[FALLBACK] Utilisation des queries msmarco-document/trec-dl-2019 via ir_datasets')
        ds = ir_datasets.load('msmarco-document/trec-dl-2019')
        rows = [{'qid': str(q.query_id), 'query': str(q.text)} for q in ds.queries_iter()]
        queries = pd.DataFrame(rows)

    queries['qid'] = queries['qid'].astype(str)
    queries['query'] = queries['query'].astype(str)
    queries = queries.drop_duplicates('qid').reset_index(drop=True)
    queries['query_clean'] = queries['query'].apply(clean_query_for_terrier)
    queries = queries[queries['query_clean'].str.len() > 0].copy()
    return queries


def pick_text_column(df, exclude_cols=()):
    exclude_cols = set(exclude_cols)
    cols = [c for c in df.columns if c not in exclude_cols]
    preferred = [c for c in cols if any(k in c.lower() for k in ['text', 'body', 'content', 'snippet', 'passage'])]
    candidates = preferred if preferred else cols
    candidates = [c for c in candidates if df[c].dtype == object or str(df[c].dtype).startswith('string')]
    if not candidates:
        raise ValueError('Aucune colonne texte détectée dans documents.tsv')
    return max(candidates, key=lambda c: df[c].astype(str).str.len().mean())


def standardize_fira_documents(path):
    raw = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, engine='python', quoting=csv.QUOTE_NONE)
    raw.columns = [str(c).strip() for c in raw.columns]
    cols = list(raw.columns)
    lower = {c.lower(): c for c in cols}
    snippet_col = None
    for name in ['snippet_id', 'passage_id', 'docno', 'document_snippet_id', 'id', 'doc_id', 'document_id']:
        if name in lower:
            snippet_col = lower[name]
            break
    if snippet_col is None:
        best = None
        best_score = -1
        for c in cols:
            s = raw[c].astype(str)
            unique_ratio = s.nunique() / max(1, len(s))
            underscore_ratio = s.str.contains('_', regex=False).mean()
            short_ratio = (s.str.len() < 80).mean()
            score = unique_ratio + underscore_ratio + short_ratio
            if score > best_score:
                best_score = score
                best = c
        snippet_col = best

    text_col = pick_text_column(raw, exclude_cols=[snippet_col])
    df = raw[[snippet_col, text_col]].rename(columns={snippet_col: 'snippet_id', text_col: 'snippet_text'}).copy()
    df['snippet_id'] = df['snippet_id'].astype(str).str.strip()
    df['snippet_text'] = df['snippet_text'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    df = df[(df['snippet_id'].str.len() > 0) & (df['snippet_text'].str.len() > 0)].copy()

    split = df['snippet_id'].str.rsplit('_', n=1, expand=True)
    if split.shape[1] == 2 and split[1].str.fullmatch(r'\d+').mean() > 0.5:
        df['doc_id'] = split[0].astype(str)
        df['snippet_position'] = split[1].astype(int)
    else:
        df['doc_id'] = df['snippet_id'].astype(str)
        df['snippet_position'] = df.groupby('doc_id').cumcount()

    df['snippet_token_len'] = df['snippet_text'].apply(lambda x: len(tok(x)))
    df = df.sort_values(['doc_id', 'snippet_position', 'snippet_id']).reset_index(drop=True)
    return df


def reconstruct_docs_from_snippets(df_snippets):
    doc_rows = []
    span_rows = []
    for doc_id, g in tqdm(df_snippets.groupby('doc_id', sort=False), desc='Reconstruit documents'):
        g = g.sort_values(['snippet_position', 'snippet_id'])
        parts = []
        cursor = 0
        for r in g.itertuples(index=False):
            snippet_text = str(r.snippet_text)
            tokens = tok(snippet_text)
            start = cursor
            end = cursor + len(tokens)
            span_rows.append({
                'doc_id': str(doc_id),
                'passage_id': str(r.snippet_id),
                'snippet_id': str(r.snippet_id),
                'snippet_position': int(r.snippet_position),
                'start_token': int(start),
                'end_token': int(end),
                'alignment_score': 1.0,
                'alignment_method': 'fira_direct_snippet_offset',
                'passage_text': snippet_text,
            })
            parts.append(snippet_text)
            cursor = end
        doc_text = '\n\n'.join(parts).strip()
        doc_rows.append({'doc_id': str(doc_id), 'doc_text': doc_text, 'num_snippets': int(len(g)), 'num_tokens': int(len(tok(doc_text)))})
    return pd.DataFrame(doc_rows), pd.DataFrame(span_rows)


if all(p.exists() for p in [PATH_QUERIES, PATH_DOC_QRELS, PATH_PASS_QRELS, PATH_SNIPPETS, PATH_RECON_DOCS, PATH_ALL_SNIPPET_SPANS]):
    topics = pd.read_csv(PATH_QUERIES, sep='\t', dtype=str)
    df_doc_qrels = pd.read_csv(PATH_DOC_QRELS, sep='\t', dtype=str)
    df_pass_qrels = pd.read_csv(PATH_PASS_QRELS, sep='\t', dtype=str)
    df_snippets = pd.read_csv(PATH_SNIPPETS, sep='\t', dtype=str)
    df_docs = pd.read_csv(PATH_RECON_DOCS, sep='\t', dtype=str)
    df_all_snippet_spans = pd.read_csv(PATH_ALL_SNIPPET_SPANS, sep='\t', dtype=str)
    print('[CACHE] queries/qrels/snippets/documents reconstruits chargés')
else:
    topics_all = read_queries_fira_or_irds(FIRA_INPUT_DIR / 'queries.tsv')
    df_doc_qrels_all = read_trec_qrels(FIRA_DATA_DIR / 'fira-trec19-notrec-qrels-docs-max.txt', 'doc_id')
    df_pass_qrels_all = read_trec_qrels(FIRA_DATA_DIR / 'fira-trec19-qrels-snippets.txt', 'passage_id')
    df_pass_qrels_all['doc_id'] = df_pass_qrels_all['passage_id'].astype(str).str.rsplit('_', n=1).str[0]

    df_snippets = standardize_fira_documents(FIRA_INPUT_DIR / 'documents.tsv')
    df_docs, df_all_snippet_spans = reconstruct_docs_from_snippets(df_snippets)

    available_doc_ids = set(df_docs['doc_id'].astype(str))
    available_qids = set(df_doc_qrels_all[df_doc_qrels_all['doc_id'].isin(available_doc_ids)]['qid'].astype(str))
    available_qids &= set(topics_all['qid'].astype(str))
    selected_qids = sorted(available_qids)
    if N_FINAL_QUERIES is not None:
        rng = np.random.default_rng(SEED)
        selected_qids = sorted(rng.choice(selected_qids, size=min(N_FINAL_QUERIES, len(selected_qids)), replace=False).astype(str))

    topics = topics_all[topics_all['qid'].isin(selected_qids)].sort_values('qid').reset_index(drop=True)
    df_doc_qrels = df_doc_qrels_all[df_doc_qrels_all['qid'].isin(selected_qids) & df_doc_qrels_all['doc_id'].isin(available_doc_ids)].copy()
    df_pass_qrels = df_pass_qrels_all[df_pass_qrels_all['qid'].isin(selected_qids) & df_pass_qrels_all['doc_id'].isin(available_doc_ids)].copy()

    save_tsv(topics, PATH_QUERIES)
    save_tsv(df_doc_qrels, PATH_DOC_QRELS)
    save_tsv(df_pass_qrels, PATH_PASS_QRELS)
    save_tsv(df_snippets, PATH_SNIPPETS)
    save_tsv(df_docs, PATH_RECON_DOCS)
    save_tsv(df_all_snippet_spans, PATH_ALL_SNIPPET_SPANS)

for df in [df_doc_qrels, df_pass_qrels]:
    df['qid'] = df['qid'].astype(str)
    df['relevance'] = df['relevance'].astype(int)
if 'doc_id' in df_doc_qrels.columns:
    df_doc_qrels['doc_id'] = df_doc_qrels['doc_id'].astype(str)
if 'passage_id' in df_pass_qrels.columns:
    df_pass_qrels['passage_id'] = df_pass_qrels['passage_id'].astype(str)
if 'doc_id' in df_pass_qrels.columns:
    df_pass_qrels['doc_id'] = df_pass_qrels['doc_id'].astype(str)
for c in ['snippet_position', 'snippet_token_len']:
    if c in df_snippets.columns:
        df_snippets[c] = df_snippets[c].astype(int)
for c in ['num_snippets', 'num_tokens']:
    if c in df_docs.columns:
        df_docs[c] = df_docs[c].astype(int)
for c in ['snippet_position', 'start_token', 'end_token']:
    if c in df_all_snippet_spans.columns:
        df_all_snippet_spans[c] = df_all_snippet_spans[c].astype(int)
if 'alignment_score' in df_all_snippet_spans.columns:
    df_all_snippet_spans['alignment_score'] = df_all_snippet_spans['alignment_score'].astype(float)

topics['qid'] = topics['qid'].astype(str)
topics['query_clean'] = topics['query_clean'].astype(str)
df_docs['doc_id'] = df_docs['doc_id'].astype(str)
df_docs['doc_text'] = df_docs['doc_text'].astype(str)

print('Requêtes:', topics.qid.nunique())
print('Documents reconstruits FiRA:', df_docs.doc_id.nunique(), 'lignes:', len(df_docs))
print('Snippets FiRA:', len(df_snippets))
print('Doc qrels:', len(df_doc_qrels), 'Snippet qrels:', len(df_pass_qrels))
safe_display(topics)
safe_display(df_docs)
safe_display(df_pass_qrels)

# %%
# =====================================================================
# 5) Index document FiRA léger sauvegardé dans Drive
# =====================================================================
def is_valid_terrier_index(index_dir):
    """Validation robuste, compatible avec plusieurs versions de PyTerrier/Terrier.
    Évite l'appel fragile getIndexStructureNames() qui peut ne pas exister.
    """
    index_dir = Path(index_dir)
    props = index_dir / 'data.properties'
    if not props.exists():
        print('[CHECK] data.properties absent')
        return False
    try:
        idx_ref = pt.IndexRef.of(str(props))
        idx = pt.IndexFactory.of(idx_ref)
        stats = idx.getCollectionStatistics()
        n_docs = int(stats.getNumberOfDocuments())
        n_terms = int(stats.getNumberOfUniqueTerms())
        n_tokens = int(stats.getNumberOfTokens())
        print(f'[CHECK] docs={n_docs}, terms={n_terms}, tokens={n_tokens}')
        return n_docs > 0 and n_terms > 0 and n_tokens > 0
    except Exception as e:
        print('[CHECK] Index invalide:', repr(e))
        return False

if is_valid_terrier_index(DOC_INDEX_DIR):
    print('[CACHE] Index FiRA document trouvé et valide')
    doc_index_ref = pt.IndexRef.of(str(DOC_INDEX_DIR / 'data.properties'))
else:
    print('[BUILD] Construction index FiRA document fields=True blocks=True...')
    if DOC_INDEX_DIR.exists():
        shutil.rmtree(DOC_INDEX_DIR)

    def docs_iter():
        for r in tqdm(df_docs.itertuples(index=False), total=len(df_docs), desc='Index FiRA docs'):
            yield {'docno': str(r.doc_id), 'text': str(r.doc_text)}

    indexer = pt.IterDictIndexer(str(DOC_INDEX_DIR), meta={'docno': 128}, fields=True, blocks=True, overwrite=True)
    doc_index_ref = indexer.index(docs_iter())
    print('[SAVE] Index créé:', DOC_INDEX_DIR)
    assert is_valid_terrier_index(DOC_INDEX_DIR), 'Index FiRA invalide après construction.'

print(pt.IndexFactory.of(doc_index_ref).getCollectionStatistics())

# %%
# =====================================================================
# 6) SDM top-50 documents sur le petit corpus FiRA
# =====================================================================
if PATH_TOPDOCS.exists():
    df_topdocs = pd.read_csv(PATH_TOPDOCS, sep='\t', dtype=str)
    df_topdocs['rank'] = df_topdocs['rank'].astype(int)
    df_topdocs['score'] = df_topdocs['score'].astype(float)
    print('[CACHE] topdocs chargés')
else:
    topics_pt = topics[['qid', 'query_clean']].rename(columns={'query_clean': 'query'}).copy()
    topics_pt['qid'] = topics_pt['qid'].astype(str)
    topics_pt['query'] = topics_pt['query'].astype(str)

    try:
        sdm_doc_pipe = pt.rewrite.SDM() >> pt.terrier.Retriever(doc_index_ref, wmodel='DirichletLM', num_results=TOPK_DOCS)
        df_topdocs = sdm_doc_pipe.transform(topics_pt).rename(columns={'docno': 'doc_id'}).copy()
    except Exception as e:
        print('[WARN] SDM a échoué, fallback Retriever DirichletLM simple:', repr(e))
        doc_pipe = pt.terrier.Retriever(doc_index_ref, wmodel='DirichletLM', num_results=TOPK_DOCS)
        df_topdocs = doc_pipe.transform(topics_pt).rename(columns={'docno': 'doc_id'}).copy()

    df_topdocs['qid'] = df_topdocs['qid'].astype(str)
    df_topdocs['doc_id'] = df_topdocs['doc_id'].astype(str)
    df_topdocs = df_topdocs[['qid', 'doc_id', 'rank', 'score']].sort_values(['qid', 'rank']).reset_index(drop=True)
    save_tsv(df_topdocs, PATH_TOPDOCS)

print('Topdocs:', df_topdocs.shape, 'requêtes:', df_topdocs.qid.nunique())
safe_display(df_topdocs)

# %%
# =====================================================================
# 7) Documents candidats = tous les top-50 documents récupérés
# =====================================================================
# Correction principale : on ne filtre plus sur les documents pertinents.
# Chaque requête garde ses top-k documents récupérés. Les documents sans qrel positif
# gardent doc_relevance = 0. Ils produiront donc des fenêtres négatives si aucun span
# FiRA ne les chevauche.

if PATH_CANDIDATE_TOPDOCS.exists():
    df_candidate_topdocs = pd.read_csv(PATH_CANDIDATE_TOPDOCS, sep='\t', dtype=str)
    for c in ['doc_rank', 'doc_relevance', 'retrieved_in_topk']:
        if c in df_candidate_topdocs.columns:
            df_candidate_topdocs[c] = df_candidate_topdocs[c].astype(int)
    if 'doc_score' in df_candidate_topdocs.columns:
        df_candidate_topdocs['doc_score'] = df_candidate_topdocs['doc_score'].astype(float)
    print('[CACHE] candidate_topdocs chargés')
else:
    available_doc_ids = set(df_docs['doc_id'].astype(str))
    doc_rel = (
        df_doc_qrels[df_doc_qrels.doc_id.astype(str).isin(available_doc_ids)]
        .rename(columns={'relevance': 'doc_relevance'})
        .copy()
    )
    doc_rel['qid'] = doc_rel['qid'].astype(str)
    doc_rel['doc_id'] = doc_rel['doc_id'].astype(str)
    doc_rel['doc_relevance'] = doc_rel['doc_relevance'].astype(int)

    df_candidate_topdocs = df_topdocs.copy()
    df_candidate_topdocs['qid'] = df_candidate_topdocs['qid'].astype(str)
    df_candidate_topdocs['doc_id'] = df_candidate_topdocs['doc_id'].astype(str)

    df_candidate_topdocs = df_candidate_topdocs.merge(
        doc_rel[['qid', 'doc_id', 'doc_relevance']].drop_duplicates(['qid', 'doc_id']),
        on=['qid', 'doc_id'],
        how='left'
    )
    df_candidate_topdocs['doc_relevance'] = df_candidate_topdocs['doc_relevance'].fillna(0).astype(int)
    df_candidate_topdocs['retrieved_in_topk'] = 1
    df_candidate_topdocs = df_candidate_topdocs.rename(columns={'rank': 'doc_rank', 'score': 'doc_score'})
    df_candidate_topdocs = df_candidate_topdocs[['qid', 'doc_id', 'doc_rank', 'doc_score', 'doc_relevance', 'retrieved_in_topk']]
    df_candidate_topdocs = df_candidate_topdocs.sort_values(['qid', 'doc_rank', 'doc_id']).reset_index(drop=True)
    save_tsv(df_candidate_topdocs, PATH_CANDIDATE_TOPDOCS)

# Alias volontaire : certains notebooks suivants utilisent encore ce nom.
df_relevant_topdocs = df_candidate_topdocs

print('Candidate topdocs:', df_candidate_topdocs.shape, 'requêtes:', df_candidate_topdocs.qid.nunique())
print('Docs avec doc_relevance > 0:', int((df_candidate_topdocs.doc_relevance.astype(int) > 0).sum()), '/', len(df_candidate_topdocs))
print('Docs avec doc_relevance = 0:', int((df_candidate_topdocs.doc_relevance.astype(int) == 0).sum()), '/', len(df_candidate_topdocs))
safe_display(df_candidate_topdocs)

# %%
# =====================================================================
# 8) Spans annotés FiRA qui se trouvent dans les documents candidats top-50
# =====================================================================
# df_spans ne contient que les passages/snippets annotés FiRA qui appartiennent
# à un document candidat top-50. Les documents sans span restent dans
# df_candidate_topdocs ; ils auront simplement des fenêtres fira_label = 0.

if PATH_ALIGNED_SPANS.exists():
    df_spans = pd.read_csv(PATH_ALIGNED_SPANS, sep='\t', dtype=str)
    for c in ['passage_relevance', 'doc_relevance', 'start_token', 'end_token', 'doc_rank']:
        if c in df_spans.columns:
            df_spans[c] = df_spans[c].astype(int)
    if 'doc_score' in df_spans.columns:
        df_spans['doc_score'] = df_spans['doc_score'].astype(float)
    if 'alignment_score' in df_spans.columns:
        df_spans['alignment_score'] = df_spans['alignment_score'].astype(float)
    print('[CACHE] spans top-50 chargés')
else:
    pass_rel = df_pass_qrels[df_pass_qrels.relevance > 0].rename(columns={'relevance': 'passage_relevance'}).copy()
    pass_rel['qid'] = pass_rel['qid'].astype(str)
    pass_rel['passage_id'] = pass_rel['passage_id'].astype(str)
    pass_rel['doc_id'] = pass_rel['doc_id'].astype(str)
    pass_rel['passage_relevance'] = pass_rel['passage_relevance'].astype(int)

    spans_base = df_all_snippet_spans.copy()
    spans_base['passage_id'] = spans_base['passage_id'].astype(str)
    spans_base['doc_id'] = spans_base['doc_id'].astype(str)

    span_rel = spans_base.merge(
        pass_rel[['qid', 'doc_id', 'passage_id', 'passage_relevance']],
        on=['doc_id', 'passage_id'],
        how='inner'
    )

    df_spans = df_candidate_topdocs.merge(span_rel, on=['qid', 'doc_id'], how='inner')
    df_spans['passage_label_name'] = df_spans['passage_relevance'].apply(relevance_name)
    df_spans = df_spans[[
        'qid', 'doc_id', 'passage_id', 'doc_rank', 'doc_score', 'doc_relevance',
        'passage_relevance', 'passage_label_name', 'start_token', 'end_token',
        'alignment_score', 'alignment_method', 'passage_text'
    ]].sort_values(['qid', 'doc_rank', 'doc_id', 'start_token']).reset_index(drop=True)
    save_tsv(df_spans, PATH_ALIGNED_SPANS)

print('Spans annotés dans top-50:', df_spans.shape, 'requêtes:', df_spans.qid.nunique() if len(df_spans) else 0)
safe_display(df_spans)

# %%
# =====================================================================
# 9) Résumé intermédiaire + métadonnées
# =====================================================================
summary = pd.DataFrame({'qid': sorted(topics.qid.astype(str).unique())})
if len(df_topdocs):
    summary = summary.merge(df_topdocs.groupby('qid').size().reset_index(name='num_topdocs'), on='qid', how='left')
if len(df_candidate_topdocs):
    summary = summary.merge(df_candidate_topdocs.groupby('qid').size().reset_index(name='num_candidate_topdocs'), on='qid', how='left')
    summary = summary.merge(df_candidate_topdocs[df_candidate_topdocs.doc_relevance.astype(int) > 0].groupby('qid').size().reset_index(name='num_positive_doc_qrels_in_topdocs'), on='qid', how='left')
    summary = summary.merge(df_candidate_topdocs[df_candidate_topdocs.doc_relevance.astype(int) == 0].groupby('qid').size().reset_index(name='num_zero_doc_qrels_in_topdocs'), on='qid', how='left')
if len(df_spans):
    summary = summary.merge(df_spans.groupby('qid').size().reset_index(name='num_aligned_passage_spans_in_topdocs'), on='qid', how='left')
    summary = summary.merge(df_spans[df_spans.passage_relevance.astype(int) >= MIN_STRONG_RELEVANCE].groupby('qid').size().reset_index(name='num_strong_spans_in_topdocs'), on='qid', how='left')
summary = summary.fillna(0)
for c in summary.columns:
    if c != 'qid':
        summary[c] = summary[c].astype(int)

save_tsv(summary, PATH_SUMMARY)
metadata = {
    'experiment_name': EXPERIMENT_NAME,
    'approach': APPROACH_NAME,
    'seed': SEED,
    'source_dataset': 'FiRA fine-grained annotations for TREC-DL 2019',
    'fira_repo': FIRA_GITHUB,
    'topk_docs_sdm': TOPK_DOCS,
    'include_all_available_relevant_docs': INCLUDE_ALL_AVAILABLE_RELEVANT_DOCS,
    'candidate_document_policy': 'all retrieved top-k documents are kept; relevant documents outside top-k are not added',
    'num_final_queries': int(topics.qid.nunique()),
    'num_reconstructed_docs': int(df_docs.doc_id.nunique()),
    'num_snippets': int(len(df_snippets)),
    'num_doc_qrels': int(len(df_doc_qrels)),
    'num_passage_qrels': int(len(df_pass_qrels)),
    'num_topdocs': int(len(df_topdocs)),
    'num_candidate_topdocs': int(len(df_candidate_topdocs)),
    'num_positive_doc_qrels_in_candidate_topdocs': int((df_candidate_topdocs.doc_relevance.astype(int) > 0).sum()),
    'num_zero_doc_qrels_in_candidate_topdocs': int((df_candidate_topdocs.doc_relevance.astype(int) == 0).sum()),
    'num_aligned_spans_in_topdocs': int(len(df_spans)),
    'min_strong_relevance': MIN_STRONG_RELEVANCE,
    'label_mapping': {'0': 'non_relevant_or_no_overlap', '1': 'fair_or_topical', '2': 'good_answer', '3': 'perfect_answer'},
    'note': 'Document-first sur FiRA. Les documents longs sont reconstruits en concaténant les snippets FiRA par doc_id. Les spans sont les snippets annotés FiRA avec offsets directs. Les fenêtres sont produites pour tous les documents candidats top-k, pas seulement pour les documents pertinents.',
}
save_json(metadata, PATH_METADATA)

display(summary.describe())
safe_display(summary)
print('Artefacts:', ARTIFACT_DIR)

# %%
# =====================================================================
# 10) Fonctions fenêtres + labels FiRA directs
# =====================================================================
def make_windows(tokens, window_size=50, overlap=25):
    step = window_size - overlap
    if step <= 0:
        raise ValueError('window_size doit être strictement supérieur à overlap')
    out = []
    wid = 0
    for start in range(0, len(tokens), step):
        end = min(start + window_size, len(tokens))
        if end <= start:
            break
        out.append({
            'window_id': wid,
            'window_start_token': start,
            'window_end_token': end,
            'window_text': ' '.join(tokens[start:end])
        })
        wid += 1
        if end == len(tokens):
            break
    return out


def label_window_against_spans(w_start, w_end, spans, min_overlap_ratio=MIN_WINDOW_OVERLAP_RATIO):
    """
    Label FiRA direct pour une fenêtre.

    Principe :
    - On cherche les passages/snippets FiRA annotés qui chevauchent la fenêtre.
    - Si aucune annotation FiRA ne chevauche la fenêtre : fira_label = 0.
    - Si une ou plusieurs annotations chevauchent la fenêtre : on prend le label FiRA maximal.
    - En cas d'égalité, on garde le span avec le plus grand nombre de tokens chevauchés.

    Labels FiRA utilisés ici :
    0 = aucun chevauchement avec une annotation pertinente FiRA
    1 = fair / topical
    2 = good
    3 = perfect
    """
    best = {
        'overlap_tokens': 0,
        'recall_span': 0.0,
        'precision_window': 0.0,
        'fira_label': 0,
        'matched_passage_id': None,
        'matched_passage_relevance': 0,
    }

    for sp in spans:
        s = int(sp['start_token'])
        e = int(sp['end_token'])

        ov = max(0, min(w_end, e) - max(w_start, s))
        if ov <= 0:
            continue

        span_len = max(1, e - s)
        window_len = max(1, w_end - w_start)
        recall = ov / span_len
        precision = ov / window_len

        if min_overlap_ratio > 0 and recall < min_overlap_ratio and precision < min_overlap_ratio:
            continue

        label = int(sp['passage_relevance'])

        if (label, ov) > (best['fira_label'], best['overlap_tokens']):
            best = {
                'overlap_tokens': int(ov),
                'recall_span': float(recall),
                'precision_window': float(precision),
                'fira_label': int(label),
                'matched_passage_id': sp['passage_id'],
                'matched_passage_relevance': int(label),
            }

    return best

# %%
# =====================================================================
# 11) Créer/recharger fenêtres 50/25 + labels FiRA directs
# =====================================================================

def _load_cached_windows_and_fira_labels():
    """Charge le cache seulement s'il correspond au format corrigé top-50-only."""
    if not (PATH_WINDOWS.exists() and PATH_WINDOW_LABELS.exists() and PATH_MAIN_DATASET.exists()):
        return None, None, None

    df_w = pd.read_csv(PATH_WINDOWS, sep='\t', dtype=str)
    df_l = pd.read_csv(PATH_WINDOW_LABELS, sep='\t', dtype=str)
    df_main = pd.read_csv(PATH_MAIN_DATASET, sep='\t', dtype=str)

    required_label_cols = {'fira_label', 'is_relevant'}
    required_main_cols = {'qid', 'doc_id', 'doc_rank', 'window_id', 'window_start', 'window_end', 'window_text', 'fira_label', 'is_relevant'}
    if not required_label_cols.issubset(df_l.columns) or not required_main_cols.issubset(df_main.columns):
        print('[CACHE] Cache incomplet ou ancien format. Reconstruction...')
        return None, None, None

    for c in ['window_id', 'window_start_token', 'window_end_token', 'doc_rank', 'doc_relevance', 'retrieved_in_topk']:
        if c in df_w.columns:
            df_w[c] = df_w[c].astype(int)

    for c in ['window_id', 'overlap_tokens', 'matched_passage_relevance', 'fira_label', 'is_relevant', 'is_strong_relevant']:
        if c in df_l.columns:
            df_l[c] = df_l[c].astype(int)

    for c in ['recall_span', 'precision_window']:
        if c in df_l.columns:
            df_l[c] = df_l[c].astype(float)

    for c in ['doc_rank', 'window_id', 'window_start', 'window_end', 'fira_label', 'is_relevant']:
        if c in df_main.columns:
            df_main[c] = df_main[c].astype(int)

    print('[CACHE] windows/labels/main dataset FiRA top-50 chargés')
    return df_w, df_l, df_main


df_windows, df_window_labels, df_main_dataset = _load_cached_windows_and_fira_labels()

if df_windows is None or df_window_labels is None or df_main_dataset is None:
    docs_map = dict(zip(df_docs['doc_id'].astype(str), df_docs['doc_text'].astype(str)))
    spans_by = defaultdict(list)
    for r in df_spans.to_dict(orient='records'):
        spans_by[(str(r['qid']), str(r['doc_id']))].append(r)

    win_rows = []
    lab_rows = []
    for r in tqdm(df_candidate_topdocs.itertuples(index=False), total=len(df_candidate_topdocs), desc='Windows top-50'):
        qid = str(r.qid)
        doc_id = str(r.doc_id)
        doc_text = docs_map.get(doc_id, '')
        tokens = tok(doc_text)
        if not tokens:
            continue

        doc_spans = spans_by.get((qid, doc_id), [])

        for w in make_windows(tokens, WINDOW_SIZE, WINDOW_OVERLAP):
            win_rows.append({
                'qid': qid,
                'doc_id': doc_id,
                'window_id': int(w['window_id']),
                'window_start_token': int(w['window_start_token']),
                'window_end_token': int(w['window_end_token']),
                'window_text': w['window_text'],
                'doc_rank': int(r.doc_rank),
                'doc_score': float(r.doc_score),
                'doc_relevance': int(r.doc_relevance),
                'retrieved_in_topk': int(r.retrieved_in_topk),
            })

            lab = label_window_against_spans(
                w['window_start_token'],
                w['window_end_token'],
                doc_spans
            )
            is_rel = int(lab['fira_label'] >= MIN_STRONG_RELEVANCE)
            lab_rows.append({
                'qid': qid,
                'doc_id': doc_id,
                'window_id': int(w['window_id']),
                **lab,
                'label_name': relevance_name(lab['fira_label']),
                'is_relevant': is_rel,
                'is_strong_relevant': is_rel,  # alias pour compatibilité
            })

    df_windows = pd.DataFrame(win_rows)
    df_window_labels = pd.DataFrame(lab_rows)

    df_main_dataset = df_windows.merge(
        df_window_labels[[
            'qid', 'doc_id', 'window_id', 'fira_label', 'is_relevant',
            'label_name', 'overlap_tokens', 'recall_span', 'precision_window',
            'matched_passage_id', 'matched_passage_relevance'
        ]],
        on=['qid', 'doc_id', 'window_id'],
        how='left'
    )
    df_main_dataset['fira_label'] = df_main_dataset['fira_label'].fillna(0).astype(int)
    df_main_dataset['is_relevant'] = df_main_dataset['is_relevant'].fillna(0).astype(int)

    df_main_dataset = df_main_dataset.rename(columns={
        'window_start_token': 'window_start',
        'window_end_token': 'window_end'
    })

    # Colonnes principales demandées en premier, colonnes de diagnostic ensuite.
    main_cols = [
        'qid', 'doc_id', 'doc_rank', 'window_id', 'window_start', 'window_end',
        'window_text', 'fira_label', 'is_relevant'
    ]
    diagnostic_cols = [
        'doc_score', 'doc_relevance', 'retrieved_in_topk', 'label_name',
        'overlap_tokens', 'recall_span', 'precision_window',
        'matched_passage_id', 'matched_passage_relevance'
    ]
    df_main_dataset = df_main_dataset[main_cols + [c for c in diagnostic_cols if c in df_main_dataset.columns]]

    save_tsv(df_windows, PATH_WINDOWS)
    save_tsv(df_window_labels, PATH_WINDOW_LABELS)
    save_tsv(df_main_dataset, PATH_MAIN_DATASET)
    print('[SAVE] windows/labels/main dataset FiRA reconstruits')

print('Fenêtres:', df_windows.shape, 'labels:', df_window_labels.shape, 'main:', df_main_dataset.shape)

if len(df_window_labels):
    print('Distribution fira_label :')
    display(df_window_labels.fira_label.astype(int).value_counts().sort_index())
    print('Distribution is_relevant :')
    display(df_window_labels.is_relevant.astype(int).value_counts().sort_index())
else:
    print('[WARN] Aucun label de fenêtre produit.')

# Vérification importante : les fenêtres doivent venir de documents top-50 uniquement.
assert INCLUDE_ALL_AVAILABLE_RELEVANT_DOCS is False
assert set(df_main_dataset['retrieved_in_topk'].astype(int).unique()).issubset({1}), 'Toutes les fenêtres doivent venir du top-k.'
assert {'qid', 'doc_id', 'doc_rank', 'window_id', 'window_start', 'window_end', 'window_text', 'fira_label', 'is_relevant'}.issubset(df_main_dataset.columns)

safe_display(df_main_dataset)
safe_display(df_windows)
safe_display(df_window_labels)

metadata = json.loads(PATH_METADATA.read_text()) if PATH_METADATA.exists() else {}
metadata.update({
    'window_size': WINDOW_SIZE,
    'window_overlap': WINDOW_OVERLAP,
    'window_label_source': 'fira_direct_from_snippet_qrels_in_top50_candidates',
    'window_min_overlap_ratio': MIN_WINDOW_OVERLAP_RATIO,
    'main_dataset': str(PATH_MAIN_DATASET),
    'main_dataset_columns': list(df_main_dataset.columns),
    'num_windows': int(len(df_windows)),
    'num_strong_windows': int((df_window_labels.fira_label.astype(int) >= MIN_STRONG_RELEVANCE).sum()) if len(df_window_labels) else 0,
    'num_non_relevant_windows': int((df_window_labels.fira_label.astype(int) == 0).sum()) if len(df_window_labels) else 0,
})
save_json(metadata, PATH_METADATA)
print('Dataset principal:', PATH_MAIN_DATASET)
print('Artefacts finaux:', ARTIFACT_DIR)

# %%
# =====================================================================
# 12) Vérification finale des fichiers produits
# =====================================================================
required_outputs = {
    'queries': PATH_QUERIES,
    'doc_qrels': PATH_DOC_QRELS,
    'snippet_qrels': PATH_PASS_QRELS,
    'snippets_standardized': PATH_SNIPPETS,
    'reconstructed_docs': PATH_RECON_DOCS,
    'all_snippet_spans': PATH_ALL_SNIPPET_SPANS,
    'topdocs': PATH_TOPDOCS,
    'candidate_topdocs': PATH_CANDIDATE_TOPDOCS,
    'aligned_spans_in_top50': PATH_ALIGNED_SPANS,
    'windows': PATH_WINDOWS,
    'window_labels': PATH_WINDOW_LABELS,
    'main_dataset': PATH_MAIN_DATASET,
    'summary': PATH_SUMMARY,
    'metadata': PATH_METADATA,
}

rows = []
for name, path in required_outputs.items():
    path = Path(path)
    rows.append({
        'artifact': name,
        'exists': path.exists(),
        'size_MB': round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else 0,
        'path': str(path)
    })
check_df = pd.DataFrame(rows)
display(check_df)

assert check_df['exists'].all(), 'Certains artefacts sont manquants.'
assert 'fira_label' in pd.read_csv(PATH_WINDOW_LABELS, sep='\t', nrows=1).columns, 'window_labels doit contenir fira_label.'
assert 'is_relevant' in pd.read_csv(PATH_WINDOW_LABELS, sep='\t', nrows=1).columns, 'window_labels doit contenir is_relevant.'

main_head = pd.read_csv(PATH_MAIN_DATASET, sep='\t', nrows=5)
expected_main_cols = ['qid', 'doc_id', 'doc_rank', 'window_id', 'window_start', 'window_end', 'window_text', 'fira_label', 'is_relevant']
assert expected_main_cols == list(main_head.columns[:len(expected_main_cols)]), 'Les colonnes principales du dataset ne sont pas dans le bon ordre.'

assert is_valid_terrier_index(DOC_INDEX_DIR), 'Index PyTerrier FiRA invalide.'
print('[OK] Notebook FiRA top-50-only terminé correctement.')
print('[OK] Dataset principal :', PATH_MAIN_DATASET)
