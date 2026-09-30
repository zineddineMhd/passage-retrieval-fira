# Auxiliary IR experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # FiRA / TREC-DL 2019 document-first sans fenêtres — labels FiRA directs
#
# Cette version reprend la structure du notebook **TREC-DL document-first sans fenêtres**, mais remplace le gros corpus MS MARCO par **FiRA**, une version plus petite avec annotations fines de TREC-DL 2019.
#
# Objectif : garder le même pipeline général, mais sans créer de fenêtres 50/25 :
#
# 1. charger queries + qrels ;
# 2. reconstruire un petit corpus document-first à partir des snippets FiRA ;
# 3. construire un index PyTerrier léger ;
# 4. récupérer les top documents ;
# 5. identifier les documents pertinents ;
# 6. utiliser directement les snippets/passages FiRA comme unités finales ;
# 7. sauvegarder `passage_units.tsv`, `summary_by_qid.tsv` et `dataset_metadata.json`.
#
# Différence importante : FiRA ne contient pas tout le corpus MS MARCO document. Ici, on indexe seulement les documents reconstruits à partir des snippets annotés FiRA. Ce notebook est donc adapté à une expérience **passage/snippet-level relevance sans fenêtres**, pas à une évaluation retrieval full-corpus sur 3,2M documents.
#
# Correction appliquée : les unités finales utilisent maintenant `fira_label` directement à partir des qrels/snippets FiRA. On ne crée plus de `silver_label`. Si un ancien cache contient seulement `silver_label` ou ne contient pas `fira_label`, il est ignoré/reconstruit automatiquement.

# %% [markdown]
# # Version FiRA — sans fenêtres

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
MIN_STRONG_RELEVANCE = 2
# Si True : on garde tous les documents FiRA pertinents disponibles, même s'ils ne sont
# pas dans le top-k BM25/SDM. C'est utile pour maximiser les données de training windows.
INCLUDE_ALL_AVAILABLE_RELEVANT_DOCS = True

DRIVE_ROOT = Path('/content/drive/MyDrive/IR_project_backup')
EXPERIMENT_NAME = 'fira_trec19_article_like_no_windows'
APPROACH_NAME = 'document_first_no_windows_fira_small'
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
PATH_TOPDOCS = ARTIFACT_DIR / 'topdocs_top50.tsv'
PATH_RELEVANT_TOPDOCS = ARTIFACT_DIR / 'relevant_topdocs.tsv'
PATH_ALIGNED_SPANS = ARTIFACT_DIR / 'fira_aligned_snippet_spans.tsv'
PATH_SUMMARY = ARTIFACT_DIR / 'summary_by_qid.tsv'
PATH_METADATA = ARTIFACT_DIR / 'dataset_metadata.json'
PATH_PASSAGE_UNITS = ARTIFACT_DIR / 'passage_units.tsv'

print('Dossier Drive:', ARTIFACT_DIR)
print('SEED =', SEED)

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
# 7) Documents pertinents disponibles dans FiRA
# =====================================================================
if PATH_RELEVANT_TOPDOCS.exists():
    df_relevant_topdocs = pd.read_csv(PATH_RELEVANT_TOPDOCS, sep='\t', dtype=str)
    df_relevant_topdocs['doc_relevance'] = df_relevant_topdocs['doc_relevance'].astype(int)
    if 'rank' in df_relevant_topdocs.columns:
        df_relevant_topdocs['rank'] = df_relevant_topdocs['rank'].astype(int)
    if 'score' in df_relevant_topdocs.columns:
        df_relevant_topdocs['score'] = df_relevant_topdocs['score'].astype(float)
    print('[CACHE] relevant_topdocs chargés')
else:
    available_doc_ids = set(df_docs['doc_id'].astype(str))
    doc_rel = df_doc_qrels[(df_doc_qrels.relevance > 0) & df_doc_qrels.doc_id.astype(str).isin(available_doc_ids)].rename(columns={'relevance': 'doc_relevance'}).copy()

    if INCLUDE_ALL_AVAILABLE_RELEVANT_DOCS:
        df_relevant_topdocs = doc_rel[['qid', 'doc_id', 'doc_relevance']].drop_duplicates(['qid', 'doc_id']).copy()
        df_relevant_topdocs = df_relevant_topdocs.merge(df_topdocs, on=['qid', 'doc_id'], how='left')
        df_relevant_topdocs['retrieved_in_topk'] = df_relevant_topdocs['rank'].notna().astype(int)
        df_relevant_topdocs = df_relevant_topdocs.sort_values(['qid', 'retrieved_in_topk', 'rank', 'doc_id'], ascending=[True, False, True, True])
        missing = df_relevant_topdocs['rank'].isna()
        df_relevant_topdocs.loc[missing, 'rank'] = TOPK_DOCS + 1 + np.arange(missing.sum())
        df_relevant_topdocs.loc[missing, 'score'] = 0.0
        df_relevant_topdocs['rank'] = df_relevant_topdocs['rank'].astype(int)
        df_relevant_topdocs['score'] = df_relevant_topdocs['score'].astype(float)
    else:
        df_relevant_topdocs = df_topdocs.merge(doc_rel[['qid', 'doc_id', 'doc_relevance']], on=['qid', 'doc_id'], how='inner')
        df_relevant_topdocs['retrieved_in_topk'] = 1

    df_relevant_topdocs = df_relevant_topdocs[['qid', 'doc_id', 'rank', 'score', 'doc_relevance', 'retrieved_in_topk']]
    df_relevant_topdocs = df_relevant_topdocs.sort_values(['qid', 'rank', 'doc_id']).reset_index(drop=True)
    save_tsv(df_relevant_topdocs, PATH_RELEVANT_TOPDOCS)

print('Relevant topdocs:', df_relevant_topdocs.shape, 'requêtes:', df_relevant_topdocs.qid.nunique())
print('Docs retrouvés top-k:', int(df_relevant_topdocs.retrieved_in_topk.astype(int).sum()), '/', len(df_relevant_topdocs))
safe_display(df_relevant_topdocs)

# %%
# =====================================================================
# 8) Spans pertinents : FiRA donne déjà les snippets, donc label FiRA direct
# =====================================================================
def _load_cached_spans_with_fira_label():
    """Charge les spans si possible et garantit la présence de fira_label.
    Si l'ancien cache ne contient pas fira_label mais contient passage_relevance,
    on crée fira_label comme alias direct de passage_relevance.
    """
    if not PATH_ALIGNED_SPANS.exists():
        return None

    df = pd.read_csv(PATH_ALIGNED_SPANS, sep='\t', dtype=str)

    # Compatibilité ancien cache : passage_relevance était déjà le label FiRA.
    if 'fira_label' not in df.columns:
        if 'passage_relevance' in df.columns:
            print('[CACHE] Ancien cache spans détecté sans fira_label. Ajout alias fira_label = passage_relevance...')
            df['fira_label'] = df['passage_relevance']
        else:
            print('[CACHE] Cache spans invalide : ni fira_label ni passage_relevance. Reconstruction...')
            return None

    # On garde passage_relevance comme alias pour compatibilité avec les notebooks modeling.
    if 'passage_relevance' not in df.columns:
        df['passage_relevance'] = df['fira_label']

    for c in ['fira_label', 'passage_relevance', 'doc_relevance', 'start_token', 'end_token', 'doc_rank']:
        if c in df.columns:
            df[c] = df[c].astype(int)
    if 'alignment_score' in df.columns:
        df['alignment_score'] = df['alignment_score'].astype(float)

    df['passage_label_name'] = df['fira_label'].apply(relevance_name)
    print('[CACHE] spans FiRA chargés')
    return df


df_spans = _load_cached_spans_with_fira_label()

if df_spans is None:
    pass_rel = df_pass_qrels[df_pass_qrels.relevance > 0].rename(columns={'relevance': 'fira_label'}).copy()
    pass_rel['passage_id'] = pass_rel['passage_id'].astype(str)
    pass_rel['doc_id'] = pass_rel['doc_id'].astype(str)

    spans_base = df_all_snippet_spans.copy()
    spans_base['passage_id'] = spans_base['passage_id'].astype(str)
    spans_base['doc_id'] = spans_base['doc_id'].astype(str)

    span_rel = spans_base.merge(
        pass_rel[['qid', 'doc_id', 'passage_id', 'fira_label']],
        on=['doc_id', 'passage_id'],
        how='inner'
    )
    df_spans = df_relevant_topdocs.merge(span_rel, on=['qid', 'doc_id'], how='inner')
    df_spans = df_spans.rename(columns={'rank': 'doc_rank', 'score': 'doc_score'})

    # Alias explicite : passage_relevance = fira_label, pour compatibilité.
    df_spans['fira_label'] = df_spans['fira_label'].astype(int)
    df_spans['passage_relevance'] = df_spans['fira_label']
    df_spans['passage_label_name'] = df_spans['fira_label'].apply(relevance_name)

    df_spans = df_spans[[
        'qid', 'doc_id', 'passage_id', 'doc_rank', 'doc_score',
        'doc_relevance', 'fira_label', 'passage_relevance', 'passage_label_name',
        'start_token', 'end_token', 'alignment_score', 'alignment_method', 'passage_text'
    ]].sort_values(['qid', 'doc_rank', 'doc_id', 'start_token']).reset_index(drop=True)

    save_tsv(df_spans, PATH_ALIGNED_SPANS)

print('Spans:', df_spans.shape, 'requêtes:', df_spans.qid.nunique() if len(df_spans) else 0)
if len(df_spans):
    print('Distribution fira_label :')
    display(df_spans.fira_label.value_counts().sort_index())
safe_display(df_spans)

# %%
# =====================================================================
# 9) Résumé + métadonnées
# =====================================================================
summary = pd.DataFrame({'qid': sorted(topics.qid.astype(str).unique())})
if len(df_topdocs):
    summary = summary.merge(df_topdocs.groupby('qid').size().reset_index(name='num_topdocs'), on='qid', how='left')
if len(df_relevant_topdocs):
    summary = summary.merge(df_relevant_topdocs.groupby('qid').size().reset_index(name='num_relevant_topdocs'), on='qid', how='left')
if len(df_spans):
    summary = summary.merge(df_spans.groupby('qid').size().reset_index(name='num_aligned_passage_spans'), on='qid', how='left')
    summary = summary.merge(df_spans[df_spans.fira_label.astype(int) >= MIN_STRONG_RELEVANCE].groupby('qid').size().reset_index(name='num_strong_spans'), on='qid', how='left')
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
    'num_final_queries': int(topics.qid.nunique()),
    'num_reconstructed_docs': int(df_docs.doc_id.nunique()),
    'num_snippets': int(len(df_snippets)),
    'num_doc_qrels': int(len(df_doc_qrels)),
    'num_passage_qrels': int(len(df_pass_qrels)),
    'num_topdocs': int(len(df_topdocs)),
    'num_relevant_topdocs': int(len(df_relevant_topdocs)),
    'num_aligned_spans': int(len(df_spans)),
    'min_strong_relevance': MIN_STRONG_RELEVANCE,
    'passage_label_source': 'fira_direct_from_snippet_qrels',
    'label_column': 'fira_label',
    'label_mapping': {
        '0': 'non_relevant',
        '1': 'fair_or_topic_relevant_does_not_answer_directly',
        '2': 'good_answer',
        '3': 'perfect_answer',
    },
    'note': 'Document-first sans fenêtres sur FiRA. Les documents longs sont reconstruits en concaténant les snippets FiRA par doc_id. Les unités finales sont les snippets/passages pertinents FiRA avec fira_label direct, sans découpage 50/25 et sans silver_label.',
}
save_json(metadata, PATH_METADATA)

display(summary.describe())
safe_display(summary)
print('Artefacts:', ARTIFACT_DIR)

# %%
# =====================================================================
# 10) Construire/recharger unités finales sans fenêtres : snippets/passages FiRA
# =====================================================================
def _load_cached_passage_units_with_fira_label():
    """Charge passage_units uniquement si les labels FiRA directs sont présents.
    Si un ancien cache contient seulement silver_label, on reconstruit.
    Si un ancien cache contient passage_relevance mais pas fira_label, on ajoute l'alias.
    """
    if not PATH_PASSAGE_UNITS.exists():
        return None

    df = pd.read_csv(PATH_PASSAGE_UNITS, sep='\t', dtype=str)

    if 'silver_label' in df.columns and 'fira_label' not in df.columns:
        print('[CACHE] Ancien passage_units avec silver_label détecté. Reconstruction avec fira_label direct...')
        return None

    if 'fira_label' not in df.columns:
        if 'passage_relevance' in df.columns:
            print('[CACHE] Ancien passage_units sans fira_label. Ajout alias fira_label = passage_relevance...')
            df['fira_label'] = df['passage_relevance']
        else:
            print('[CACHE] passage_units invalide : aucun label FiRA détecté. Reconstruction...')
            return None

    # Alias pour compatibilité avec les anciens notebooks modeling.
    if 'passage_relevance' not in df.columns:
        df['passage_relevance'] = df['fira_label']

    for c in ['doc_rank', 'doc_relevance', 'fira_label', 'passage_relevance', 'start_token', 'end_token', 'is_strong_relevant', 'retrieved_in_topk']:
        if c in df.columns:
            df[c] = df[c].astype(int)
    if 'doc_score' in df.columns:
        df['doc_score'] = df['doc_score'].astype(float)
    if 'alignment_score' in df.columns:
        df['alignment_score'] = df['alignment_score'].astype(float)

    df['label_name'] = df['fira_label'].apply(relevance_name)
    df['is_strong_relevant'] = (df['fira_label'].astype(int) >= MIN_STRONG_RELEVANCE).astype(int)
    print('[CACHE] passage_units FiRA chargés')
    return df


df_passage_units = _load_cached_passage_units_with_fira_label()

if df_passage_units is None:
    df_passage_units = df_spans.copy()
    if len(df_passage_units):
        df_passage_units['unit_id'] = df_passage_units.apply(lambda r: f'{r["qid"]}__{r["doc_id"]}__{r["passage_id"]}', axis=1)

        # Label final = label FiRA direct du snippet/passage.
        if 'fira_label' not in df_passage_units.columns:
            df_passage_units['fira_label'] = df_passage_units['passage_relevance'].astype(int)
        df_passage_units['fira_label'] = df_passage_units['fira_label'].astype(int)
        df_passage_units['passage_relevance'] = df_passage_units['fira_label']  # alias compatibilité
        df_passage_units['label_name'] = df_passage_units['fira_label'].apply(relevance_name)
        df_passage_units['is_strong_relevant'] = (df_passage_units['fira_label'] >= MIN_STRONG_RELEVANCE).astype(int)

        # Ajout pratique : texte de la requête pour faciliter le training ensuite.
        query_map = dict(zip(topics['qid'].astype(str), topics['query'].astype(str)))
        query_clean_map = dict(zip(topics['qid'].astype(str), topics['query_clean'].astype(str)))
        df_passage_units['query'] = df_passage_units['qid'].astype(str).map(query_map)
        df_passage_units['query_clean'] = df_passage_units['qid'].astype(str).map(query_clean_map)

        ordered_cols = [
            'unit_id', 'qid', 'query', 'query_clean', 'doc_id', 'passage_id',
            'passage_text', 'fira_label', 'passage_relevance', 'label_name', 'is_strong_relevant',
            'doc_relevance', 'doc_rank', 'doc_score', 'start_token', 'end_token',
            'alignment_score', 'alignment_method'
        ]
        ordered_cols = [c for c in ordered_cols if c in df_passage_units.columns]
        df_passage_units = df_passage_units[ordered_cols].sort_values(['qid', 'doc_rank', 'doc_id', 'start_token']).reset_index(drop=True)
    save_tsv(df_passage_units, PATH_PASSAGE_UNITS)

print('Passage units:', df_passage_units.shape, 'requêtes:', df_passage_units.qid.nunique() if len(df_passage_units) else 0)
if len(df_passage_units):
    print('Distribution fira_label :')
    display(df_passage_units.fira_label.value_counts().sort_index())
safe_display(df_passage_units)

metadata = json.loads(PATH_METADATA.read_text()) if PATH_METADATA.exists() else {}
metadata.update({
    'num_passage_units': int(len(df_passage_units)),
    'num_strong_passage_units': int((df_passage_units.fira_label.astype(int) >= MIN_STRONG_RELEVANCE).sum()) if len(df_passage_units) else 0,
    'final_unit_type': 'fira_snippet_passage_no_windows',
    'final_label_column': 'fira_label',
    'final_label_source': 'fira_direct_from_snippet_qrels',
    'uses_silver_label': False,
})
save_json(metadata, PATH_METADATA)
print('Artefacts finaux:', ARTIFACT_DIR)

# %%
# =====================================================================
# 11) Vérification finale des fichiers produits
# =====================================================================
required_outputs = {
    'queries': PATH_QUERIES,
    'doc_qrels': PATH_DOC_QRELS,
    'snippet_qrels': PATH_PASS_QRELS,
    'snippets_standardized': PATH_SNIPPETS,
    'reconstructed_docs': PATH_RECON_DOCS,
    'all_snippet_spans': PATH_ALL_SNIPPET_SPANS,
    'topdocs': PATH_TOPDOCS,
    'relevant_topdocs': PATH_RELEVANT_TOPDOCS,
    'aligned_spans': PATH_ALIGNED_SPANS,
    'passage_units': PATH_PASSAGE_UNITS,
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
        'path': str(path),
    })
check_df = pd.DataFrame(rows)
display(check_df)

assert check_df['exists'].all(), 'Certains artefacts sont manquants.'
assert is_valid_terrier_index(DOC_INDEX_DIR), 'Index PyTerrier FiRA invalide.'
assert len(df_passage_units) > 0, 'Aucune unité passage/snippet produite.'

cols = pd.read_csv(PATH_PASSAGE_UNITS, sep='\t', nrows=1).columns

assert 'passage_relevance' in cols, 'passage_units.tsv doit contenir passage_relevance.'
assert 'silver_label' not in cols, 'passage_units.tsv ne doit plus contenir silver_label.'

print('[OK] Notebook FiRA sans fenêtres terminé correctement avec passage_relevance direct.')