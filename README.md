# Passage Retrieval on FiRA / TREC-DL

Research-oriented reproduction/adaptation of **answer-passage retrieval** experiments inspired by *Retrieving Passages and Finding Answers*, using FiRA annotations and TREC-DL 2019 resources under a constrained academic setting.

The project compares lexical and positional passage-ranking approaches while being explicit about a key scientific limitation: this is **not a strict reproduction** of the original GOV2 experiment. The collection, document reconstruction and relevance annotations differ, so absolute scores should not be compared directly with the source paper.

## Problem

Given an information need and a set of initially retrieved documents, rank text passages/windows so that passages containing the answer appear as high as possible.

The main adapted pipeline uses:

- TREC-DL 2019 queries/documents,
- FiRA answer-passage annotations,
- SDM top-50 document retrieval,
- reconstructed document text,
- 50-token windows with a 25-token stride,
- character-span alignment for FiRA relevance labels.

## Models evaluated

- Query Likelihood (**QL**)
- Sequential Dependence Model (**SDM**)
- **QL-Interpolated**
- Positional models: PM-TFIDF, PM-Dirichlet, PM-SkewedGaussian
- Relevance-model / query-expansion variants over documents and passages
- Input-document-quality ablations (Top-5 / 10 / 25 / 50 / oracle-style relevant subset)

## Main retained result

On the 43-query FiRA/TREC-DL adaptation, **SDM** was the strongest of the main lexical baselines:

| Model | CharMAP | CharP@1 | CharP@10 |
| --- | ---: | ---: | ---: |
| QL | 0.179333 | 0.599026 | 0.613519 |
| **SDM** | **0.182393** | **0.600078** | **0.615437** |
| QL-Interpolated | 0.178045 | 0.581493 | 0.608264 |
| PM-TFIDF | 0.056372 | 0.267400 | 0.282412 |
| PM-Dirichlet | 0.083935 | 0.438033 | 0.389038 |
| PM-SkewedGaussian | 0.062396 | 0.397794 | 0.352316 |

A secondary window-level evaluation also gave SDM the strongest values among these six models (MAP 0.140809, P@10 0.672093, nDCG@10 0.535954).

## Repository structure

```text
.
├── experiments/
│   ├── 01_build_fira_window_dataset.py
│   ├── 02_main_passage_ranking.py
│   └── auxiliary/
│       ├── msmarco_build_dataset.py
│       ├── msmarco_modeling.py
│       ├── trecdl_build_no_windows.py
│       ├── trecdl_modeling_sdm.py
│       ├── trecdl_modeling_char_eval.py
│       ├── fira_no_windows_build.py
│       └── fira_no_windows_modeling.py
├── docs/
│   ├── RESULTS.md
│   └── REPRODUCIBILITY.md
├── requirements.txt
└── README.md
```

## Scientific scope and limitations

This repository deliberately preserves the distinction between **internal comparison** and **paper reproduction**.

The original GOV2 collection and manual annotations were not reusable in the academic environment. MS MARCO was investigated but was not retained as the primary setting because of its relevance structure, and executing the full TREC-DL collection was too expensive for the available compute. The final protocol therefore uses an adapted FiRA/TREC-DL 2019 setting.

Consequently:

- absolute scores are **not directly comparable** to the paper;
- positional models behave differently from the original reported setting;
- first-stage document retrieval quality strongly constrains downstream passage retrieval;
- a natural next step is hybrid lexical + semantic/dense retrieval while preserving the same evaluation format.

## Academic context

**Sorbonne Université — Master 1 MIND, Information Retrieval / Information Access, 2025–2026**

Authors:
- **Zineddine Mohammedi**
- **Wafaa Berrais**
- **Lina Mnemoi**

This repository represents the joint academic project and does not invent an undocumented individual contribution split.