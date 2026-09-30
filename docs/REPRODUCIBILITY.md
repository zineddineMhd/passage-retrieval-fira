# Reproducibility notes

The original project was executed in Google Colab and uses PyTerrier / `ir_datasets` plus local Google Drive paths.

## Main workflow

1. Build the FiRA/TREC-DL window dataset with `01_build_fira_window_dataset.py`.
2. Run `02_main_passage_ranking.py` to score and evaluate QL, SDM, interpolated and positional models, plus the later ablations.

The `auxiliary/` directory preserves earlier or alternative experiments on MS MARCO, TREC-DL without windows and FiRA variants. They are useful for understanding the experimental path but are not the final primary protocol.

## Data

Large source collections are not redistributed. You must configure the corresponding `ir_datasets` resources and output paths before running the scripts.

## Important evaluation caveat

This code supports an adapted experiment, not a strict GOV2 reproduction. Do not compare the absolute FiRA/TREC-DL values directly with the source paper as if they were measured on the same collection and annotations.