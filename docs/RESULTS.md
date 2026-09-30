# Verified results

## Main character-span evaluation

43 queries were evaluated with FiRA-aligned character spans.

| Model | CharMAP | CharP@1 | CharP@10 |
| --- | ---: | ---: | ---: |
| QL | 0.179333 | 0.599026 | 0.613519 |
| SDM | **0.182393** | **0.600078** | **0.615437** |
| QL-Interpolated | 0.178045 | 0.581493 | 0.608264 |
| PM-TFIDF | 0.056372 | 0.267400 | 0.282412 |
| PM-Dirichlet | 0.083935 | 0.438033 | 0.389038 |
| PM-SkewedGaussian | 0.062396 | 0.397794 | 0.352316 |

## Secondary window-level evaluation

| Model | MAP | P@1 | P@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| QL | 0.139041 | 0.627907 | 0.662791 | 0.526032 |
| SDM | **0.140809** | 0.627907 | **0.672093** | **0.535954** |
| QL-Interpolated | 0.138655 | 0.627907 | 0.665116 | 0.521305 |
| PM-TFIDF | 0.062901 | 0.325581 | 0.334884 | 0.277802 |
| PM-Dirichlet | 0.090469 | 0.465116 | 0.451163 | 0.377336 |
| PM-SkewedGaussian | 0.066484 | 0.418605 | 0.413953 | 0.332830 |

## Input-document-quality study

Using PM-SkewedGaussian, changing the candidate-document set materially changed passage performance. An oracle-style subset restricted to documents containing strongly relevant windows improved CharMAP from 0.062396 (Top-50) to 0.080547, illustrating how strongly passage retrieval depends on the first-stage document retrieval.

## Interpretation

The strongest conclusion is comparative rather than absolute: in this adapted setting, SDM consistently outperformed the other main lexical/positional baselines, while the positional models did not reproduce the ranking reported in the original paper.
