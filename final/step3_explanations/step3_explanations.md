# Step 3 — do the explanations point to the drug's targets?

Model: step 1's recipe run (20261109-184612_PAPER_input_1-recipe), its best fold: 3,773 held-out pairs, 3 seed(s), SHAP DeepExplainer. A pathway's score for a drug = mean |SHAP| of its EXP, MUT and CNV features over that drug's pairs (cell-line side only; DG is built from the targets, so it is reported only as a circular control). 128 drugs with ≥ 8 pairs; 97 have annotated targets in PID pathways.

## A. Target recovery

| Metric | Model | Random |
| --- | --- | --- |
| AUROC (target pathways vs others) | 0.499 | 0.500 |
| Precision@5 | 0.052 | 0.103 |
| Hit@5 | 0.216 | — |
| Precision@10 | 0.065 | 0.103 |
| Hit@10 | 0.464 | — |
| Precision@20 | 0.072 | 0.103 |
| Hit@20 | 0.680 | — |
| AUROC of the DG attribution (circular control) | 0.630 | 0.500 |

**Permutation control** (target sets shuffled between drugs, 1000×): mean AUROC 0.499 vs 0.498 under the null (p = 0.470); P@10 0.065 vs 0.054 (p = 0.019).

## B. Stability across seeds

Per drug, the pathway scores of two seeds correlate at Spearman 0.735; their top-10 pathways overlap with Jaccard 0.410.

## C. Deletion test (faithfulness)

A pathway is deleted by replacing its EXP / MUT / CNV values with the training mean; the change is measured on the drug's held-out pairs (mean |Δ prediction| in z units, and the within-drug PCC with the measured response).

| Comparison | First | Second | Drugs where first > second | Wilcoxon p |
| --- | --- | --- | --- | --- |
| top-5 vs 5 random pathways: mean |Δ prediction| | 0.0819 | 0.0436 | 128 / 128 | 9.5e-23 |
| top-10 vs 10 random: mean |Δ prediction| | 0.1225 | 0.0663 | 128 / 128 | 9.5e-23 |
| 10 random vs top-10: within-drug PCC after deletion | 0.8009 | 0.7919 | 80 / 128 | 3.3e-05 |
| target pathways vs as many random: mean |Δ prediction| | 0.1106 | 0.1020 | 53 / 97 | 0.003 |

Within-drug PCC before deletion: 0.805.

## D. Top cell-line pathways (all drugs)

| pathway | score |
| --- | --- |
| PID_GLYPICAN_1PATHWAY | 0.03494 |
| PID_ARF_3PATHWAY | 0.03394 |
| PID_MAPK_TRK_PATHWAY | 0.02921 |
| PID_WNT_CANONICAL_PATHWAY | 0.02858 |
| PID_RAS_PATHWAY | 0.02823 |
| PID_P73PATHWAY | 0.02776 |
| PID_TGFBR_PATHWAY | 0.02776 |
| PID_RAC1_REG_PATHWAY | 0.02771 |
| PID_NFAT_TFPATHWAY | 0.02753 |
| PID_S1P_META_PATHWAY | 0.02742 |
| PID_DNA_PK_PATHWAY | 0.02724 |
| PID_RHODOPSIN_PATHWAY | 0.02654 |
| PID_ILK_PATHWAY | 0.02572 |
| PID_ERBB_NETWORK_PATHWAY | 0.02566 |
| PID_CD8_TCR_DOWNSTREAM_PATHWAY | 0.02366 |
