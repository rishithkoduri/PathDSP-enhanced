# Table 2: Random-Split Performance Comparison (In-Distribution Reference)

| Model | MAE (lower is better) | RMSE (lower is better) | R2 (higher is better) | Pearson r | Spearman rho |
|---|---|---|---|---|---|
| DrugMean | 0.4999 | 0.6590 | 0.7024 | 0.8382 | 0.7648 |
| CellMean | 0.8759 | 1.1454 | 0.1010 | 0.3199 | 0.3571 |
| AdditiveMean | 0.3877 | 0.5209 | 0.8141 | 0.9023 | 0.8670 |
| ElasticNet | 0.4125 | 0.5476 | 0.7945 | 0.8924 | 0.8510 |
| RandomForest | 0.3938 | 0.5265 | 0.8100 | 0.9003 | 0.8612 |
| XGBoost | 0.3620 | 0.4834 | 0.8399 | 0.9173 | 0.8829 |
| PathDSP_FNN | **0.3028** | **0.4056** | **0.8873** | 0.9421 | 0.9122 |
| PathDSP-PIU (Proposed Ensemble) | 0.3084 | 0.4078 | 0.8860 | **0.9437** | **0.9168** |

*Note: Bold denotes the best performance per metric. On in-distribution random splits where both drugs and cell lines are observed in training, the baseline PathDSP FNN achieves slightly lower MAE/RMSE, while the proposed PathDSP-PIU ensemble achieves higher correlation metrics.*
