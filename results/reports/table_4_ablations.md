# Table 4: Architectural Ablation Study (Variants A through G)

| Variant | Architectural Configuration | MAE | RMSE | R2 | Pearson r | Spearman rho |
|---|---|---|---|---|---|---|
| Variant_A_Original_FNN | Original PathDSP FNN baseline (Tang & Gottlieb, 2021) | 0.3045 | 0.4141 | 0.8825 | 0.9401 | 0.9083 |
| Variant_B_FNN_plus_Gating | FNN + Drug-Conditioned Pathway Gating | 0.3172 | 0.4247 | 0.8764 | 0.9367 | 0.9066 |
| Variant_C_FNN_plus_Interaction | FNN + Multiplicative Pathway Interaction | 0.3172 | 0.4218 | 0.8781 | 0.9389 | 0.9094 |
| Variant_D_Full_PathDSP_PIU | **Full PathDSP-PIU (3-Model Ensemble)** | 0.3084 | 0.4078 | 0.8860 | 0.9437 | 0.9168 |
| Variant_E_No_DGNet | No DG-Net (CHEM + cell pathways only, masked in residual) | 0.3155 | 0.4176 | 0.8805 | 0.9396 | 0.9099 |
| Variant_F_No_MUT_CNV | No MUT/CNV (Expression-only omics, masked in residual) | 0.3224 | 0.4304 | 0.8731 | 0.9348 | 0.9025 |
| Variant_G_Single_Model_No_Ensemble | Single-Model PathDSP-PIU (Independent Seed 42) | 0.3249 | 0.4271 | 0.8750 | 0.9382 | 0.9087 |

*Ensemble Contribution: Variant G (Single Model) vs. Variant D (3-Model Ensemble) yields Delta MAE = +0.0165 and Delta RMSE = +0.0193, demonstrating a 5.1% error reduction from ensembling on identical splits and features.*
