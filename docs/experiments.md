# Experimental Protocols and Evaluation Methodology

## 1. Experimental Regimes

Four distinct evaluation protocols are implemented to rigorously assess in-distribution and out-of-distribution generalization:

### Experiment A: Random Pair Split (In-Distribution Reference)
- 70% Training (26,412 pairs), 10% Validation (3,773 pairs for early stopping), 10% Calibration (3,773 pairs for conformal prediction), 10% Test (3,774 pairs for unbiased evaluation).
- Eliminates conformal calibration optimization leakage by decoupling early stopping model selection from nonconformity score quantile fitting.
- Evaluates interpolation performance where both drugs and cell lines have been observed during training.

### Experiment B: Leave-One-Drug-Out (LODO)
- Simulates in-silico screening for a completely unseen novel chemical compound.
- For each evaluated drug $d^*$, all observations involving $d^*$ are placed in the test set.
- The remaining 143 drugs are partitioned into training and validation sets.
- Tested across top prioritized drugs (e.g. PD0325901, Camptothecin, 5-Fluorouracil, Oxaliplatin, Luminespib).

### Experiment C: Leave-One-Cell-Line-Out (LOCO)
- Simulates clinical application to an unseen cancer patient / cell line.
- For each evaluated cell line $c^*$, all pairs involving $c^*$ are held out in the test set.
- Tested across prioritized cell lines representing diverse cancer lineages.

### Experiment D: Double-Cold-Start
- Extreme out-of-distribution evaluation: both drug and cell line are simultaneously unseen during training.
- 15% of drugs and 15% of cell lines are held out.
- The test set consists strictly of pairs $(d, c)$ where $d \in \mathcal{D}_{\text{test}}$ and $c \in \mathcal{C}_{\text{test}}$.
- Intermediate pairs (e.g. unseen drug with seen cell line) are removed from the training candidate pool to ensure zero entity leakage.

---

## 2. Comparison Baseline Models

1. **Drug-Mean Predictor**: Historical mean response for each drug; global mean fallback for unseen drugs.
2. **Cell-Mean Predictor**: Historical mean response for each cell line; global mean fallback for unseen cells.
3. **Additive-Mean Predictor**: Two-way additive main-effects model $(\hat{y}_{d,c} = \bar{y} + (\bar{y}_d - \bar{y}) + (\bar{y}_c - \bar{y}))$.
4. **ElasticNet**: L1/L2 penalized regularized linear regression.
5. **Random Forest**: 100 trees, maximum depth 16.
6. **XGBoost**: Gradient boosted decision trees (150 estimators, learning rate 0.05).
7. **PathDSP FNN**: Faithful reproduction of the 5-layer ELU neural network from Tang & Gottlieb (2021).
8. **PathDSP-PIU (Proposed)**: 3-model deep ensemble with drug-conditioned pathway gating and multiplicative interaction.

---

## 3. Ablation Variants

- **Variant A**: Original PathDSP FNN baseline.
- **Variant B**: FNN + Pathway Gating only ($g(d, p) \odot h(c, p)$).
- **Variant C**: FNN + Multiplicative Interaction only ($z(d, p) = (U e_d) \odot (V h)$).
- **Variant D**: Full PathDSP-PIU model (Gating + Multiplicative Interaction).
- **Variant E**: Full model without DG-Net (strictly masked across all encoder and residual pathways).
- **Variant F**: Full model without somatic mutation and CNV pathways (expression-only omics, strictly masked across all pathways).
- **Variant G**: Single independently initialized PathDSP-PIU model (evaluating contribution of ensembling against single-network variance).

---

## 4. Statistical Evaluation Protocols

To rigorously evaluate whether predictive gains in cold-start regimes are statistically meaningful or merely artifacts of sample-level clustering, four bootstrap resampling schemes are executed ($B = 1,000$ iterations, $\alpha = 0.05$):

1. **Observation-Level Paired Bootstrap**:
   - Resamples individual response test pairs $(d_i, c_i)$ with replacement.
   - Evaluates $\Delta\mathrm{MAE} = \mathrm{MAE}_{\mathrm{baseline}} - \mathrm{MAE}_{\mathrm{proposed}}$.

2. **Drug-Clustered Bootstrap**:
   - Resamples unique test drugs with replacement; includes all observation pairs for selected drugs.
   - Accounts for correlated errors across cell lines exposed to the same drug.

3. **Cell-Clustered Bootstrap**:
   - Resamples unique test cell lines with replacement; includes all observation pairs for selected cell lines.
   - Accounts for correlated errors across drugs tested in the same tumor line.

4. **Two-Way Pigeonhole Clustered Bootstrap (Owen, 2007)**:
   - Resamples unique test drugs and unique test cell lines independently with replacement.
   - Assigns observation multiplicity weights $w_i = m_d(d_i) \times m_c(c_i)$ to account for cluster draw frequencies across the Cartesian cross-product.
   - Evaluates weighted $\Delta\mathrm{MAE}$ and $\Delta\mathrm{RMSE}$, providing the gold standard test of statistical significance under double cold-start clustering.

