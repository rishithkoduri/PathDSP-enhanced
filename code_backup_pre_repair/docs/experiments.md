# Experimental Protocols and Evaluation Methodology

## 1. Experimental Regimes

Four distinct evaluation protocols are implemented to rigorously assess in-distribution and out-of-distribution generalization:

### Experiment A: Random Pair Split (In-Distribution Reference)
- 80% Training (30,185 pairs), 10% Validation (3,773 pairs), 10% Test (3,774 pairs).
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
- **Variant D**: Full PathDSP-PIU model.
- **Variant E**: Full model without DG-Net (evaluating reliance on drug-target pathways).
- **Variant F**: Full model without somatic mutation and CNV pathways (expression-only omics).
- **Variant G**: Full model single-seed deterministic run (evaluating contribution of ensemble).
