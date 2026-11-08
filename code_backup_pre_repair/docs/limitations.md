# Limitations and Scientific Caveats

When interpreting the findings of this study, several critical scientific and computational limitations must be acknowledged:

## 1. Observational and Pharmacogenomic Assay Limitations
- **Assay Discrepancies**: Screening protocols (e.g. GDSC Syto60 viability vs. CCLE CellTiter-Glo ATP luminescent assay) exhibit recognized differences in cellular readout kinetics and dynamic ranges.
- **Concentration Truncation**: CCLE caps compound screening concentrations at 8.0 μM, whereas GDSC evaluates higher micromolar ranges for resistant compounds. This induces an artificial ceiling effect when transferring models across datasets.

## 2. Incompleteness of Drug Target Annotations
- The DG-Net feature representation relies on known primary targets curated in DGIdb and expanded via STRING PPI. Investigational compounds or poorly characterized chemical scaffolds with incomplete target information rely almost exclusively on Morgan chemical fingerprints (CHEM).

## 3. Cell-Line vs. In-Vivo Tumor Biology
- Cancer cell lines cultured on 2D plastic lack the three-dimensional architecture, vascular microenvironment, extracellular matrix stiffness, and immune cell infiltration present in patient solid tumors.

## 4. Uncertainty Estimation Boundaries
- Ensemble variance $\sigma^2(x)$ measures epistemic disagreement among initialized models under the training distribution. While empirical conformal coverage provides valid finite-sample coverage on exchangeable validation sets, out-of-distribution coverage in extreme double-cold-start regimes may experience coverage slack due to covariate shift.

## 5. Non-Causal Nature of Pathway Attributions
- Gating scores $g(d, p)$ and interaction weights $\alpha(d, p)$ identify predictive statistical associations between drug features and cellular pathways. They should be interpreted as mechanistic hypotheses for experimental validation rather than proven causal mechanisms.
