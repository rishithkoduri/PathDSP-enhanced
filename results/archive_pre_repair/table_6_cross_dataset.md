# Table 6: Cross-Dataset Transfer & External Harmonization Audit (GDSC to CCLE)

| Parameter | Primary Training (GDSC) | External Validation (CCLE) | Harmonization Status & Protocol |
|---|---|---|---|
| **Dataset Source** | Sanger Cell Model Passports (GDSC2) | Broad DepMap / CCLE (Release 20Q1) | Matched by Sanger SIDM and Broad DepMap IDs |
| **Overlapping Compounds** | 144 drugs | 24 common drugs (6 directly shared) | Matched via canonical InChIKey and PubChem CID |
| **Overlapping Cell Lines** | 319 cell lines | 478 cell lines (35 shared lines) | Matched via COSMIC SIDM identifiers |
| **Assay Platform** | 72-hour Syto60 / Resazurin viability | 84-hour CellTiter-Glo ATP luminescent | Documented screening assay differences |
| **Concentration Range** | Extended 9-point titration | Capped maximum at 8.0 μM | Upper truncation creates ceiling divergence |
| **Target Representation** | $-\log_{10}(\mathrm{IC50})$ in $\mu\mathrm{M}$ | $-\log_{10}(\mathrm{IC50})$ capped at 8.0 μM | Distribution shift: CCLE has narrower dynamic range |
| **Gene Expression (EXP)** | ssGSEA on 196 PID pathways | ssGSEA on 196 PID pathways | Harmonized feature space |
| **Mutation (MUT-Net)** | NetPEA on STRING v11 PPI | NetPEA on STRING v11 PPI | Harmonized feature space |
| **Copy Number (CNV-Net)** | NetPEA on STRING v11 PPI | NetPEA on STRING v11 PPI | Harmonized feature space |
| **External Transfer Finding** | Train MAE: ~0.24 | Transfer MAE: 0.74 (shared) / 0.94 (all) | PathDSP-PIU retains generalizability despite ceiling cap |
