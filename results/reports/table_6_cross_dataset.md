# Table 6: Cross-Dataset Transfer Protocol & Assay Harmonization Audit (GDSC to CCLE)

| Parameter | Primary Training Dataset (GDSC2) | External Transfer Target (CCLE) | Harmonization Specification & Impact |
|---|---|---|---|
| **Data Release / Source** | Sanger Cell Model Passports (GDSC2) | Broad Institute DepMap / CCLE (20Q1) | Matched via Sanger SIDM and Broad DepMap IDs |
| **Overlapping Compounds** | 144 compounds (curated Zenodo 7532963) | 24 common compounds (6 directly shared) | Matched via canonical InChIKey and PubChem CID |
| **Overlapping Cell Lines** | 319 cancer cell lines | 478 cancer cell lines (35 shared lines) | Matched via COSMIC SIDM identifiers |
| **Assay Platform** | 72-hour Syto60 / Resazurin viability | 84-hour CellTiter-Glo ATP luminescent | Screening assay kinetics and ATP vs DNA dye differences |
| **Dose Titration** | Extended 9-point dilution series | Capped maximum concentration (8.0 μM) | Upper concentration ceiling causes artificial distribution shift |
| **Target Metric** | Continuous $-\log_{10}(\mathrm{IC50})$ [$\mu$M] | $-\log_{10}(\mathrm{IC50})$ truncated at 8.0 μM | Ceiling truncation restricts dynamic range in CCLE |
| **Gene Expression (EXP)** | ssGSEA on 196 PID pathways | ssGSEA on 196 PID pathways | Identical PID pathway projection pipeline |
| **Mutation (MUT-Net)** | NetPEA on STRING v11 PPI network | NetPEA on STRING v11 PPI network | Network propagation enrichment harmonized |
| **Copy Number (CNV-Net)** | NetPEA on STRING v11 PPI network | NetPEA on STRING v11 PPI network | Network propagation enrichment harmonized |
| **External Validation Status** | Benchmarked & Verified | **Planned Future Work** | Full external deployment requires batch-effect adjustment |
