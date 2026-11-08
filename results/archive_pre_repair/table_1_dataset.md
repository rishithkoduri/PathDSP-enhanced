# Table 1: Dataset Characteristics (GDSC Primary Benchmark)

| Characteristic | Measured Value | Unit / Format | Description |
|---|---|---|---|
| **Response Observations (Pairs)** | **37,732** | Pairs | Screened drug–cell line pairs |
| **Unique Compounds / Drugs** | **144** | Chemical Entities | Audited FDA-approved & investigational compounds |
| **Unique Cancer Cell Lines** | **319** | Cell Lines | Sanger / COSMIC SIDM models |
| **Chemical Fingerprint (CHEM)** | **256** | Binary Bits | Radius-2 Morgan fingerprints |
| **Drug-Gene Pathways (DG-Net)** | **196** | Continuous Scores | NetPEA target-network enrichment |
| **Expression Pathways (EXP)** | **187** | Continuous Scores | ssGSEA enrichment (aligned to 196 reference pathways) |
| **Mutation Pathways (MUT-Net)** | **196** | Continuous Scores | NetPEA somatic mutation network enrichment |
| **Copy Number Pathways (CNV-Net)** | **196** | Continuous Scores | NetPEA copy number variation enrichment |
| **Total Flat Features** | **1,031** | Features | Combined multi-omics representation |
| **Target Variable (`resp`)** | **$-\log_10(\mathrm{IC50})$** | $\mu\mathrm{M}$ scale | Mean: -1.0622, Std: 1.2038, Range: [-4.07, 3.74] |
| **Missing Values / NaNs** | **0** | Counts | 100% complete feature and response matrix |
| **Matrix Density** | **82.14%** | Percentage | 37732 / (144 × 319) |
