# Data Sources, Preprocessing, and Provenance

## 1. Primary Dataset (GDSC)

The primary dataset is derived from the **Genomics of Drug Sensitivity in Cancer (GDSC)**, released as part of the authoritative PathDSP Zenodo benchmark:
- **Zenodo Record**: [7532963](https://zenodo.org/records/7532963)
- **Uncompressed Raw File**: `input.txt` (SHA-256: `4f8db09c20ef6f01ed500786cc4458ee860cfaf80d8cdb7d2ede3898d99b2eee`)
- **Total Observations**: 37,732 drug–cell line pairs
- **Unique Compounds**: 144 drugs
- **Unique Cell Lines**: 319 Sanger Cell Model Passport cancer lines

### 1.1 Target Variable Definition
- **Column**: `resp`
- **Definition**: $-\log_{10}(\mathrm{IC50})$ where $\mathrm{IC50}$ is measured in micromolar ($\mu\mathrm{M}$).
- **Summary Statistics**:
  - Mean: $-1.0622$
  - Standard Deviation: $1.2038$
  - Minimum: $-4.0736$
  - 25th Percentile: $-1.9152$
  - Median: $-1.2822$
  - 75th Percentile: $-0.5249$
  - Maximum: $+3.7387$
- **Semantic Interpretation**: Higher values denote greater cytotoxic sensitivity; lower values denote resistance.

### 1.2 Drug Count Curation (144 vs. 153 Drugs)
While the narrative text of Tang & Gottlieb (2021) mentions 153 GDSC screening compounds, the official Zenodo release (record 7532963) distributes an audited dataset containing **144 unique compounds** across 319 cancer cell lines (37,732 pair rows). The 9-compound reduction results from:
1. Strict filtering of compounds lacking definitive canonical SMILES / PubChem identifiers required for generating unambiguous 256-bit radius-2 Morgan fingerprints.
2. Removal of screening compounds without curated target annotations in ChEMBL / STRING v11 required for computing the 196-dimensional NetPEA drug-gene target network enrichment scores (`TARGETnNEIGHBOR_`).
3. Complete observation filtering ensuring zero missing values across all 1,031 features.

All experiments in this implementation strictly adhere to the 144 verified compounds to ensure zero imputation artifacts and full data provenance.

---

## 2. Feature Schema and Modalities

| Block | Prefix in File | Dimensionality | Description |
|---|---|---|---|
| **CHEM** | `CHEM_mBit_0` ... `255` | 256 bits | Binary [0, 1] Morgan chemical fingerprints (radius 2) |
| **DG-Net** | `TARGETnNEIGHBOR_` | 196 pathways | NetPEA drug target & network-expanded pathway enrichment |
| **EXP** | `EXP_` | 187 pathways | ssGSEA gene expression enrichment (aligned to 196 PID pathways with zero imputation) |
| **MUT-Net** | `MUT_` | 196 pathways | NetPEA somatic mutation network enrichment on STRING v11 PPI |
| **CNV-Net** | `CNV_` | 196 pathways | NetPEA copy-number alteration enrichment |
| **Total** | - | **1,031 features** | $256 + 196 + 187 + 196 + 196 = 1031$ |

### 2.1 The 9 Missing Expression Pathways
In the official Zenodo data release, ssGSEA gene expression enrichment is present for 187 out of 196 PID pathways. The 9 missing pathways are:
`PID_ALK2_PATHWAY`, `PID_IL5_PATHWAY`, `PID_INTEGRIN4_PATHWAY`, `PID_P38_GAMMA_DELTA_PATHWAY`, `PID_RANBP2_PATHWAY`, `PID_S1P_S1P4_PATHWAY`, `PID_TCR_JNK_PATHWAY`, `PID_TCR_RAS_PATHWAY`, `PID_VEGF_VEGFR_PATHWAY`.

In PathDSP-PIU, these 9 pathways are harmonized by mapping to the canonical 196 pathway catalog and assigned zero (neutral standard-scaled value), preserving consistent channel alignment $(196 \times 3)$.

---

## 3. Preprocessing and Leakage Prevention

Strict scientific safeguards are enforced:
1. **Disjoint Partitioning**: No sample appears across train, validation, and test splits.
2. **Train-Only Standardization**: All `StandardScaler` transformations are fit strictly on training samples; validation and test sets are transformed using train parameters.
3. **Cold-Start Enforcement**: In LODO, LOCO, and Double-Cold-Start regimes, test entities are completely removed from the training graph.
