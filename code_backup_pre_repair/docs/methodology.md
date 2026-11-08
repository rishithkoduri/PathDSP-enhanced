# Methodology: PathDSP-PIU

## 1. Scientific Motivation and Problem Formulation

Drug sensitivity prediction in personalized oncology aims to estimate the therapeutic response of a patient or cancer cell line to a given compound based on multi-omics molecular profiling. The seminal work of Tang & Gottlieb (*Scientific Reports*, 2021) introduced **PathDSP**, demonstrating that compressing high-dimensional genomic features into curated biological pathways improves both predictive performance and interpretability.

However, standard deep neural networks (including PathDSP's original feedforward neural network) suffer from two fundamental limitations:
1. **Unconditioned Linear Concatenation**: Features from drugs (chemical fingerprints, drug-target enrichment) and cell lines (gene expression, somatic mutations, copy-number alterations) are merely concatenated into a flat vector. The network must learn multi-omics drug–pathway interactions through generic dense layers, often memorizing drug or cell-line identity rather than true biochemical interaction mechanisms.
2. **Deterministic Point Predictions without Uncertainty**: Standard FNNs provide single scalar predictions with no measure of confidence. When presented with unseen compounds, rare cancer genotypes, or novel disease contexts, the model cannot flag predictions prone to large errors.

**PathDSP-PIU** (*Pathway Interaction with Uncertainty*) addresses these limitations through:
- **Drug-conditioned pathway gating**: Dynamically re-weighting cell-line pathway activity conditioned on drug properties.
- **Low-rank multiplicative pathway interaction**: Explicit modeling of drug–pathway synergy $(I_{d,p} = (U e_d) \odot (V h'_{c,p}))$.
- **Ensemble-based uncertainty quantification**: Decoupling aleatoric and epistemic uncertainty via deep ensembles and split conformal calibration.

---

## 2. Mathematical Formulation

Let $d \in \mathcal{D}$ denote a therapeutic compound and $c \in \mathcal{C}$ denote a cancer cell line. The ground-truth response $y_{d,c} \in \mathbb{R}$ is defined as the negative log-transformed half-maximal inhibitory concentration:
$$y_{d,c} = -\log_{10}(\mathrm{IC50})$$
where $\mathrm{IC50}$ is measured in micromolar ($\mu\mathrm{M}$). Higher $y_{d,c}$ strictly indicates higher drug sensitivity.

### 2.1 Feature Representations
- **Drug Chemical Structure ($\mathbf{x}_{\text{chem}} \in \{0, 1\}^{256}$)**: 256-bit Morgan fingerprints (radius 2) computed via RDKit.
- **Drug-Target Network Enrichment ($\mathbf{x}_{\text{dg}} \in \mathbb{R}^{196}$)**: NetPEA enrichment scores across 196 cancer signaling pathways from the Pathway Interaction Database (PID).
- **Cell Multi-Omics Pathway Matrix ($\mathbf{X}_c \in \mathbb{R}^{P \times M}$)**:
  - $P = 196$ pathways.
  - $M = 3$ modalities:
    - Modality 0: Gene expression pathway enrichment via ssGSEA ($\mathbf{x}_{\text{exp}} \in \mathbb{R}^{196}$).
    - Modality 1: Somatic mutation pathway enrichment via NetPEA on STRING v11 PPI ($\mathbf{x}_{\text{mut}} \in \mathbb{R}^{196}$).
    - Modality 2: Copy-number alteration enrichment via NetPEA ($\mathbf{x}_{\text{cnv}} \in \mathbb{R}^{196}$).

---

### 2.2 Model Architecture

#### A. Drug Encoder
Concatenates chemical structure and drug-target network enrichment:
$$\mathbf{x}_{\text{drug}} = [\mathbf{x}_{\text{chem}} \,\|\, \mathbf{x}_{\text{dg}}] \in \mathbb{R}^{452}$$
$$\mathbf{e}_d = \text{LayerNorm}\left(\text{GELU}\left(\mathbf{W}_{d2} \cdot \text{Dropout}(\text{GELU}(\mathbf{W}_{d1} \mathbf{x}_{\text{drug}}))\right)\right) \in \mathbb{R}^{d_e}$$
where $d_e = 128$.

#### B. Cell-Line Pathway Encoder
Encodes each pathway $p \in \{1, \dots, P\}$ from its 3-channel multi-omics vector $\mathbf{X}_c[p, :] \in \mathbb{R}^3$:
$$\mathbf{h}(c, p) = \text{LayerNorm}\left(\text{GELU}\left(\mathbf{W}_{c2} \cdot \text{Dropout}(\text{GELU}(\mathbf{W}_{c1} \mathbf{X}_c[p, :]))\right)\right) \in \mathbb{R}^{d_h}$$
Global cell line representation:
$$\mathbf{h}_c = \frac{1}{P} \sum_{p=1}^P \mathbf{h}(c, p) \in \mathbb{R}^{d_h}$$
where $d_h = 32$.

#### C. Drug-Conditioned Pathway Gating
For each pathway $p$, the relevance of the pathway is modulated by the global drug embedding $\mathbf{e}_d$ and the drug's specific target enrichment in that pathway $\mathbf{x}_{\text{dg}}[p]$:
$$\mathbf{g}(d, p) = \sigma\left(\mathbf{w}_g^\top \cdot \text{GELU}\left(\mathbf{W}_g [\mathbf{e}_d \,\|\, \mathbf{h}(c, p) \,\|\, \mathbf{x}_{\text{dg}}[p]]\right)\right) \in [0, 1]$$
$$\mathbf{h}'(d, p) = \mathbf{g}(d, p) \odot \mathbf{h}(c, p) \in \mathbb{R}^{d_h}$$

#### D. Multiplicative Pathway Interaction
To capture non-linear biochemical interaction between drug modes and cellular pathway states:
$$\mathbf{u}(d) = \mathbf{U} \mathbf{e}_d \in \mathbb{R}^{d_i}, \quad \mathbf{v}(d, p) = \mathbf{V} \mathbf{h}'(d, p) \in \mathbb{R}^{d_i}$$
$$\mathbf{z}(d, p) = \mathbf{u}(d) \odot \mathbf{v}(d, p) \in \mathbb{R}^{d_i}$$
Pathway importance weights are computed using an attention mechanism:
$$\alpha(d, p) = \frac{\exp(\mathbf{w}_\alpha^\top [\mathbf{u}(d) \,\|\, \mathbf{v}(d, p)])}{\sum_{p'=1}^P \exp(\mathbf{w}_\alpha^\top [\mathbf{u}(d) \,\|\, \mathbf{v}(d, p')])}$$
The aggregated drug–pathway interaction representation:
$$\mathbf{z}(d) = \sum_{p=1}^P \alpha(d, p) \cdot \mathbf{z}(d, p) \in \mathbb{R}^{d_i}$$
where $d_i = 32$.

#### E. Regression Head & Residual Connection
To preserve the direct main-effects signal from the original PathDSP architecture, a linear residual projection is applied to the flat feature vector $\mathbf{x}_{\text{flat}} \in \mathbb{R}^{1031}$:
$$\mathbf{r}_{\text{res}} = \text{Dropout}(\text{GELU}(\mathbf{W}_r \mathbf{x}_{\text{flat}})) \in \mathbb{R}^{64}$$
Joint representation:
$$\mathbf{f}_{\text{joint}} = [\mathbf{e}_d \,\|\, \mathbf{h}_c \,\|\, \mathbf{z}(d) \,\|\, \mathbf{r}_{\text{res}}] \in \mathbb{R}^{128 + 32 + 32 + 64 = 256}$$
$$\hat{y}_{d,c} = \mathbf{w}_{\text{out}}^\top \cdot \text{MLP}(\mathbf{f}_{\text{joint}}) \in \mathbb{R}$$

---

## 3. Uncertainty Estimation and Calibration

### 3.1 Deep Ensemble
An ensemble of $K = 3$ independently initialized models $\theta_1, \theta_2, \theta_3$ is trained using distinct random seeds (42, 1042, 2042):
- Predictive Mean:
  $$\mu(x) = \frac{1}{K} \sum_{k=1}^K \hat{y}(x; \theta_k)$$
- Predictive Variance:
  $$\sigma^2(x) = \frac{1}{K-1} \sum_{k=1}^K (\hat{y}(x; \theta_k) - \mu(x))^2$$
- Predictive Uncertainty: $\sigma(x) = \sqrt{\sigma^2(x)}$.

### 3.2 Split Conformal Prediction
On a held-out calibration set $\mathcal{D}_{\text{cal}} = \{(x_i, y_i)\}_{i=1}^n$, normalized nonconformity scores are computed:
$$s_i = \frac{|y_i - \mu(x_i)|}{\sigma(x_i) + \epsilon}$$
For target miscoverage level $\alpha \in \{0.10, 0.05\}$, the empirical conformal quantile is:
$$\hat{q}_{1-\alpha} = \text{Quantile}\left(\{s_i\}_{i=1}^n, \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$$
The locally-adaptive prediction interval for a test sample $x^*$ is:
$$\mathcal{C}_{1-\alpha}(x^*) = \left[\mu(x^*) - \hat{q}_{1-\alpha}(\sigma(x^*) + \epsilon), \; \mu(x^*) + \hat{q}_{1-\alpha}(\sigma(x^*) + \epsilon)\right]$$
This guarantees exact finite-sample coverage:
$$P(y^* \in \mathcal{C}_{1-\alpha}(x^*)) \ge 1 - \alpha$$
under exchangeability.
