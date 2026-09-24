# PathGDT — Explainable Drug-Sensitivity (v3)

Reproduces PathDSP and extends it with **PathGDT**: a BPE-SMILES Transformer **+ a
molecular-graph GNN**, joined to the cancer pathways by **cross-attention**. One
Streamlit app does the whole loop — build real features from a zip, cross-validate
(incl. unseen-cell / unseen-drug), explain with SHAP, **save models**, and **predict
on new data** with a saved model.

```
DRUG ─┬ SMILES → BPE tokens → Transformer ─┐
      └ SMILES → molecular graph → GNN ─────┤
                                            ├─ cross-attention → head → −logIC50 → IC50
CANCER ─ EXP+MUT+CNV pathway scores (tokens)┘   (+ DG drug-target & CHEM fingerprint)
```

The GNN is a **pure-PyTorch GCN** (Kipf–Welling message passing) — **no `torch_geometric`
needed**, so there are no `torch-scatter`/`torch-sparse` wheel headaches on Windows. It's
built from the drug's atoms/bonds via RDKit, on by default, and only affects PathGDT (the
FNN baseline ignores it). Turn it off with the **GNN drug-graph branch** checkbox.

**Result so far (leave-cell-out, z-scored, comparable to the paper):**
PathGDT **RMSE 0.53 / MAE 0.40** vs PathDSP LOCO **0.59 / 0.45** — i.e. it *beats*
PathDSP on the honest unseen-cell test, while being explainable.

---

## Install & run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build → RTX 4060
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
streamlit run app.py
```
Header must read **⚡ GPU: NVIDIA GeForce RTX 4060 (cuda:0)**. (CUDA never sees the AMD
iGPU, so `cuda:0` is always the RTX 4060.)

## Two modes (sidebar)

### 🚂 Train & evaluate
Upload one zip of the 5 CSVs (`OmicsExpression…`, `OmicsSomaticMutations…Hotspot`,
`OmicsCNGene…`, `Model.csv`, `GDSC_DATASET.csv`). The app unpacks, reads all rows,
builds features, **deletes the unpacked files**, then cross-validates with a **live
epoch / loss / ETA** readout. Split options include **`ALL (mixed + LOCO + LODO)`** to
run all three in one go, plus **▶ Compare 4** (FNN & PathGDT × mixed & LOCO → one table).
Outputs: metrics, per-fold table, predicted-vs-actual, **ROC-AUC** (sensitive vs
resistant), training curve, SHAP pathway attribution, and downloads
(**predictions incl. IC50**, **trained model .pt**).

### 🔮 Predict with a saved model
Pick a model from the **/models dropdown** (or upload a `.pt`), upload a new-data zip in
the same 5-CSV format, and get predictions with **LN(IC50) and IC50 (µM)** — plus metrics
and an ROC curve if the new data carries responses. Your existing
`pathgdt_pathgdt_leave-cell-out.pt` loads here directly.

## Settings — mostly just a preset

| Preset | Model | Notes |
|---|---|---|
| **Fast** | d96·1L·20ep·b128 | quick checks |
| **Balanced** ← default | d128·2L·30ep·b128 | high quality **and** fast |
| **Research** | d192·3L·45ep·b96 | best quality |

Recommended: **PathGDT · Balanced · bpe · Z-score ON · 5 folds · AMP ON · Save ON**,
split = **leave-cell-out** (or **ALL** for the full table).

> **Speed fix (important):** pathway self-attention over 588 tokens is what made v2 take
> hours. v3 keeps the drug↔pathway **cross-attention** (the interpretable part) but drops
> the O(588²) pathway self-attention by default, so runs are **minutes, not hours**. If you
> want the extra-heavy variant, tick *Pathway self-attention* in Advanced (much slower).
> Leave *torch.compile* OFF on a laptop GPU — it triggers slow recompiles.
> **CUDA OOM?** lower batch in Advanced (128 → 96 → 64).

## Reload a saved model in code
```python
import torch
b = torch.load("models/pathgdt_pathgdt_leave-cell-out_YYYYMMDD-HHMM.pt",
               map_location="cuda:0", weights_only=False)
# b has: model_kind, cfg, cancer_cols, extra_cols, tok, norm, split, state_dict
```
The saved model is trained on **100% of the data** (CV models are only for scoring and are
discarded). Every saved model lands in `./models/` and shows up in the Predict dropdown.

## Notes
- Data pipeline verified on real DepMap+GDSC files (45,637 rows × 1040 features).
- VRAM is freed after each run; a **🧹 Free GPU memory** button is in the sidebar. (The CUDA
  caching allocator keeps some memory *reserved* even after freeing — that's normal; it's
  released when the app process exits.)
- Degrades gracefully: no `tokenizers` → regex tokenizer; no `shap`/multimodal → gradient×input.
- **R²/PCC/SCC are scale-free** (compare across papers); RMSE/MAE depend on z-score vs native units.
- First GPU run is the real checkpoint — if anything throws, send the traceback.
