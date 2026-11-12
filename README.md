# PathDSP-X

**Explainable drug sensitivity prediction through cancer pathway enrichment.** Team G1589 · mentor Dr. T. Swathi.
Base paper: Tang & Gottlieb, *Sci. Rep.* 11, 3128 (2021).

PathDSP-X (PathDSP eXtended) is a Streamlit app with a GPU training engine. It reproduces PathDSP and adds Model A (the paper's FNN with a training recipe and a drug-wise z-scored target), Model B (a pathway-graph model trained on the 5 IMPROVE CSA studies) and calibrated 90 % prediction intervals. Its four pages are **Train**, **Live**, **Results** and **Predict**, in a dark or light theme.

---

## Quick start (Windows · RTX 4060)

Double-click **`start.bat`** — that is all, every time. It

1. checks the Python environment (`.venv`) and creates it the first time;
2. checks every package and installs only what is missing, quietly, one line per package (PyTorch with CUDA shows its
   download progress — about 2–3 GB, once);
3. checks the project files: app files, reference tables, trained models in `models/`, datasets in `data/`;
4. verifies that every package imports, then opens the app in your browser.

The full check (GPU, PyTorch test, importing every package) runs only the first time and after a package changes; it
then remembers the result in `.venv/pathdspx_check.json`. Every later start only compares package versions with that
file (under a second) and opens the app. To force the full check again, run `start.bat full` in the project folder.
Keep its window open while you use the app; close it to stop. No venv activation or `Set-ExecutionPolicy` is needed. If Python itself is missing, it installs
Python 3.12 with `winget` and asks you to run it again.

- **First data load:** reading `input.txt` once takes about 10–60 s and writes a cache to `data/cache/`. Every later load takes under 1 s.
- **Optional, faster optimizer:** Train page → *Performance* → *Install Triton* installs the Triton build that matches your
  PyTorch, then *Test* checks the fused kernel on your GPU.

## Folder structure

```
PathDSP/
├─ app.py                 Streamlit entry: theme, navigation, live brand bar (GPU in use / idle)
├─ start.bat              one double-click: check / install what is missing, verify, open the app
├─ train.py               training worker (started by the app; also a CLI)
├─ core/                  backend, no UI code
│  ├─ config.py           paths, the 3 model definitions + defaults, datasets, paper reference values
│  ├─ data.py             loaders for input.txt / processed_data (GDSC, CCLE) / your own files + binary cache
│  ├─ nets.py             PathDSPNet, ProposedNet, batched (fold-parallel) form
│  ├─ engine.py           jobs/splits, GPU slot pool trainer, optimizer, VRAM sizing, time model
│  ├─ runner.py           one run: pool, live status, results files, SHAP, CCLE test, benchmark
│  ├─ metrics.py          RMSE / MAE / R² / PCC / Spearman / per-drug PCC, sensitivity class
│  ├─ explain.py          SHAP (DeepExplainer / GradientExplainer), pathway attention
│  ├─ uncertainty.py  compare.py   conformal intervals · paired fold tests (Phase 1)
│  ├─ pathways.py  graph.py        PID pathway graph + pathway-graph model (Phase 2)
│  ├─ csa.py              IMPROVE CSA benchmark: 5 studies, official splits, cross-study matrix (Phase 3),
│  │                      leave-study-out tables + few-shot recalibration (Phase 4)
│  ├─ molenc.py           learned drug graph encoder (Phase 4, upgrade M)
│  ├─ gbm.py              LightGBM model: training, prediction, TreeSHAP (v5.9)
│  ├─ blend.py            cross-fitted blend of two runs, e.g. a network + LightGBM (v5.9)
│  └─ store.py            run + model files, Predictor
├─ tools/                 setup_env.py (used by start.bat) · build_csa.py + csa_features.py (rebuild data/csa) ·
│                         final_plan.py + final_analysis.py (final validation plan)
├─ ui/                    frontend
│  ├─ theme.py  charts.py  common.py
│  └─ page_train.py  page_live.py  page_results.py  page_predict.py
├─ data/                  input.zip, processed_data_GDSC.zip, csa/, + any dataset you add   (not in git)
├─ refs/                  cell_info.tsv (cell line name / tissue / cancer type), drug_info.tsv (targets / pathway)
├─ models/                <run_id>.pt + .json — the models Predict can use               (not in git)
├─ runs/                  one folder per training run (config, status, folds, predictions, curves, SHAP)  (not in git)
├─ final/                 final validation plan: FINAL_SUMMARY.md + one folder per step
└─ .streamlit/config.toml Nothing-OS theme (dark base; the Light / Dark switch is in the header)
```

## The three models

All three use the PathDSP network: 1031 → 1000 → 800 → 500 → 100 → 1, ELU, dropout 0.1, RMSE loss, gradient clipping ±5, weight decay 1e-5, early stopping with patience 30 and the best epoch restored, at most 800 epochs, and an inner 10 % validation split. The defaults below are pre-filled on the Train page.

| | **Paper exact** | **Repo exact** | **Proposed system** |
|---|---|---|---|
| Source | Table S2 + Methods | GitHub `FNN.py` / `leaveOneGroupOut_FNN.py` | Milestone-1 architecture (slide 11) |
| Init | He-normal | Kaiming-uniform, bias 0.01 | He-normal |
| Optimizer | AdaMax, lr 0.0004 | Adam, lr 0.0004 | AdaMax, lr 0.0004 |
| Batch | 8 (see note 1) | 12 | 8 |
| Features | z-scored on each training fold | raw, as in the released code | z-scored |
| Interaction module | — | — | DG(p)·EXP/MUT/CNV(p) for 579 aligned pathway pairs → 128 → joined to the FNN |
| Response target | −log10 IC50 | −log10 IC50 | −log10 IC50 **or** drug-wise z-score |
| Mixed CV | 10-fold × 5 repeats | 10-fold × 1 | 10-fold × 5 |

The number of folds (k) can be changed on the Train page (paper and repo: 10).

**LightGBM (v5.9)** is the non-neural reference model: gradient-boosted trees (L2 regression) on exactly the same
1,031 features, folds, inner validation split and targets (raw, drug-wise z, study-wise z). Defaults: learning rate
0.05, 63 leaves, ≥ 20 pairs per leaf, 50 % of the features and 80 % of the pairs per tree, L2 1, at most 5,000 rounds,
early stopping after 100 rounds without a lower validation RMSE (best round kept). It trains on the CPU (all physical
cores, one model at a time); a CSA pair matrix is binned from row batches, so leave-study-out needs about 1 GB of RAM.
SHAP is exact TreeSHAP. Install it once: Train → Model → LightGBM → *Install LightGBM* (or `start.bat`, which
installs it). **Results → Compare runs → Blend** averages two runs' held-out predictions, w·A + (1 − w)·B, with each
fold's weight fitted on the other folds (cross-fitting), and tests the blend against both runs fold by fold.

1. Table S2 does not give a batch size. 8 comes from the header of the authors' `FNN.py`, which lists the paper's settings: He normal, AdaMax, lr 0.0004, RMSE.
2. Folds are the same for every model: rows shuffled with the seed, then `KFold(10, shuffle, seed)`, with the inner split `random_state=seed`, exactly as in `FNN.py`. LOCO and LODO are leave-one-out as in `leaveOneGroupOut_FNN.py`, or grouped 10-fold if you want them fast.
3. **Drug-wise z target:** each drug's mean and sd are fitted on the training fold only. Leave-drug-out is not possible with this target, because a held-out drug has no training responses.

## Data (`data/`)

- **`input.zip`:** the authors' `input.txt` (Zenodo 7532963, the GitHub example input). 37,732 pairs, 144 drugs, 319 cell lines, 1,031 features (CHEM 256 · DG 196 · EXP 187 · MUT 196 · CNV 196).
- **`processed_data_GDSC.zip`:** the authors' `processed_data` (Zenodo 4729069), covering GDSC (40,365 pairs, 153 drugs) and CCLE, which is used for the external test.
- **Response:** −log10 IC50, with IC50 in µM, for these two (the paper's unit).
- **`csa/` (Phase 3):** the IMPROVE cross-study benchmark — CCLE, CTRPv2, gCSI, GDSCv1, GDSCv2 (587,709 pairs, 749 drugs,
  785 cell lines) with PathDSP features built the authors' way for all five studies (`tools/build_csa.py`: Morgan
  radius 3 / 256 bits, NetPEA on STRING v12.5 for DG / MUT / CNV, ssGSEA for EXP, MSigDB PID gene sets). Response =
  AAC = 1 − AUC (higher = more sensitive). Each study appears in the Data list as *CSA · <study>*; the evaluation
  *CSA splits* uses the benchmark's own 10 splits, and the cross-study test applies every split model to the four
  other studies (Results → Cross-study: G, Gn, Ga, Gna; cross-study inputs are clipped to the feature ranges of the
  split's training data). Rebuilding needs `pip install rdkit gseapy`.
- **Multi-study training (Phase 4, PathDSP-X v2):** *CSA · all 5 studies* pools the five studies (each row keeps its
  study). Evaluation *Leave-study-out* trains on four studies and tests on all of the fifth. New options:
  target **S** (study-wise z: each study standardised on its training rows; an unseen study gets the average source
  scale), **R** (within-drug ranking loss on drug-chunked mini-batches), **H** (study-adversarial training: a study
  discriminator behind a gradient-reversal layer), and for the graph model a **random pathway graph** control
  (degree-preserving rewiring). Results → Leave-study-out compares each held-out study with the single-study models
  (G[s, t]) and shows few-shot recalibration (slope shrunk towards 1) with split-conformal 90 % intervals.
- **M · learned drug encoder (Phase 4):** a graph neural network over each drug's molecule (`data/csa/molecules.npz`,
  24,562 atoms of the 749 drugs; `core/molenc.py`) learns a 64-number drug embedding end-to-end; any of the four
  networks gets it as 64 extra inputs (`MOL_` block in SHAP). CSA datasets only. Held-out-study inputs are clipped to
  the training range (as in the cross-study test). Each step encodes only the drugs in its mini-batch (identical
  embeddings, several times faster and lighter than encoding all 749).
- **Your own data:** upload a file on the Train page (*Add a dataset*) or copy it into `data/`; it appears in the Data
  list within a few seconds. Accepted: a table (.txt / .tsv / .csv, or a .zip holding one) in the layout of `input.txt` —
  columns `drug`, `cell`, feature columns prefixed `CHEM_`, `DG_` (or `TARGETnNEIGHBOR_`), `EXP_`, `MUT_`, `CNV_`, and a
  response column `resp` (−log10 IC50; `LN_IC50` or `IC50` in µM are converted) — or another processed_data zip.

## Pages

- **Train:** model → data (built-in + your files, upload) → evaluation → settings (epochs, patience, batch, learning
  rate, **folds k**, repeats, seed) → **performance** (models at once, *Max performance*, fused Triton optimizer with
  install + test) → *Estimate time* (measures the real step time on your GPU) → *Start*. Training runs in its own
  process, so closing the browser does not stop it.
- **Live:** refreshes every second:
  - elapsed, time left (expected + fastest / worst-case range), models done (and how many wait for a GPU slot), epochs,
    models on the GPU / slots, samples per second, ms per step, GPU memory held by the run;
  - progress of the current epoch, the mean loss curve over all models, results so far;
  - **status of every model on the GPU after each epoch**: epoch, train / val loss, best val loss and epoch, a bar of
    *epochs without improvement* (it stops at the patience), and its status;
  - every finished model with its test RMSE / PCC, plus Stop / Resume.
- **Results:** refreshes every 2 s while a run trains. Has these tabs:
  - one per evaluation (Mixed / Leave-cell-out / Leave-drug-out): RMSE, MAE, R², PCC, Spearman and per-drug PCC, each
    with the paper's value; the drug-mean baseline; predicted-vs-measured; training curves; per-fold RMSE; time per model;
  - **SHAP · pathways:** importance by block and the top pathways;
  - **CCLE external** (processed_data runs only);
  - **Compare runs.**
- **Predict:** choose a model, a drug and a cell line ("NAME — cancer type [SIDM]"). Each list has a **search box**
  above it: drugs by name, target, pathway or id (e.g. `BRAF`, `apoptosis`); cell lines by name, cancer type, tissue
  or id (e.g. `melanoma`, `HCT-116`, `SIDM00783`). Every word typed must match. The presentation models are listed first
  and named by role: **MODEL A**, **BASE PAPER (rerun)**, **MODEL A · F1 final**, **MODEL B · Run A / Run B**. Shows the predicted −log10 IC50 and
  IC50 in µM, the measured value, the absolute error (from the held-out cross-validation prediction), the diagnosis
  (where the prediction sits among all cell lines screened with that drug) and a chart of that drug's responses.
- **Top bar (every page):** the **Light / Dark** switch (remembered in the address, `?theme=light`), GPU name, memory and load, and below it whether the GPU is **in use** (training) or **idle**.

## Speed

- **GPU slot pool:** every fold-model of a run — k folds × repeats, every leave-one-out fold, and the all-data model —
  trains in a pool of GPU slots, as many at once as fit in free VRAM, as one batched network. When a model early-stops,
  its slot is refilled with the next waiting model at the following epoch, so the GPU stays full until the queue is
  empty. Each model keeps its own data, scaling, early stopping and optimizer state (its own step counter); the maths
  is identical to training it alone (checked to 1e-16 in float64).
- **Max performance** (default on): 95 % of the free VRAM for fold-models and high CPU priority for the training process.
  If the GPU runs out of memory, the number of slots halves automatically.
- **Fused optimizer (optional):** at batch 8–12 the optimizer update is the bottleneck. With Triton installed (Train page →
  *Install Triton*), clipping, weight decay and the AdaMax/Adam update run as one GPU kernel. The app checks the fused
  kernel against the standard update at start-up and uses it only if they match; otherwise it uses the standard ops.
- **Resume:** *Stop* keeps every finished model, and *Resume* continues with the rest.

## Metrics

- **RMSE, MAE:**
  - Mixed: mean ± sd between the 5 repeated runs, as in the paper, or ± over the k folds when there is one run.
  - Leave-one-out: mean ± sd over the held-out cell lines or drugs (the paper's LOCO/LODO numbers).
- **R², PCC, Spearman:** computed on all held-out predictions pooled.
- **Leave-study-out (Phase 4):** every metric is the mean ± sd over the 5 held-out studies; each held-out study is
  scored on all of its pairs. `fewshot.csv`: k = 10 … 1,000 labelled pairs of the new study recalibrate the scale
  (y ≈ a·ŷ + b, 20 draws), scored on the rest.
- **CSA splits:** every metric is the mean ± sd over the 10 official splits; the cross-study matrix G[s, t] is the
  mean over the 10 split models trained on s and tested on all of t (within-study: each split's own test set).
- **Per-drug PCC:** the mean, over drugs, of the correlation between measured and predicted response across that drug's cell lines. This is the cell-specific signal that drug potency alone cannot give; the drug-mean baseline scores 0 on it.

## Command line

```bat
python train.py --model paper --data input --schemes mixed
python train.py --model repo --data processed --schemes mixed LOCO LODO --folds 10
python train.py --model proposed --data input --schemes mixed --target drug_z
python train.py --resume <run_id>
python train.py --benchmark --model paper --data input
python train.py --check-fused
python train.py --model graph --data csa-CTRPv2 --schemes csa --cross
python train.py --model graph --data csa-ALL --schemes loso --target study_z --rank --adv
python train.py --model graph --data input --schemes mixed --random-graph
python train.py --model lgbm --data input --schemes mixed --target drug_z
python train.py --model lgbm --data csa-ALL --schemes loso --no-final
python tools/final_plan.py            # the final validation plan (v6.0)
python tools/final_plan.py --plan     # its settings, without training
```

## Final validation plan (v6.0)

Double-click **`final_plan.bat`** (or run `python tools/final_plan.py`). It runs every remaining step in order and
can be stopped and started again at any time; it continues where it stopped.

| # | Step | What it does | Rough time (RTX 4060) |
|---|---|---|---|
| 1 | Recipe on fresh folds | Paper exact vs Paper exact + C·D×3·E·F·G·Z on input.txt, mixed 10-fold, **seed 43** (folds that played no part in choosing the recipe); paired per-fold tests | ≈ 3 h |
| 2 | Leakage audit | checks on step 1's files (split integrity, per-fold scaling, drug-wise z statistics, conformal calibration) + four leak demonstrations, clean vs leaky on the same folds (Ridge; LightGBM for test-based early stopping) | ≈ 0.5 h (CPU) |
| 3 | Explanation check | SHAP of step 1's recipe model on its best fold: do the top cell-line pathways contain the drug's annotated targets (AUROC, Precision@K, Hit@K, permutation control)? Deletion test and seed-to-seed stability | ≈ 10–30 min |
| 4 | Unseen cells / drugs | grouped 10-fold LOCO and LODO: Paper exact vs the recipe (LODO without the z target), with drug-mean / cell-mean / global-mean baselines | ≈ 5–7 h |
| 5 | Finalise | the recipe is kept for the final IC50 model only if step 1 confirms it (better on ≥ 8 of 10 folds, Wilcoxon p < 0.05) | seconds |
| 6 | Train-on-everything | IC50: all of processed_data · GDSC (with the CCLE external test); AAC: all 587,709 CSA pairs with the Run A recipe and with the Run B recipe. These are the models Predict uses | ≈ 3–5 h |
| 7 | Summary | `final/FINAL_SUMMARY.md` and **`final/final_results.zip`** (reports, tables, run files — no model weights) | seconds |

- `final_plan.bat --plan` shows every run's exact settings without training anything.
- The training runs are ordinary app runs: open the app (`start.bat`) to watch them on Live and Results.
- Do not start other trainings from the app while the plan runs. Live → Stop pauses the current run; start
  `final_plan.bat` again to continue it.

## Results, models and data in git

Datasets (`data/`), training runs (`runs/`) and trained models (`models/`) are large, so they are **not pushed** (see
`.gitignore`); share them separately (Drive or a GitHub release). The final validation results are included in
`final/` (`FINAL_SUMMARY.md` and one folder per step).

## Demo models for the presentation

| Show | Model in Predict | Why |
|---|---|---|
| Base paper | `BASE PAPER (rerun)` · `20261109-175325` | the authors' FNN, rerun by us on input.txt |
| Model A | `MODEL A` · `20261109-184612` (1-recipe) | same FNN + recipe + drug-wise z; has held-out predictions and 90 % intervals |
| Model A final | `MODEL A · F1 final` · `20261110-013147` | trained on all of processed_data · GDSC (no held-out error or interval: it saw every pair) |
| Model B | `MODEL B · Run A · leave-one-study-out` · `20261108-082640` | predicts each study from the other four; held-out AAC + interval |
| Model B final | `F2 final` / `F3 final` | trained on all 5 studies (Run A raw AAC / Run B ranking) |

Example pairs (Model A vs Base paper, held-out = the fair prediction, from the *Absolute error* tile):

| Drug (target) | Cell line | Biology | Measured | Model A held-out | Base held-out | Diagnosis |
|---|---|---|---|---|---|---|
| Nutlin-3a (MDM2) | HCT-116 (colon) | TP53 wild-type → drug works | −0.79 | −1.28 (error 0.48) | −1.68 (0.89) | Sensitive (87 %) |
| Nutlin-3a (MDM2) | SW620 (colon) | TP53 mutant → drug fails | −2.93 | −2.24 (0.69) | −2.49 (0.43) | Resistant (6 %) |
| Dabrafenib (BRAF) | HT-29 (colon) | BRAF V600E | −0.08 | −0.63 (0.56) | −0.81 (0.73) | Sensitive (99 %) |
| Lapatinib (EGFR/HER2) | UACC-812 (breast) | HER2 amplified | −0.44 | −0.78 (0.34) | −1.03 (0.59) | Sensitive (92 %) |
