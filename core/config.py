"""
Project paths, the three model definitions (with their default settings) and the dataset registry.
Everything else in the project reads its constants from here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"            # input.zip, processed_data_GDSC.zip (+ cache/ written automatically)
CACHE = DATA / "cache"
REFS = ROOT / "refs"            # cell_info.tsv, drug_info.tsv
MODELS = ROOT / "models"        # <run_id>.pt + <run_id>.json  (used by Predict)
RUNS = ROOT / "runs"            # one folder per training run
for _p in (DATA, CACHE, MODELS, RUNS):
    _p.mkdir(parents=True, exist_ok=True)

LN10 = 2.302585092994046
APP_NAME = "PathDSP"
APP_TITLE = "Explainable drug sensitivity prediction through cancer pathway enrichment"
UNIT = "−log10 IC50"            # response unit of the GDSC datasets (IC50 in µM); the CSA studies use AAC (UNITS)
UNITS = {
    "ic50": {"key": "ic50", "name": "−log10 IC50", "long": "−log10 IC50 (µM)", "ic50": True},
    "aac": {"key": "aac", "name": "AAC", "long": "AAC = 1 − AUC of the dose–response fit (0–1)", "ic50": False},
}

# ----------------------------------------------------------------------------------------------------------------------
# The three models. All are the PathDSP fully-connected network (1000-800-500-100-1, ELU, dropout 0.1, RMSE loss,
# gradient-value clipping ±5, weight decay 1e-5, early stopping patience 30 with best-epoch restore, max 800 epochs,
# inner 10 % validation split) — they differ only in what is listed per model.
# ----------------------------------------------------------------------------------------------------------------------
_COMMON = dict(epochs=800, patience=30, lr=4e-4, weight_decay=1e-5, clip=5.0, dropout=0.1, val_frac=0.1,
               fac=False, fac_dim=16, gate=False, gate_init=0.1, plateau=False, plateau_patience=5, plateau_delta=1e-3, ensemble=1,
               ema=False, loss="rmse", val_metric="batch",
               rank=False, rank_weight=0.5, rank_chunk=8, adv=False, adv_weight=0.1, adv_ramp=10, mol=False)
# upgrades A–H and R, all off by default

MODEL_SPECS = {
    "paper": {
        "name": "Paper exact",
        "tag": "PAPER",
        "one_line": "PathDSP as described in the paper (Tang & Gottlieb 2021, Table S2 + Methods).",
        "defaults": {**_COMMON, "init": "he_normal", "optimizer": "adamax", "batch": 8, "feat_scale": True,
                     "interaction": False, "target": "raw", "repeats": 5},
        "notes": [
            "He-normal init · AdaMax · lr 0.0004 · dropout 0.1 · early stopping 30 · clip 5 · ELU · 800 epochs (Table S2)",
            "features z-scored on each training fold; 10-fold CV repeated 5× with different splits (Methods)",
            "batch 8 — not in Table S2; taken from the header of the authors' FNN.py, which lists the paper's "
            "settings (He normal, AdaMax, lr 0.0004, RMSE)",
        ],
    },
    "repo": {
        "name": "Repo exact",
        "tag": "REPO",
        "one_line": "PathDSP exactly as the released code trains it (GitHub FNN.py / leaveOneGroupOut_FNN.py).",
        "defaults": {**_COMMON, "init": "he_uniform", "optimizer": "adam", "batch": 12, "feat_scale": False,
                     "interaction": False, "target": "raw", "repeats": 1},
        "notes": [
            "Kaiming-uniform init (bias 0.01) · Adam · lr 0.0004 · batch 12 · weight decay 1e-5 · clip 5",
            "features fed as they are (the released code does not z-score)",
            "one 10-fold run: rows shuffled with seed 42, KFold(shuffle, 42), inner split random_state 42",
        ],
    },
    "proposed": {
        "name": "Proposed system",
        "tag": "PROPOSED",
        "one_line": "PathDSP FNN + drug-pathway × cell-pathway interaction module; raw or drug-wise z-scored target.",
        "defaults": {**_COMMON, "init": "he_normal", "optimizer": "adamax", "batch": 8, "feat_scale": True,
                     "interaction": True, "d_int": 128, "target": "raw", "repeats": 5},
        "notes": [
            "interaction module: for every PID pathway p, DG(p)·EXP(p), DG(p)·MUT(p), DG(p)·CNV(p) "
            "(579 pairs) → Linear 128 → ELU → dropout, joined to the 1,031 PathDSP features before the FNN",
            "response target compared: raw −log10 IC50 or drug-wise z-score (fitted on each training fold)",
            "same training recipe and protocol as Paper exact, so the two are directly comparable",
        ],
    },
}
MODEL_SPECS["graph"] = {
    "name": "Pathway graph",
    "tag": "GRAPH",
    "one_line": "PathDSP-X: the 196 PID pathways as a graph, a drug-conditioned attention over them, and the best "
                "training recipe.",
    "defaults": {**_COMMON, "init": "he_normal", "optimizer": "adamax", "batch": 8, "feat_scale": True,
                 "interaction": False, "target": "drug_z", "repeats": 1, "arch": "graph", "g_mode": "hybrid",
                 "g_dim": 32, "g_layers": 2, "g_tau": 0.05, "plateau": True, "plateau_patience": 5,
                 "plateau_delta": 1e-3, "ema": True, "loss": "mse", "val_metric": "global"},
    "notes": [
        "nodes: every PID pathway with its 4 scores for the pair (DG, EXP, MUT, CNV) + a learned pathway embedding; "
        "edges: pathways sharing genes (Jaccard \u2265 0.05, MSigDB C2 CP:PID gene sets), 2 rounds of message passing",
        "the drug's fingerprint (CHEM) forms a query that weighs the pathways (attention) — the weights are saved as "
        "the model's pathway explanation",
        "readout: hybrid = the PathDSP network on [features ; graph summary] (default), pure = a small head on the "
        "graph summary only · defaults = the best recipe (drug-wise z target, C, E, F, G)",
    ],
}
GBM_MODEL = "lgbm"
MODEL_SPECS[GBM_MODEL] = {
    "name": "LightGBM",
    "tag": "LGBM",
    "one_line": "Gradient-boosted decision trees on exactly the PathDSP features — the non-neural reference model.",
    "defaults": {**_COMMON, "epochs": 5000, "patience": 100, "lr": 0.05, "batch": 1, "dropout": 0.0, "weight_decay": 0.0,
                 "init": "none", "optimizer": "lightgbm", "feat_scale": False, "interaction": False, "target": "raw",
                 "repeats": 1, "arch": GBM_MODEL, "gb_leaves": 63, "gb_min_leaf": 20, "gb_colsample": 0.5,
                 "gb_subsample": 0.8, "gb_l2": 1.0},
    "notes": [
        "LightGBM L2 regression on the same 1,031 features, folds, inner 10 % validation split and response targets "
        "as the networks; early stopping on the validation RMSE with the best iteration restored",
        "63 leaves per tree · learning rate 0.05 · 50 % of the features and 80 % of the pairs per tree · L2 1 · "
        "255 histogram bins · deterministic",
        "trained on the CPU (all physical cores, one model at a time) · exact TreeSHAP explanations · combine it with "
        "a network under Results → Compare runs → Blend",
    ],
}
GB_LEAVES = (15, 31, 63, 127, 255)
GB_MIN_LEAF = (5, 10, 20, 50, 100)
GB_COLSAMPLE = (0.2, 0.3, 0.5, 0.7, 1.0)
GB_SUBSAMPLE = (0.5, 0.7, 0.8, 1.0)
GB_L2 = (0.0, 0.1, 1.0, 10.0)
MODEL_KEYS = list(MODEL_SPECS)
GRAPH_MODES = ("hybrid", "pure")
GRAPH_DIMS = (16, 32, 64)
GRAPH_LAYERS = (1, 2, 3)
GRAPH_TAUS = (0.02, 0.05, 0.1)

# Settings the Train page exposes (everything else is fixed by the model definition)
EDITABLE = ("epochs", "patience", "batch", "lr", "repeats")

# ----------------------------------------------------------------------------------------------------------------------
# Optional upgrades (Train page → Settings). A and B change the Proposed network; C and D change the training recipe
# and can be used with every model (a Paper / Repo exact run with C or D is then no longer "exact" — the run says so).
# ----------------------------------------------------------------------------------------------------------------------
UPGRADES = {
    "fac": {"letter": "A", "name": "Factorization", "models": ("proposed",),
            "help": "Adds a drug tower (CHEM + DG → 256 → rank) and a cell tower (EXP / MUT / CNV → 256 → rank); their "
                    "dot product ÷ rank is added to the network's output. Models every drug × cell interaction, not "
                    "only the pathway-matched ones. A low rank (8–16) cannot memorise the 144 × 319 drug × cell "
                    "matrix; 64 = the first version."},
    "gate": {"letter": "B", "name": "Gated interaction", "models": ("proposed",),
             "help": "Multiplies the interaction module's 128 outputs by learnable gates that start small (0.1), so "
                     "the interaction path begins almost closed and opens only as far as it helps (a start at exactly "
                     "0 — the first version — leaves its weights without a gradient and lets weight decay shrink "
                     "them while the gate opens). Needs the interaction module."},
    "plateau": {"letter": "C", "name": "LR on plateau", "models": ("paper", "repo", "proposed", "graph"),
                "help": "Halves a model's learning rate when its validation loss has not dropped by at least "
                        "'min. improvement' for 'LR patience' epochs (never below 1/64 of the set rate). Early "
                        "stopping (patience above, any improvement counts) and best-epoch restore are unchanged."},
    "ensemble": {"letter": "D", "name": "Seed ensemble", "models": ("paper", "repo", "proposed", "graph"),
                 "help": "Every fold is trained by several models with different initial weights, batch order and "
                         "dropout masks; their predictions are averaged. Results also show the single-seed scores, "
                         "so one run gives both. Training time × seeds."},
    "ema": {"letter": "E", "name": "EMA weights", "models": ("paper", "repo", "proposed", "graph"),
            "help": "Keeps a moving average of each model's weights over about the last epoch of optimizer steps; "
                    "validation, early stopping, the saved model and the test predictions use the averaged "
                    "weights. Smooths out the epoch-to-epoch jitter of small-batch training. About 5–10 % slower."},
}
UPGRADES.update({
    "loss": {"letter": "F", "name": "MSE loss", "models": ("paper", "repo", "proposed", "graph"),
             "help": "The optimizer minimises the mini-batch MSE instead of the authors' mini-batch RMSE. sqrt(MSE) "
                     "scales every mini-batch's gradient by 1 / its own RMSE, so batches the model already fits well "
                     "get the largest steps; the MSE gradient weights every pair equally — the unbiased direction for "
                     "the RMSE the results report. Train loss is still shown as RMSE."},
    "val_metric": {"letter": "G", "name": "Global val RMSE", "models": ("paper", "repo", "proposed", "graph"),
                   "help": "Early stopping and the best epoch use the RMSE over the whole validation set (the metric "
                           "the results report) instead of the authors' mean of per-mini-batch RMSEs, which is "
                           "biased low. Validation numbers on Live are then a little higher — that is the bias "
                           "removed, not a worse model."},
})
UPGRADES.update({
    "rank": {"letter": "R", "name": "Within-drug ranking", "models": ("paper", "repo", "proposed", "graph"),
             "help": "Adds a ranking loss: for every two cell lines of the same drug (and study) in a mini-batch, the "
                     "more sensitive one must get the higher prediction (RankNet, softplus of the scaled difference). "
                     "Mini-batches are built from runs of 8 rows of one drug so such pairs exist. Targets what "
                     "matters for choosing cell lines / patients for a drug — the per-drug ranking — and ignores "
                     "a study's overall AAC scale. Loss = training loss + weight × target-scale × ranking loss "
                     "(target-scale = sd² of the training target for MSE, sd for RMSE: the weight means the same "
                     "for a raw and a z-scored target). Best together with target S."},
    "adv": {"letter": "H", "name": "Study-adversarial", "models": ("paper", "repo", "proposed", "graph"),
            "help": "Multi-study training only (CSA · all 5 studies). A small discriminator tries to tell from the "
                    "network's last hidden layer which study a pair came from; a gradient-reversal layer makes the "
                    "network learn features that hide the study (domain-adversarial training, DANN), so what it "
                    "learns transfers to a study it never saw. Its weight ramps up over the first 10 epochs."},
})
UPGRADES["mol"] = {
    "letter": "M", "name": "Drug graph encoder", "models": ("paper", "repo", "proposed", "graph"),
    "help": "CSA datasets. A small graph neural network reads each drug's molecule (atoms + bonds, 3 rounds of message "
            "passing) and learns a 64-number drug embedding, trained end-to-end with the rest; the network gets it as "
            "64 extra inputs (the pathway-graph model also in its drug query). Unlike the fixed fingerprint, it can "
            "learn which chemistry matters — aimed at drugs the model never saw. Slower per step."}
RANK_WEIGHTS = (0.1, 0.25, 0.5, 1.0, 2.0)
ADV_WEIGHTS = (0.01, 0.03, 0.1, 0.3, 1.0)
GRAPH_KINDS = ("pid", "random")
FAC_RANKS = (8, 16, 32, 64)
DROPOUTS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
WEIGHT_DECAYS = (0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
MAX_ENSEMBLE = 5


def upgrades_tag(spec):
    """'A·B·C·D×3' style short tag of the upgrades a spec uses ('' = none)"""
    out = []
    if spec.get("fac"):
        out.append(f"A{int(spec.get('fac_dim', 64))}")
    if spec.get("gate") and spec.get("interaction"):
        out.append("B")
    if spec.get("plateau"):
        out.append("C")
    n = int(spec.get("ensemble", 1) or 1)
    if n > 1:
        out.append(f"D×{n}")
    if spec.get("ema"):
        out.append("E")
    if spec.get("loss", "rmse") == "mse":
        out.append("F")
    if spec.get("val_metric", "batch") == "global":
        out.append("G")
    if spec.get("adv"):
        out.append("H")
    if spec.get("mol"):
        out.append("M")
    if spec.get("rank"):
        out.append("R")
    if spec.get("target") == "drug_z":
        out.append("Z")
    if spec.get("target") == "study_z":
        out.append("S")
    return "·".join(out)

# ----------------------------------------------------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------------------------------------------------
DATASETS = {
    "input": {
        "name": "input.txt",
        "label": "Authors' input.txt — Zenodo 7532963 (GitHub example input)",
        "archive": "input.zip",
        "loose": "input.txt",
    },
    "processed": {
        "name": "processed_data · GDSC",
        "label": "Authors' processed_data — GDSC (Zenodo 4729069)",
        "archive": "processed_data_GDSC.zip",
        "loose": "processed_data",
        "external": "ccle",
    },
}
DATA_KEYS = list(DATASETS)

SCHEMES = {
    "mixed": {"name": "Mixed", "long": "10-fold CV on random drug–cell pairs", "group": None},
    "LOCO": {"name": "Leave-cell-out", "long": "unseen cell lines", "group": "cells"},
    "LODO": {"name": "Leave-drug-out", "long": "unseen drugs", "group": "drugs"},
    "csa": {"name": "CSA splits", "long": "the benchmark's 10 official train / val / test splits", "group": None},
    "loso": {"name": "Leave-study-out", "long": "train on four studies, test on all of the unseen fifth",
             "group": "study"},
}
LOSO_SCHEME = "loso"              # only for the pooled CSA dataset (csa-ALL)
CSA_SCHEME = "csa"                # only for the IMPROVE CSA studies (core/csa.py)
N_FOLDS = 10

# paper results (Tang & Gottlieb 2021, −log10 IC50) shown next to ours
PAPER_REF = {
    "mixed": {"RMSE": (0.35, 0.02), "MAE": (0.24, 0.02), "R2": (0.92, None), "PCC": (0.96, None)},
    "LOCO": {"RMSE": (0.59, 0.17), "MAE": (0.45, 0.15)},
    "LODO": {"RMSE": (0.98, 0.62), "MAE": (0.83, 0.58)},
    "ccle": {"RMSE": (1.15, None), "MAE": (0.93, None)},
}

STOP_FILE = "STOP"
