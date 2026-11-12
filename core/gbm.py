"""
LightGBM — gradient-boosted decision trees on exactly the PathDSP features (v5.9).

The non-neural reference model. It uses the same feature matrix (CHEM, DG, EXP, MUT, CNV), the same folds and inner
validation split, the same response targets (raw / drug-wise z / study-wise z, converted back before any metric) and
writes the same files as the networks, so every page, the paired test and the blend work on it unchanged.

  trees     L2 regression, early stopping on the inner validation RMSE (best iteration restored)
  memory    a CSA pair matrix is never materialised: LightGBM reads its rows in batches (lightgbm.Sequence) and keeps
            only the binned data (column-wise, ~0.6 GB for 520k training pairs) instead of the 2.1 GB float matrix;
            the binned data is freed as soon as the model is trained
  SHAP      exact TreeSHAP (LightGBM pred_contrib) — no background sample needed
  CPU       one model at a time on all physical CPU cores (the GPU is not used)
"""
import importlib
import importlib.util
import os
import time

import numpy as np

BATCH = 16384                     # rows per batch when LightGBM reads a pair matrix
SAMPLE = 100_000                  # rows sampled to place the histogram bins (LightGBM default 200k; memory)
PRED_CHUNK = 65536                # rows per prediction call
CURVE_POINTS = 400                # curves.csv keeps at most about this many iterations per model


def available():
    importlib.invalidate_caches()
    return importlib.util.find_spec("lightgbm") is not None


def version():
    try:
        import importlib.metadata as md
        return md.version("lightgbm")
    except Exception:
        return None


def threads():
    """physical CPU cores (LightGBM is slower on hyper-threads), at least 1"""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def params(spec, seed, n_threads=None):
    sub = float(spec.get("gb_subsample", 0.8))
    return {"objective": "regression", "metric": "rmse", "learning_rate": float(spec["lr"]),
            "num_leaves": int(spec.get("gb_leaves", 63)), "min_data_in_leaf": int(spec.get("gb_min_leaf", 20)),
            "feature_fraction": float(spec.get("gb_colsample", 0.5)), "bagging_fraction": sub,
            "bagging_freq": 1 if sub < 1.0 else 0, "lambda_l2": float(spec.get("gb_l2", 1.0)), "max_bin": 255,
            "bin_construct_sample_cnt": SAMPLE, "seed": int(seed), "deterministic": True, "force_col_wise": True,
            "feature_pre_filter": False, "num_threads": int(n_threads or threads()), "verbosity": -1}


def _sequence_class():
    import lightgbm as lgb

    class PairRows(lgb.Sequence):
        """rows `idx` of a data.PairX, read in batches (drug block + cell block assembled per batch)"""
        batch_size = BATCH

        def __init__(self, X, idx):
            self.X, self.idx = X, np.asarray(idx, dtype=np.int64)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            # single rows feed LightGBM's bin sampling, which needs float64; batches are pushed as float32
            if isinstance(i, (int, np.integer)):
                return np.asarray(self.X[self.idx[int(i)]], dtype=np.float64)
            if isinstance(i, slice):
                return np.ascontiguousarray(self.X[self.idx[i]], dtype=np.float32)
            return np.ascontiguousarray(self.X[self.idx[np.asarray(i, dtype=np.int64)]], dtype=np.float64)

    return PairRows


def _data(X, rows):
    """what lgb.Dataset gets for these rows: the dense sub-matrix (in-memory datasets) or a Sequence (pair matrix)"""
    if hasattr(X, "stats"):                                  # data.PairX
        return _sequence_class()(X, rows)
    return np.ascontiguousarray(X[np.asarray(rows, dtype=np.int64)], dtype=np.float32)


def rows_dense(X, rows):
    """dense float32 rows of X (any number, in chunks for a PairX)"""
    rows = np.asarray(rows, dtype=np.int64)
    if not hasattr(X, "stats"):
        return np.ascontiguousarray(X[rows], dtype=np.float32)
    out = np.empty((len(rows), X.shape[1]), dtype=np.float32)
    for i in range(0, len(rows), BATCH):
        out[i:i + BATCH] = X[rows[i:i + BATCH]]
    return out


def predict_rows(booster, X, rows, best_iter):
    rows = np.asarray(rows, dtype=np.int64)
    out = [booster.predict(rows_dense(X, rows[i:i + PRED_CHUNK]), num_iteration=best_iter)
           for i in range(0, len(rows), PRED_CHUNK)]
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def predict_pairs(booster, Dm, Cm, di, ci, best_iter):
    """predictions for arbitrary (drug index, cell index) pairs of the drug / cell feature tables"""
    di, ci = np.asarray(di, dtype=np.int64), np.asarray(ci, dtype=np.int64)
    out = []
    for i in range(0, len(di), PRED_CHUNK):
        xb = np.concatenate([Dm[di[i:i + PRED_CHUNK]], Cm[ci[i:i + PRED_CHUNK]]], axis=1).astype(np.float32)
        out.append(booster.predict(xb, num_iteration=best_iter))
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def thin(hist_train, hist_val, best_it, n_max=CURVE_POINTS):
    """iteration numbers kept for curves.csv: every k-th, the best and the last"""
    n = len(hist_val)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = max(1, int(np.ceil(n / n_max)))
    keep = set(range(k - 1, n, k)) | {n - 1}
    if 1 <= best_it <= n:
        keep.add(best_it - 1)
    return np.array(sorted(keep), dtype=np.int64)


def train(X, target, job, spec, seed, on_iter=None, stop_check=None, n_threads=None):
    """one model: trains on job.tr_in, early-stops on job.va -> dict(booster, best_iter, hist, seconds, iters).
    on_iter(it, train_rmse, val_rmse, best_it, best_val) is called after every boosting round; stop_check() -> True
    raises StopTraining."""
    import lightgbm as lgb
    t0 = time.perf_counter()
    prm = params(spec, seed, n_threads)
    y = np.asarray(target, dtype=np.float64)
    dtr = lgb.Dataset(_data(X, job.tr_in), label=y[job.tr_in], params=prm, free_raw_data=True)
    dva = lgb.Dataset(_data(X, job.va), label=y[job.va], reference=dtr, params=prm, free_raw_data=True)
    hist = {"train": [], "val": []}
    state = {"best": float("inf"), "best_it": 0}
    patience = max(1, int(spec["patience"]))

    def cb(env):
        res = {(e[0], e[1]): e[2] for e in env.evaluation_result_list}
        tr_, va_ = float(res[("train", "rmse")]), float(res[("val", "rmse")])
        hist["train"].append(tr_)
        hist["val"].append(va_)
        it = env.iteration + 1
        if va_ < state["best"]:
            state["best"], state["best_it"] = va_, it
        if on_iter:
            on_iter(it, tr_, va_, state["best_it"], state["best"])
        if stop_check and stop_check():
            raise StopTraining()
        if it - state["best_it"] >= patience:                # the same rule as the networks' early stopping
            raise lgb.callback.EarlyStopException(state["best_it"] - 1, [])
    cb.order = 30
    booster = lgb.train(prm, dtr, num_boost_round=max(1, int(spec["epochs"])), valid_sets=[dtr, dva],
                        valid_names=["train", "val"], callbacks=[cb])
    best_it = int(state["best_it"]) or booster.current_iteration()
    booster.best_iteration = best_it
    booster.free_dataset()                                   # the binned training / validation data is not needed
    del dtr, dva
    return {"booster": booster, "best_iter": best_it, "iters": len(hist["val"]), "hist": hist,
            "best_val": float(state["best"]), "seconds": time.perf_counter() - t0}


class StopTraining(Exception):
    """the user pressed Stop during a LightGBM model"""


def contributions(booster, Xs, best_iter):
    """exact TreeSHAP values [n, F] (bias column dropped), in model output units"""
    c = booster.predict(np.ascontiguousarray(Xs, dtype=np.float32), num_iteration=best_iter, pred_contrib=True)
    return np.asarray(c)[:, :-1]


def load_booster(model_str):
    import lightgbm as lgb
    return lgb.Booster(model_str=model_str)


def benchmark_seconds(X, job, spec, seed, rounds=30):
    """measured seconds per boosting round on a job's training rows (the first rounds of a real model)"""
    t = {}

    def on_iter(it, *_):
        if it == 3:
            t["t3"] = time.perf_counter()
        if it == rounds:
            t["tN"] = time.perf_counter()
    s2 = dict(spec, epochs=rounds, patience=rounds + 1)
    t0 = time.perf_counter()
    r = train(X, np.zeros(len(X), dtype=np.float32) + np.random.RandomState(0).rand(len(X)).astype(np.float32),
              job, s2, seed, on_iter)
    per = (t["tN"] - t["t3"]) / max(1, rounds - 3) if "tN" in t and "t3" in t else r["seconds"] / max(1, r["iters"])
    build = max(0.0, (t.get("t3", t0) - t0) - 3 * per)       # binning the data before the first round
    return per, build
