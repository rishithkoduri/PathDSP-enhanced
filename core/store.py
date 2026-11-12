"""
Files of runs and models.

runs/<run_id>/
    config.json      settings of the run (written before it starts)
    status.json      live state, rewritten about once per second by the training process
    folds.csv        one row per finished fold-model (its test metrics, epochs, wall time)
    preds.csv        held-out predictions of every fold (drug, cell, measured, predicted, drug-mean baseline)
    curves.csv       train / validation loss per epoch of every model (seed 1 of a seed ensemble)
    members.csv      seed ensemble only: test metrics of every seed on its own
    cross.csv        CSA studies: every official-split model's metrics on the four other studies (cross-study)
    summary.json     final metrics + timing (written when the run ends)
    shap_*.csv       SHAP feature / block / pathway importance
    external_ccle.json   external test on CCLE (processed_data runs)
    log.txt          console output of the training process
models/<run_id>.pt + .json   the model used by Predict (all-data model, or the best fold if none was trained)
"""
import datetime as _dt
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import data as D
from . import nets

FOLD_COLS = ["scheme", "repeat", "fold", "group", "n_test", "RMSE", "MAE", "R2", "PCC", "SCC", "best_epoch",
             "epochs_run", "seconds", "parallel", "group_index"]
MEMBER_COLS = ["scheme", "repeat", "fold", "member", "RMSE", "MAE", "R2", "PCC", "SCC", "best_epoch", "epochs_run",
               "seconds"]
from . import csa as _csa                                      # noqa: E402
CROSS_COLS = list(_csa.CROSS_COLS)                             # cross.csv columns


# ======================================================================================================================
# runs
# ======================================================================================================================
def new_run_id(model, data):
    return f"{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{C.MODEL_SPECS[model]['tag']}_{data}"


def run_dir(rid):
    return C.RUNS / rid


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_config(rid, cfg):
    os.makedirs(run_dir(rid), exist_ok=True)
    D.write_json(run_dir(rid) / "config.json", cfg)


def config(rid):
    return read_json(run_dir(rid) / "config.json", {})


def status(rid):
    return read_json(run_dir(rid) / "status.json", {}) or {}


def summary(rid):
    return read_json(run_dir(rid) / "summary.json", {}) or {}


def pid_alive(pid):
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).status() != psutil.STATUS_ZOMBIE
    except Exception:
        try:
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False


def live_state(st):
    """status.json state, corrected for a training process that died without saying so"""
    s = st.get("state", "")
    if s in ("starting", "loading", "running", "explaining"):
        if not pid_alive(st.get("pid")) and time.time() - float(st.get("updated", 0)) > 15:
            return "interrupted"
    return s or "unknown"


def list_runs():
    out = []
    if not C.RUNS.exists():
        return out
    for d in sorted(C.RUNS.iterdir(), reverse=True):
        if d.is_dir() and (d / "config.json").exists():
            st = status(d.name)
            out.append({"id": d.name, "config": config(d.name), "status": st, "state": live_state(st)})
    return out


def active_run():
    for r in list_runs():
        if r["state"] in ("starting", "loading", "running", "explaining"):
            return r
    return None


def append_csv(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def read_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def folds(rid):
    return read_csv(run_dir(rid) / "folds.csv")


def preds(rid):
    return read_csv(run_dir(rid) / "preds.csv")


def curves(rid):
    return read_csv(run_dir(rid) / "curves.csv")


def done_keys(rid):
    f = folds(rid)
    if f.empty:
        return set()
    return {f"{a}|{b}|{c}" for a, b, c in zip(f["scheme"], f["repeat"], f["fold"])}


def clean_partial(rid):
    """drop prediction / curve rows of folds that were interrupted before folds.csv recorded them"""
    keys = done_keys(rid) | ({"final|1|0"} if final_done(rid) else set())
    for name in ("preds.csv", "curves.csv", "members.csv", "cross.csv"):
        p = run_dir(rid) / name
        df = read_csv(p)
        if df.empty:
            continue
        k = df["scheme"].astype(str) + "|" + df["repeat"].astype(str) + "|" + df["fold"].astype(str)
        keep = df[k.isin(keys)]
        if len(keep) != len(df):
            keep.to_csv(p, index=False)


def delete_run(rid):
    shutil.rmtree(run_dir(rid), ignore_errors=True)
    for ext in (".pt", ".json"):
        p = C.MODELS / f"{rid}{ext}"
        if p.exists():
            p.unlink()


def scheme_summaries(rid):
    """metrics per scheme computed from the run's files (works while the run is still training)"""
    from . import metrics as MT
    cfg = config(rid)
    f, p = folds(rid), preds(rid)
    out = {}
    if f.empty or p.empty:
        return out
    for s in [x for x in C.SCHEMES if x in set(f["scheme"])]:
        out[s] = MT.summarize(p[p["scheme"] == s], f[f["scheme"] == s], s, cfg.get("grouping", "logo"))
    return out


# ======================================================================================================================
# models
# ======================================================================================================================
def model_paths(rid):
    return C.MODELS / f"{rid}.pt", C.MODELS / f"{rid}.json"


def final_done(rid):
    meta = read_json(model_paths(rid)[1], {}) or {}
    return meta.get("kind") == "final" and model_paths(rid)[0].exists()


def save_model(rid, bundle, meta):
    pt, js = model_paths(rid)
    tmp = str(pt) + ".tmp"
    torch.save(bundle, tmp)
    os.replace(tmp, pt)
    D.write_json(js, meta)


def update_model_meta(rid, **kw):
    js = model_paths(rid)[1]
    meta = read_json(js, None)
    if meta is not None:
        meta.update(kw)
        D.write_json(js, meta)


def list_models():
    out = []
    for js in sorted(C.MODELS.glob("*.json"), reverse=True):
        meta = read_json(js, None)
        if meta and (C.MODELS / (js.stem + ".pt")).exists():
            meta["id"] = js.stem
            out.append(meta)
    return out


def load_bundle(model_id):
    return torch.load(C.MODELS / f"{model_id}.pt", map_location="cpu", weights_only=False)


class Predictor:
    """CPU predictor for a saved bundle: rows of raw features (bundle feature order) -> −log10 IC50.
    A seed-ensemble bundle (several state_dicts) averages its models."""

    def __init__(self, bundle):
        self.b = bundle
        self.features = list(bundle["features"])
        t = bundle.get("target", {"kind": "raw"})
        self.target = t
        self.dpos = {d: i for i, d in enumerate(t.get("drugs", []))}
        self.booster = None
        if bundle.get("booster") is not None:               # (v5.9) LightGBM: trees on the raw features
            from . import gbm as G
            self.booster = G.load_booster(bundle["booster"])
            self.best_iteration = int(bundle.get("best_iteration") or 0) or None
            self.fill = np.asarray(bundle["feat_mean"], "float32")
            self.nets, self.net, self.mpos = [], None, {}
            return
        pairs = bundle.get("pairs")
        self.nets = []
        for sd_ in (bundle.get("state_dicts") or [bundle["state_dict"]]):
            net = nets.build(bundle["spec"], len(self.features), pairs, self.features)
            net.load_state_dict(sd_)
            net.eval()
            self.nets.append(net)
        self.net = self.nets[0]
        self.mu = np.asarray(bundle["mu"], "float32"); self.sd = np.asarray(bundle["sd"], "float32")
        self.fill = np.asarray(bundle.get("feat_mean", bundle["mu"]), "float32")
        t = bundle.get("target", {"kind": "raw"})
        self.target = t
        self.dpos = {d: i for i, d in enumerate(t.get("drugs", []))}
        self.mpos = {d: i for i, d in enumerate(bundle.get("mol_drugs") or [])}       # (M) drug -> molecule

    def __call__(self, X, drugs=None):
        X = np.array(X, dtype="float32", copy=True)
        miss = ~np.isfinite(X)
        if miss.any():                                      # a feature the data lacks -> training mean
            X[miss] = np.broadcast_to(self.fill, X.shape)[miss]
        if self.booster is not None:
            p = np.asarray(self.booster.predict(X, num_iteration=self.best_iteration), dtype=np.float64)
            return self._to_response(p, drugs)
        Z = (X - self.mu) / self.sd
        out = []
        mol = nets.is_mol(self.b["spec"])
        if mol:                                             # (M) a drug without a molecule cannot be predicted
            midx = np.array([self.mpos.get(str(d), -1) for d in drugs], dtype=np.int64)
            ok = midx >= 0
            mt = torch.as_tensor(np.maximum(midx, 0))
        with torch.no_grad():
            ch = 1024 if nets.is_graph(self.b["spec"]) else 8192
            for i in range(0, len(Z), ch):
                zb = torch.from_numpy(Z[i:i + ch])
                if mol:
                    out.append(np.mean([n(zb, mt[i:i + ch]).view(-1).double().numpy() for n in self.nets], axis=0))
                else:
                    out.append(np.mean([n(zb).view(-1).double().numpy() for n in self.nets], axis=0))
        p = np.concatenate(out) if out else np.zeros(0)
        if mol:
            p = np.where(ok, p, np.nan)
        return self._to_response(p, drugs)

    def _to_response(self, p, drugs):
        """model output -> response unit (raw / drug-wise z / study-wise z target)"""
        if self.target.get("kind") == "study_z":            # (S) the average source study's scale
            p = p * float(self.target["sd"]) + float(self.target["mu"])
        if self.target.get("kind") == "drug_z":
            idx = np.array([self.dpos.get(d, -1) for d in drugs])
            mu = np.where(idx >= 0, np.asarray(self.target["mu"])[np.maximum(idx, 0)], np.nan)
            sd = np.where(idx >= 0, np.asarray(self.target["sd"])[np.maximum(idx, 0)], np.nan)
            p = p * sd + mu
        return p
