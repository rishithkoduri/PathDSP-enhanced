"""
One training run, executed by train.py in its own process (so training continues when the browser tab is closed
and the app stays responsive). Writes results as each fold-model finishes and a live status.json ~once per second.
"""
import datetime as _dt
import math
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import data as D
from . import engine as E
from . import explain as X
from . import metrics as MT
from . import nets
from . import store as S


def _now():
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def validate(cfg):
    spec = cfg["spec"]
    if not cfg.get("schemes") and not cfg.get("final", True):
        return "Choose at least one evaluation or the all-data model."
    if cfg.get("model") == C.GBM_MODEL:
        from . import gbm as G
        if not G.available():
            return ("LightGBM is not installed — press 'Install LightGBM' under Model, or run: "
                    "python -m pip install lightgbm")
        on = [C.UPGRADES[k]["letter"] for k in ("fac", "gate", "plateau", "ema", "rank", "adv", "mol") if spec.get(k)]
        on += ["D"] if E.ensemble_size(spec) > 1 else []
        if on or spec.get("interaction") or spec.get("arch", C.GBM_MODEL) != C.GBM_MODEL:
            return ("Upgrades " + "·".join(on or ["A–M"]) + " and the interaction module belong to the neural "
                    "networks — LightGBM runs without them.")
    if not D.available().get(cfg.get("data")):
        return f"Dataset '{cfg.get('data')}' is not in the data folder."
    if spec.get("target") == "drug_z" and "LODO" in cfg.get("schemes", []):
        return ("A drug-wise z-scored target needs each drug's training responses — leave-drug-out has none for the "
                "held-out drug. Use the raw target for leave-drug-out.")
    study = (D.meta(cfg.get("data")) or {}).get("study", "")
    if C.CSA_SCHEME in cfg.get("schemes", []) and (not D.is_csa(cfg.get("data")) or study == "ALL"):
        return "The official CSA splits exist only for each single CSA benchmark study (CSA · CCLE … GDSCv2)."
    if C.LOSO_SCHEME in cfg.get("schemes", []) and study != "ALL":
        return "Leave-study-out needs the pooled dataset 'CSA · all 5 studies'."
    if spec.get("target") == "drug_z" and C.LOSO_SCHEME in cfg.get("schemes", []):
        return ("A drug-wise z-scored target needs each drug's training responses, and a held-out study has drugs "
                "the other four never tested — use study-wise z (S) or raw for leave-study-out.")
    if spec.get("target") == "study_z" and not D.is_csa(cfg.get("data")):
        return "The study-wise z target (S) is for the CSA benchmark datasets."
    if spec.get("adv") and study != "ALL":
        return ("Study-adversarial training (H) needs several studies in the training data — use 'CSA · all 5 "
                "studies'.")
    if spec.get("g_graph", "pid") == "random" and not nets.is_graph(spec):
        return "The random-graph control applies to the pathway-graph model."
    if spec.get("mol"):
        from . import molenc as M
        if not D.is_csa(cfg.get("data")):
            return "The learned drug encoder (M) works on the CSA benchmark datasets (their drugs have molecules)."
        if not M.available():
            return "The drug encoder (M) needs data/csa/molecules.npz — run: python tools/build_csa.py --molecules-only"
    if (D.is_csa(cfg.get("data")) and cfg.get("cross") and C.CSA_SCHEME in cfg.get("schemes", [])
            and spec.get("target") == "drug_z"):
        return ("The cross-study test predicts drugs the model never saw, and a drug-wise z-scored target needs each "
                "drug's training mean and sd — use the raw target, or switch the cross-study test off.")
    if not 1 <= E.ensemble_size(spec) <= C.MAX_ENSEMBLE:
        return f"Seeds per fold must be between 1 and {C.MAX_ENSEMBLE}."
    if nets.is_graph(spec):
        from . import pathways as PW
        if PW.gmt_path() is None:
            return (f"The pathway-graph model needs the PID gene sets: put c2.cp.pid.*.symbols.gmt (MSigDB C2 CP:PID) "
                    f"into {C.REFS}.")
    return None


def _same_recipe(a, b):
    """earlier runs whose epoch counts can inform the time estimate"""
    return (all(a.get(k) == b.get(k) for k in ("epochs", "patience"))
            and all(bool(a.get(k)) == bool(b.get(k)) for k in ("plateau", "ema")))


class Run:
    def __init__(self, rid):
        self.rid = rid
        self.dir = S.run_dir(rid)
        self.cfg = S.config(rid)
        self.spec = self.cfg["spec"]
        self.st = {"run_id": rid, "pid": os.getpid(), "state": "starting", "message": "Starting…",
                   "started": time.time(), "updated": time.time(),
                   "prior_seconds": float((S.status(rid) or {}).get("elapsed", 0.0)) if self.cfg.get("resumed") else 0.0}
        self._last_write = 0.0
        self.curve = []
        self.recent = []
        self.ep_done = {}                                   # scheme -> list of epochs_run of finished models
        self.rounds = 0
        self.ep_sum = {}                                    # own epoch -> [sum train, sum val, n] over all models
        self.best = {}                                      # scheme -> best test RMSE (fold kept for SHAP)
        self.shap_scheme = None

    # ------------------------------------------------------------------ status
    def status(self, force=False, **kw):
        self.st.update(kw)
        now = time.time()
        self.st["updated"] = now
        self.st["elapsed"] = self.st["prior_seconds"] + (now - self.st["started"])
        if force or now - self._last_write >= 0.9:
            self._last_write = now
            self.st["curve"] = self.curve[-400:]
            self.st["recent"] = self.recent[-40:]
            try:
                D.write_json(self.dir / "status.json", self.st)
            except Exception:
                pass

    def stop_check(self):
        return (self.dir / C.STOP_FILE).exists()

    # ------------------------------------------------------------------ main
    def run(self):
        try:
            err = validate(self.cfg)
            if err:
                raise ValueError(err)
            if self.cfg.get("model") == C.GBM_MODEL:            # (v5.9) gradient-boosted trees, on the CPU
                self._run_gbm()
                self._finish()
                return
            self.status(True, state="loading", message="Loading data…")
            self.ds = ds = D.load(self.cfg["data"], lambda f, m: self.status(message=m, load_progress=f))
            self.prior = self._prior_epochs()
            jobs = E.make_jobs(ds, self.cfg)
            if self.cfg.get("resumed"):
                S.clean_partial(self.rid)
            done = S.done_keys(self.rid) | ({"final|1|0"} if S.final_done(self.rid) else set())
            f_prev = S.folds(self.rid)
            if not f_prev.empty:                                          # resumed run: earlier sessions' folds
                for s_, g in f_prev.groupby("scheme"):
                    self.ep_done.setdefault(s_, []).extend(g["epochs_run"].astype(float).tolist())
                bf = self.dir / "bestfold.pt"
                if bf.exists():
                    b = torch.load(bf, map_location="cpu", weights_only=False)
                    self.best[b["scheme"]] = float(b["rmse"])
            pending = [j for j in jobs if j.key not in done]
            self.n_jobs, self.n_done = len(jobs), len(jobs) - len(pending)
            self.shap_scheme = "mixed" if "mixed" in self.cfg["schemes"] else (self.cfg["schemes"] or [None])[0]
            self.ctx = E.Context(ds, self.spec, self.cfg["seed"])
            self.tgt_kind = self.spec.get("target", "raw")
            par = self.cfg.get("parallel", "auto")
            max_perf = self.cfg.get("perf", "max") == "max"
            P = (E.auto_parallel(len(pending), self.ctx.arch, self.spec, ds.n, max_perf=max_perf)
                 if par in ("auto", 0, None) else max(1, int(par)))
            P = max(1, min(P, max(1, len(pending))))
            self.status(True, state="running", message="Training", n_jobs=self.n_jobs, done_jobs=self.n_done,
                        parallel=P, gpu=E.gpu_info(), device=str(self.ctx.dev), optimizer=self.ctx.opt_name,
                        perf="max" if max_perf else "balanced")
            self._train(pending, P)
            self._finish()
        except E.Stopped:
            self.status(True, state="stopped", message="Stopped — finished folds are saved; Resume continues.")
        except Exception as e:
            self.status(True, state="failed", message=f"{type(e).__name__}: {e}", error=traceback.format_exc())
            raise
        finally:
            try:
                os.remove(self.dir / C.STOP_FILE)
            except OSError:
                pass

    # ------------------------------------------------------------------ the pool
    def _prepare(self, job):
        t, inv, base = E.target_of(self.ds, job, self.tgt_kind)
        job.extra.update(target=t, inv=inv, base=base)
        if job.scheme == C.LOSO_SCHEME and hasattr(self.ds.X, "value_range"):
            # the held-out study has drugs / cell lines the model never saw: clip their inputs to the range of the
            # training rows (as in the cross-study test), so a near-constant training feature cannot explode
            job.extra["clip"] = self.ds.X.value_range(job.tr)

    def _train(self, pending, P):
        self.step_time = E.StepTime()
        self.P = P
        self.cur_steps = math.ceil(max(len(j.tr_in) for j in pending) / int(self.spec["batch"])) if pending else 1
        self.queued = len(pending)
        self._models = []
        self._last_gpu = 0.0
        self._ens = {}                                      # fold key -> finished ensemble members (upgrade D)
        while pending:
            self._ens.clear()
            self.status(True, pool={"round": self.rounds, "slots": P, "active": 0, "queued": len(pending),
                                    "step": 0, "steps": self.cur_steps, "epochs": int(self.spec["epochs"])})
            oom = False
            try:
                E.Pool(self.ctx, pending, P).run(self._prepare, self._on_step, self._on_round, self._on_done,
                                                 self.stop_check)
            except torch.cuda.OutOfMemoryError:
                if P == 1:
                    raise
                oom = True
            if not oom:
                break
            import gc                                       # outside the except block, so the memory is really freed
            gc.collect()
            torch.cuda.empty_cache()
            P = max(1, P // 2)
            self.P = P
            self.status(True, parallel=P, message=f"GPU memory full — continuing with {P} models at once")
            done = S.done_keys(self.rid) | ({"final|1|0"} if S.final_done(self.rid) else set())
            pending = [j for j in pending if j.key not in done]   # every member of an unfinished fold is retrained
            for j in pending:
                j.extra.clear()
            self.n_done = self.n_jobs - len(pending)
            self.status(True, done_jobs=self.n_done)
        self.ctx.release()

    def _gpu(self, force=False):
        now = time.time()
        if force or now - self._last_gpu >= 2.0:
            self._last_gpu = now
            try:
                return E.gpu_info()
            except Exception:
                pass
        return self.st.get("gpu")

    def _on_step(self, info):
        p = dict(self.st.get("pool", {}))
        p.update(round=info["round"], step=info["step"], steps=info["steps"], active=info["active"],
                 slots=info["slots"], queued=info["queued"], train_loss_now=info["train_loss"])
        self.queued = info["queued"]
        self.status(pool=p, gpu=self._gpu(), eta=self._eta(info["step"] / max(info["steps"], 1), info["active"]))

    def _on_round(self, info):
        A = info["active"]
        self.rounds = info["round"]
        self.cur_steps = info["steps"]
        self.queued = info["queued"]
        self.step_time.add(A, info["round_seconds"] / max(info["steps"], 1))
        for m in info["models"]:
            acc = self.ep_sum.setdefault(m["ep"], [0.0, 0.0, 0])
            acc[0] += m["train"]; acc[1] += m["val"]; acc[2] += 1
        self.curve = [[e, round(v[0] / v[2], 5), round(v[1] / v[2], 5), v[2]] for e, v in sorted(self.ep_sum.items())]
        self._models = info["models"]
        eps = [m["ep"] for m in info["models"]] or [0]
        sps = A * int(self.spec["batch"]) * info["steps"] / max(info["train_seconds"], 1e-9)
        p = dict(self.st.get("pool", {}))
        p.update(round=info["round"], active=A, slots=info["slots"], queued=info["queued"], steps=info["steps"],
                 step=info["steps"], round_seconds=info["round_seconds"],
                 step_ms=1000 * info["round_seconds"] / max(info["steps"], 1), samples_per_s=sps,
                 train_loss=float(np.mean([m["train"] for m in info["models"]])),
                 val_loss=float(np.mean([m["val"] for m in info["models"]])), epoch_min=int(min(eps)),
                 epoch_max=int(max(eps)), patience=info["patience"], epochs=info["epochs"])
        self.status(True, pool=p, models=info["models"], gpu=self._gpu(force=True), eta=self._eta(0.0, A))

    # ------------------------------------------------------------------ results of one model / one fold
    def _on_done(self, job, r):
        """a model stopped. Without a seed ensemble that finishes its fold; with one (upgrade D) the fold is finished
        when its last member stops, and the members' predictions are averaged."""
        self.ep_done.setdefault(job.scheme, []).append(float(r["epochs_run"]))
        self.n_done += 1
        buf = self._ens.setdefault(job.key, [])
        buf.append((job, r))
        if len(buf) >= E.ensemble_size(self.spec):
            del self._ens[job.key]
            buf.sort(key=lambda t: t[0].member)
            self._fold_done(buf)
        self.status(True, done_jobs=self.n_done)

    def _fold_done(self, buf):
        ds, spec = self.ds, self.spec
        job, r0 = buf[0]
        rs = [r for _, r in buf]
        n_ens = len(rs)
        states = [nets.to_state_dict(r["params"], 0, spec, ds.features, ds.interaction_pairs) for r in rs]
        best_epoch = int(round(float(np.mean([r["best_epoch"] for r in rs]))))
        epochs_run = int(round(float(np.mean([r["epochs_run"] for r in rs]))))
        seconds = float(max(r["seconds"] for r in rs))
        curves = pd.DataFrame({"scheme": job.scheme, "repeat": job.rep, "fold": job.fold,      # curves of seed 1
                               "epoch": np.arange(1, len(r0["hist"]["train"]) + 1),
                               "train": r0["hist"]["train"], "val": r0["hist"]["val"]})
        if job.final:
            self._save_model(states, r0, job, "final", best_epoch, epochs_run)
            S.append_csv(self.dir / "curves.csv", curves)
            self.recent.append({"model": "all-data model", "RMSE": None, "PCC": None, "epochs": epochs_run,
                                "best_epoch": best_epoch, "seconds": seconds})
        else:
            te = job.te
            codes = ds.drug_codes[te]
            y = ds.y[te].astype("float64")
            preds_m = [E.inverse(r["pred_t"], job.extra["inv"], codes, ds.scodes[te]) for r in rs]
            pred = np.mean(preds_m, axis=0) if n_ens > 1 else preds_m[0]
            m = MT.regression(y, pred)
            if (job.scheme == C.CSA_SCHEME and self.cfg.get("cross")
                    and job.extra["inv"].get("kind") in ("raw", "study_z")):
                self._cross(job, rs)                                      # before folds.csv: a resume redoes both
            S.append_csv(self.dir / "preds.csv", pd.DataFrame({
                "scheme": job.scheme, "repeat": job.rep, "fold": job.fold, "drug": ds.drugs[te], "cell": ds.cells[te],
                "actual": np.round(y, 6), "pred": np.round(pred, 6), "base": np.round(job.extra["base"], 6)}))
            S.append_csv(self.dir / "curves.csv", curves)
            if n_ens > 1:                                                 # every seed on its own, for comparison
                rows = []
                for (j_, r_), p_ in zip(buf, preds_m):
                    mm = MT.regression(y, p_)
                    rows.append({"scheme": job.scheme, "repeat": job.rep, "fold": job.fold, "member": j_.member,
                                 **{q: mm[q] for q in ("RMSE", "MAE", "R2", "PCC", "SCC")},
                                 "best_epoch": r_["best_epoch"], "epochs_run": r_["epochs_run"],
                                 "seconds": round(r_["seconds"], 2)})
                S.append_csv(self.dir / "members.csv", pd.DataFrame(rows, columns=S.MEMBER_COLS))
            S.append_csv(self.dir / "folds.csv", pd.DataFrame([{
                "scheme": job.scheme, "repeat": job.rep, "fold": job.fold, "group": job.group, "n_test": len(te),
                **{q: m[q] for q in ("RMSE", "MAE", "R2", "PCC", "SCC")}, "best_epoch": best_epoch,
                "epochs_run": epochs_run, "seconds": round(seconds, 2), "parallel": self.P,
                "group_index": min(r["round0"] for r in rs)}], columns=S.FOLD_COLS))
            self.recent.append({"model": E.label_of(job, seed=False),
                                "RMSE": m["RMSE"], "PCC": m["PCC"], "epochs": epochs_run,
                                "best_epoch": best_epoch, "seconds": seconds})
            if job.scheme == self.shap_scheme and job.rep == 1 and m["RMSE"] < self.best.get(job.scheme, np.inf):
                self.best[job.scheme] = m["RMSE"]
                torch.save({"state_dict": states[0], "state_dicts": states, "mu": r0["mu"], "sd": r0["sd"],
                            "tr": job.tr, "te": job.te, "inv": job.extra["inv"], "scheme": job.scheme,
                            "fold": job.fold, "rmse": m["RMSE"], "best_epoch": best_epoch}, self.dir / "bestfold.pt")
        for j_, _ in buf:
            j_.extra.clear()

    def _cross(self, job, rs):
        """CSA cross-study test: this split's model(s) predict every pair of the four other studies. Inputs are clipped
        to the range of each feature over the drugs / cell lines of the split's training rows: a feature that is nearly
        constant in the source study would otherwise put an unseen drug hundreds of sd outside anything the network
        saw (the within-study test never needs this — its drugs and cell lines all occur in training)."""
        from . import csa as CSA
        if not hasattr(self, "_xrows"):
            self._xrows = {}
        lo, hi = self.ds.X.value_range(job.tr)
        rows = []
        for t in CSA.STUDIES:
            if t == self.ds.study:
                continue
            if t not in self._xrows:
                self._xrows[t] = CSA.target_rows(t)
            di, ci, y, dn = self._xrows[t]
            self.status(message=f"Cross-study test · split {job.fold - 1} → {t} ({len(y):,} pairs)")
            p = np.mean([E.inverse(E.predict_pairs(self.ctx, r["params"], r["mu"], r["sd"], di, ci, lo, hi),
                                   job.extra["inv"], None) for r in rs], axis=0)
            m = MT.regression(y, p)
            rows.append({"scheme": job.scheme, "repeat": job.rep, "fold": job.fold, "source": self.ds.study,
                         "target": t, "n": int(len(y)), **{q: m[q] for q in ("RMSE", "MAE", "R2", "PCC", "SCC")},
                         "per_drug_PCC": MT.per_drug_pcc(pd.DataFrame({"drug": dn, "actual": y, "pred": p}))})
        S.append_csv(self.dir / "cross.csv", pd.DataFrame(rows, columns=S.CROSS_COLS))
        self.status(message="Training")

    def _target_dict(self, inv):
        """how Predict converts the model's output back to the response unit"""
        ds = self.ds
        if inv.get("kind") == "drug_z":
            return {"kind": "drug_z", "drugs": list(ds.drug_list), "mu": np.asarray(inv["mu"]), "sd": np.asarray(inv["sd"])}
        if inv.get("kind") == "study_z":                    # a new pair has no study: the average source study
            return {"kind": "study_z", "mu": float(inv["fallback"][0]), "sd": float(inv["fallback"][1]),
                    "studies": list(inv.get("studies", [])), "study_mu": np.asarray(inv["mu"]),
                    "study_sd": np.asarray(inv["sd"])}
        return {"kind": "raw"}

    def _bundle(self, states, mu, sd, inv, kind, n_train, extra=None):
        ds = self.ds
        dg, cl, _ = ds.interaction_pairs
        tgt = self._target_dict(inv)
        out = {"format": "pathdsp-app-1", "run_id": self.rid, "model": self.cfg["model"], "spec": self.spec,
               "data": self.cfg["data"], "features": list(ds.features),
               "pairs": (dg, cl) if self.spec.get("interaction") else None,
               "mu": np.asarray(mu, "float32"), "sd": np.asarray(sd, "float32"),
               "feat_mean": ds.X.mean(0).astype("float32"), "target": tgt, "state_dict": states[0], "kind": kind,
               "n_train": int(n_train), "created": _now(), **(extra or {})}
        if len(states) > 1:                                 # (D) Predict averages every seed
            out["state_dicts"] = list(states)
        if nets.is_mol(self.spec):                          # (M) drug name -> molecule (CSA drug index)
            from . import csa as CSA
            out["mol_drugs"] = [str(x) for x in CSA.package()["labels"]]
        return out

    def _model_meta(self, kind, n_train, extra=None):
        ms = C.MODEL_SPECS[self.cfg["model"]]
        tag = ms["tag"] + (f"\u00b7{self.spec.get('g_mode', 'hybrid')}" if nets.is_graph(self.spec) else "")
        if nets.is_graph(self.spec) and self.spec.get("g_graph", "pid") == "random":
            tag += "\u00b7random-graph"
        return {"run_id": self.rid, "model": self.cfg["model"], "model_name": ms["name"], "tag": tag,
                "data": self.cfg["data"], "data_name": D.dataset_name(self.cfg["data"]), "kind": kind,
                "target": self.spec.get("target", "raw"), "upgrades": C.upgrades_tag(self.spec),
                "n_train": int(n_train), "created": _now(), **(extra or {})}

    def _save_model(self, states, r, job, kind, best_epoch, epochs_run):
        b = self._bundle(states, r["mu"], r["sd"], job.extra["inv"], kind, len(job.tr),
                         {"best_epoch": best_epoch, "epochs_run": epochs_run})
        S.save_model(self.rid, b, self._model_meta(kind, len(job.tr)))

    # ------------------------------------------------------------------ LightGBM (v5.9)
    def _run_gbm(self):
        """gradient-boosted trees: the same jobs (folds, inner validation split, targets) as the networks, one model
        at a time on all CPU cores; writes the same files"""
        import gc
        from . import gbm as G
        self.status(True, state="loading", message="Loading data…")
        self.ds = ds = D.load(self.cfg["data"], lambda f, m: self.status(message=m, load_progress=f))
        self.prior = self._prior_epochs()
        jobs = E.make_jobs(ds, self.cfg)
        if self.cfg.get("resumed"):
            S.clean_partial(self.rid)
        done = S.done_keys(self.rid) | ({"final|1|0"} if S.final_done(self.rid) else set())
        f_prev = S.folds(self.rid)
        if not f_prev.empty:
            for s_, g in f_prev.groupby("scheme"):
                self.ep_done.setdefault(s_, []).extend(g["epochs_run"].astype(float).tolist())
            bf = self.dir / "bestfold.pt"
            if bf.exists():
                b = torch.load(bf, map_location="cpu", weights_only=False)
                self.best[b["scheme"]] = float(b["rmse"])
        pending = [j for j in jobs if j.key not in done]
        self.n_jobs, self.n_done = len(jobs), len(jobs) - len(pending)
        self.shap_scheme = "mixed" if "mixed" in self.cfg["schemes"] else (self.cfg["schemes"] or [None])[0]
        self.tgt_kind = self.spec.get("target", "raw")
        self.P = 1
        self.n_threads = G.threads()
        self.gb_sec = {}                                    # job key -> seconds per boosting round
        self.gb_build = []                                  # seconds to bin a job's data
        dev = {"cuda": False, "name": f"CPU · {self.n_threads} cores"}
        self.status(True, state="running", message="Training", n_jobs=self.n_jobs, done_jobs=self.n_done, parallel=1,
                    gpu=dev, device="cpu", optimizer=f"LightGBM {G.version() or ''} · {self.n_threads} threads",
                    perf=self.cfg.get("perf", "max"))
        for i, job in enumerate(pending):
            if self.stop_check():
                raise E.Stopped()
            self._prepare(job)
            self.queued = len(pending) - i - 1
            self.status(True, message=f"Binning the data · {E.label_of(job, seed=False)} "
                                      f"({len(job.tr_in):,} training pairs)", models=[],
                        pool={"active": 0, "slots": 1, "queued": self.queued + 1, "steps": 0,
                              "epochs": int(self.spec["epochs"]), "patience": int(self.spec["patience"])})
            clock = {"t0": time.time(), "t1": None, "last": 0.0}

            def on_iter(it, tr_, va_, best_it, best_v, job=job, clock=clock):
                now = time.time()
                if clock["t1"] is None:
                    clock["t1"] = now
                    self.gb_build.append(now - clock["t0"])
                acc = self.ep_sum.setdefault(it, [0.0, 0.0, 0])
                acc[0] += tr_; acc[1] += va_; acc[2] += 1
                if now - clock["last"] < 1.0 and it > 1:
                    return
                clock["last"] = now
                sec = (now - clock["t1"]) / max(1, it - 1) if it > 1 else 0.0
                if sec > 0:
                    self.gb_sec[job.key] = sec
                keys = sorted(self.ep_sum)
                k = max(1, int(np.ceil(len(keys) / 400)))
                self.curve = [[e, round(self.ep_sum[e][0] / self.ep_sum[e][2], 5),
                               round(self.ep_sum[e][1] / self.ep_sum[e][2], 5), self.ep_sum[e][2]]
                              for e in keys[k - 1::k]]
                m = {"label": E.label_of(job, seed=False), "ep": it, "train": tr_, "val": va_, "best": best_v,
                     "best_ep": best_it, "wait": it - best_it, "improved": it == best_it, "stopping": False,
                     "seconds": now - clock["t0"], "lr": float(self.spec["lr"])}
                self.status(message="Training", models=[m], eta=self._eta_gbm(it, best_it, sec),
                            pool={"round": it, "active": 1, "slots": 1, "queued": self.queued, "steps": 0, "step": 0,
                                  "epochs": int(self.spec["epochs"]), "patience": int(self.spec["patience"]),
                                  "epoch_min": it, "epoch_max": it, "round_seconds": sec, "step_ms": 1000 * sec,
                                  "samples_per_s": len(job.tr_in) / sec if sec > 0 else None,
                                  "train_loss": tr_, "val_loss": va_})
            try:
                r = G.train(ds.X, job.extra["target"], job, self.spec, int(self.cfg["seed"]), on_iter, self.stop_check,
                            self.n_threads)
            except G.StopTraining:
                raise E.Stopped()
            self._gbm_done(job, r)
            del r
            job.extra.clear()
            gc.collect()

    def _gbm_done(self, job, r):
        from . import gbm as G
        ds = self.ds
        self.ep_done.setdefault(job.scheme, []).append(float(r["iters"]))
        self.n_done += 1
        booster, bi = r["booster"], int(r["best_iter"])
        keep = G.thin(r["hist"]["train"], r["hist"]["val"], bi)
        curves = pd.DataFrame({"scheme": job.scheme, "repeat": job.rep, "fold": job.fold, "epoch": keep + 1,
                               "train": np.asarray(r["hist"]["train"])[keep], "val": np.asarray(r["hist"]["val"])[keep]})
        if job.final:
            b = self._gbm_bundle(booster.model_to_string(num_iteration=bi), bi, job.extra["inv"], "final", len(job.tr),
                                 {"best_epoch": bi, "epochs_run": int(r["iters"])})
            S.save_model(self.rid, b, self._model_meta("final", len(job.tr)))
            S.append_csv(self.dir / "curves.csv", curves)
            self.recent.append({"model": "all-data model", "RMSE": None, "PCC": None, "epochs": int(r["iters"]),
                                "best_epoch": bi, "seconds": r["seconds"]})
        else:
            te = job.te
            self.status(True, message=f"Predicting {len(te):,} held-out pairs · {E.label_of(job, seed=False)}")
            y = ds.y[te].astype("float64")
            pred = E.inverse(G.predict_rows(booster, ds.X, te, bi), job.extra["inv"], ds.drug_codes[te], ds.scodes[te])
            m = MT.regression(y, pred)
            if (job.scheme == C.CSA_SCHEME and self.cfg.get("cross")
                    and job.extra["inv"].get("kind") in ("raw", "study_z")):
                self._cross_gbm(job, booster, bi)                          # before folds.csv: a resume redoes both
            S.append_csv(self.dir / "preds.csv", pd.DataFrame({
                "scheme": job.scheme, "repeat": job.rep, "fold": job.fold, "drug": ds.drugs[te], "cell": ds.cells[te],
                "actual": np.round(y, 6), "pred": np.round(pred, 6), "base": np.round(job.extra["base"], 6)}))
            S.append_csv(self.dir / "curves.csv", curves)
            S.append_csv(self.dir / "folds.csv", pd.DataFrame([{
                "scheme": job.scheme, "repeat": job.rep, "fold": job.fold, "group": job.group, "n_test": len(te),
                **{q: m[q] for q in ("RMSE", "MAE", "R2", "PCC", "SCC")}, "best_epoch": bi,
                "epochs_run": int(r["iters"]), "seconds": round(r["seconds"], 2), "parallel": 1,
                "group_index": self.n_done}], columns=S.FOLD_COLS))
            self.recent.append({"model": E.label_of(job, seed=False), "RMSE": m["RMSE"], "PCC": m["PCC"],
                                "epochs": int(r["iters"]), "best_epoch": bi, "seconds": r["seconds"]})
            if job.scheme == self.shap_scheme and job.rep == 1 and m["RMSE"] < self.best.get(job.scheme, np.inf):
                self.best[job.scheme] = m["RMSE"]
                torch.save({"booster": booster.model_to_string(num_iteration=bi), "tr": job.tr, "te": job.te,
                            "inv": job.extra["inv"], "scheme": job.scheme, "fold": job.fold, "rmse": m["RMSE"],
                            "best_epoch": bi}, self.dir / "bestfold.pt")
        self.status(True, done_jobs=self.n_done, models=[], message="Training")

    def _cross_gbm(self, job, booster, bi):
        """CSA cross-study test of one LightGBM split model (trees need no input clipping: a value outside the
        training range falls into the outermost bin)"""
        from . import csa as CSA
        from . import gbm as G
        if not hasattr(self, "_xrows"):
            self._xrows = {}
        rows = []
        for t in CSA.STUDIES:
            if t == self.ds.study:
                continue
            if t not in self._xrows:
                self._xrows[t] = CSA.target_rows(t)
            di, ci, y, dn = self._xrows[t]
            self.status(message=f"Cross-study test · split {job.fold - 1} → {t} ({len(y):,} pairs)")
            p = E.inverse(G.predict_pairs(booster, self.ds.X.D, self.ds.X.C, di, ci, bi), job.extra["inv"], None)
            m = MT.regression(y, p)
            rows.append({"scheme": job.scheme, "repeat": job.rep, "fold": job.fold, "source": self.ds.study,
                         "target": t, "n": int(len(y)), **{q: m[q] for q in ("RMSE", "MAE", "R2", "PCC", "SCC")},
                         "per_drug_PCC": MT.per_drug_pcc(pd.DataFrame({"drug": dn, "actual": y, "pred": p}))})
        S.append_csv(self.dir / "cross.csv", pd.DataFrame(rows, columns=S.CROSS_COLS))

    def _gbm_bundle(self, model_str, best_iter, inv, kind, n_train, extra=None):
        ds = self.ds
        return {"format": "pathdsp-app-1", "run_id": self.rid, "model": self.cfg["model"], "spec": self.spec,
                "data": self.cfg["data"], "features": list(ds.features), "pairs": None, "booster": model_str,
                "best_iteration": int(best_iter), "feat_mean": ds.X.mean(0).astype("float32"),
                "target": self._target_dict(inv), "kind": kind, "n_train": int(n_train), "created": _now(),
                **(extra or {})}

    def _eta_gbm(self, it, best_it, sec):
        """seconds left: the current model's expected stop (earliest = best + patience) + the queued models' boosting
        rounds (rounds earlier models needed, this run else earlier runs) × seconds per round + their data binning"""
        try:
            if sec <= 0:
                return None
            E_, pat = int(self.spec["epochs"]), int(self.spec["patience"])
            dist = self._dist(list(self.cfg.get("schemes", [])) + ["final"])
            lb = min(E_, best_it + pat)
            cond = [d for d in (dist or []) if d > it]
            exp_end = min(E_, max(lb, float(np.median(cond)) if len(cond) >= 3 else lb))
            nq = int(self.queued)
            q_exp = np.clip(E.expected_ends(nq, dist, default=min(E_, 1000)), min(pat + 1.0, E_), E_)
            build = float(np.mean(self.gb_build)) if self.gb_build else 0.0
            secs = list(self.gb_sec.values()) or [sec]
            s_q = float(np.mean(secs))
            tail = 30.0
            return {"expected": (exp_end - it) * sec + float(q_exp.sum()) * s_q + nq * build + tail,
                    "min": (lb - it) * sec + nq * (min(pat + 1.0, E_) * s_q + build) + tail,
                    "max": (E_ - it) * sec + nq * (E_ * s_q + build) + tail,
                    "basis": "finished models" if dist is not None else f"default (~{min(E_, 1000)} rounds)"}
        except Exception:
            return None

    # ------------------------------------------------------------------ time estimates
    def _prior_epochs(self):
        """epochs that earlier runs of the same model + dataset needed (per scheme)"""
        out = {}
        for r in S.list_runs():
            c = r["config"]
            if r["id"] == self.rid or c.get("model") != self.cfg["model"] or c.get("data") != self.cfg["data"] \
                    or not _same_recipe(c.get("spec") or {}, self.spec):
                continue
            f = S.folds(r["id"])
            if not f.empty:
                for s_, g in f.groupby("scheme"):
                    out.setdefault(s_, []).extend(g["epochs_run"].astype(float).tolist())
        return out

    def _dist(self, schemes):
        v = []
        for s_ in schemes:
            v += self.ep_done.get(s_, [])
        if len(v) < 3:
            for s_ in schemes:
                v += self.prior.get(s_, [])
        if len(v) < 3:
            v = self.ep_done.get("mixed", []) + self.prior.get("mixed", []) + sum(self.ep_done.values(), [])
        return v if len(v) >= 3 else None

    def _eta(self, round_frac, active):
        """expected / fastest / worst-case seconds left. Active models: their own epoch and patience state; queued
        models: the epochs finished models needed (this run, else earlier runs of the same model + data)."""
        try:
            E_, pat = int(self.spec["epochs"]), int(self.spec["patience"])
            st = self.step_time
            if not st.pts:
                return None
            dist = self._dist(list(self.cfg.get("schemes", [])) + ["final"])
            rem, rem_min, rem_max = [], [], []
            for m in self._models:
                if m.get("stopping"):
                    continue
                ep, lb_end = m["ep"], min(E_, m["ep"] + pat - m["wait"])          # earliest possible stop
                cond = [d for d in (dist or []) if d > ep]
                guess = float(np.median(cond)) if len(cond) >= 3 else (lb_end + pat * 0.5 if dist else max(lb_end, min(140.0, E_)))
                rem.append(min(E_, max(lb_end, guess)) - ep)
                rem_min.append(lb_end - ep)
                rem_max.append(E_ - ep)
            nq = int(self.queued)
            q_exp = np.clip(E.expected_ends(nq, dist), min(pat + 1.0, E_), E_)
            P, steps = max(1, int(self.P)), self.cur_steps
            done_now = round_frac * steps * st(max(active, 1))                   # part of the current epoch done
            tail = 30.0                                                         # summary + SHAP
            exp = E.simulate_pool(rem, q_exp, P, steps, st) - done_now
            mn = E.simulate_pool(rem_min, np.full(nq, min(pat + 1.0, E_)), P, steps, st) - done_now
            mx = E.simulate_pool(rem_max, np.full(nq, float(E_)), P, steps, st) - done_now
            return {"expected": max(exp, 0.0) + tail, "min": max(mn, 0.0) + tail, "max": max(mx, 0.0) + tail,
                    "basis": "finished models" if dist is not None else "default (~140 epochs)"}
        except Exception:
            return None

    # ------------------------------------------------------------------ end of run
    def _finish(self):
        bf = self.dir / "bestfold.pt"
        if not S.final_done(self.rid) and bf.exists():                    # no all-data model -> Predict uses best fold
            b = torch.load(bf, map_location="cpu", weights_only=False)
            extra = {"best_epoch": b["best_epoch"], "fold": f"{b['scheme']} fold {b['fold']}"}
            if b.get("booster") is not None:                                   # (v5.9) LightGBM
                bundle = self._gbm_bundle(b["booster"], b["best_epoch"], b["inv"], "bestfold", len(b["tr"]), extra)
            else:
                bundle = self._bundle(b.get("state_dicts") or [b["state_dict"]], b["mu"], b["sd"], b["inv"], "bestfold",
                                      len(b["tr"]), extra)
            S.save_model(self.rid, bundle, self._model_meta("bestfold", len(b["tr"]), {"fold": f"{b['scheme']} fold {b['fold']}"}))
        sums = S.scheme_summaries(self.rid)
        self.status(True, state="explaining", message="Computing SHAP (pathway-level insight)…", models=[], eta=None)
        shap_info = self._shap(bf)
        ext = self._external()
        f = S.folds(self.rid)
        timing = {"wall_seconds": self.st["elapsed"], "rounds": self.rounds,
                  "models": int(len(f)) + (1 if S.final_done(self.rid) else 0),
                  "fold_seconds_mean": float(f["seconds"].mean()) if not f.empty else None,
                  "epochs_mean": float(f["epochs_run"].mean()) if not f.empty else None,
                  "parallel": self.st.get("parallel"), "device": self.st.get("gpu", {}).get("name")}
        fewshot = None
        if C.LOSO_SCHEME in self.cfg.get("schemes", []):
            from . import csa as CSA
            try:
                fs = CSA.fewshot(S.preds(self.rid), S.folds(self.rid), int(self.cfg["seed"]))
                if not fs.empty:
                    fs.to_csv(self.dir / "fewshot.csv", index=False)
                    fewshot = True
            except Exception as e:
                fewshot = {"error": f"{type(e).__name__}: {e}"}
        cross = None
        if D.is_csa(self.cfg.get("data")) and (D.meta(self.cfg.get("data")) or {}).get("study") != "ALL":
            from . import csa as CSA
            try:
                t = CSA.run_table(self.rid)
                cross = None if t.empty else t.replace({np.nan: None}).to_dict("records")
            except Exception as e:
                cross = {"error": f"{type(e).__name__}: {e}"}
        D.write_json(self.dir / "summary.json", {"run_id": self.rid, "finished": _now(), "schemes": sums,
                                                 "timing": timing, "shap": shap_info, "external": ext,
                                                 "cross": cross, "fewshot": fewshot})
        S.update_model_meta(self.rid, cv={s: {k: v.get(k) for k in ("RMSE", "RMSE_sd", "PCC", "per_drug_PCC")}
                                          for s, v in sums.items()})
        self.status(True, state="finished", message="Finished", eta=None, done_jobs=self.n_done, models=[])

    def _shap(self, bf):
        if not bf.exists():
            return None
        try:
            ds = self.ds
            b = torch.load(bf, map_location="cpu", weights_only=False)
            if b.get("booster") is not None:                                   # (v5.9) LightGBM: exact TreeSHAP
                from . import gbm as G
                rng = np.random.RandomState(int(self.cfg["seed"]))
                te = np.sort(rng.choice(b["te"], min(500, len(b["te"])), replace=False))
                Xs = G.rows_dense(ds.X, te)
                vals = G.contributions(G.load_booster(b["booster"]), Xs, None)
                feat, blocks, pathways = X.package(vals, Xs, ds.features)
                feat.to_csv(self.dir / "shap_features.csv", index=False)
                blocks.to_csv(self.dir / "shap_blocks.csv", index=False)
                pathways.to_csv(self.dir / "shap_pathways.csv", index=False)
                return {"method": "TreeSHAP (LightGBM, exact)", "fold": f"{b['scheme']} fold {b['fold']}",
                        "background": None, "explained": len(te),
                        "unit": {"drug_z": "drug-wise z", "study_z": "study-wise z"}.get(
                            self.spec.get("target"), D.unit_of(self.cfg.get("data"))["name"])}
            rng = np.random.RandomState(int(self.cfg["seed"]))
            bg = np.sort(rng.choice(b["tr"], min(100, len(b["tr"])), replace=False))
            te = np.sort(rng.choice(b["te"], min(500, len(b["te"])), replace=False))
            dev = E.device()
            mol = nets.is_mol(self.spec)
            feat, blocks, pathways, method = X.explain_fold(
                b["state_dict"], self.spec, ds.features, ds.interaction_pairs, b["mu"], b["sd"], ds.X[bg], ds.X[te], dev,
                progress=lambda f: self.status(message=f"Computing SHAP… {int(f * 100)} %"),
                mol_bg=ds.X.di[bg] if mol else None, mol_te=ds.X.di[te] if mol else None)
            feat.to_csv(self.dir / "shap_features.csv", index=False)
            blocks.to_csv(self.dir / "shap_blocks.csv", index=False)
            pathways.to_csv(self.dir / "shap_pathways.csv", index=False)
            if nets.is_graph(self.spec):                                # drug-conditioned pathway attention
                X.attention_table(b["state_dict"], self.spec, ds.features, b["mu"], b["sd"], ds.X[te],
                                  ds.drugs[te], ds.X.di[te] if mol else None).to_csv(
                    self.dir / "attention_pathways.csv", index=False)
            n_ens = len(b.get("state_dicts") or [1])
            if n_ens > 1:
                method += f" · seed 1 of {n_ens}"
            return {"method": method, "fold": f"{b['scheme']} fold {b['fold']}", "background": len(bg),
                    "explained": len(te), "unit": "drug-wise z" if self.spec.get("target") == "drug_z" else
                    D.unit_of(self.cfg.get("data"))["name"]}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _external(self):
        ext_key = D.external_for(self.cfg["data"])
        if not ext_key:
            return None
        try:
            r = external_test(self.rid, ext_key)
            return r
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}


def external_test(rid, ext_key="ccle"):
    """Apply the run's model (trained on GDSC) to CCLE, as in the paper; writes external_ccle.json."""
    b = S.load_bundle(rid)
    ds = D.load(ext_key)
    Xa = D.align(ds, b["features"])
    pred = S.Predictor(b)(Xa, ds.drugs)
    ok = np.isfinite(pred)
    if ok.sum() < 3:
        out = {"error": "A drug-wise z-scored model can only predict drugs it was trained on, and too few CCLE drugs "
                        "share a name with them — external testing needs the raw target."}
        D.write_json(S.run_dir(rid) / "external_ccle.json", out)
        return out
    m = MT.regression(ds.y[ok], pred[ok])
    df = pd.DataFrame({"drug": ds.drugs[ok], "cell": ds.cells[ok], "actual": ds.y[ok], "pred": pred[ok]})
    out = {**m, "per_drug_PCC": MT.per_drug_pcc(df), "pairs": int(ok.sum()), "drugs": int(df["drug"].nunique()),
           "cells": int(df["cell"].nunique()), "model_kind": b.get("kind"),
           "missing_features": int(sum(f not in set(ds.features) for f in b["features"]))}
    D.write_json(S.run_dir(rid) / "external_ccle.json", out)
    return out


# ======================================================================================================================
# benchmark: measured seconds per step -> time estimate before a run starts
# ======================================================================================================================
def benchmark(cfg, n_steps=40):
    if cfg.get("model") == C.GBM_MODEL:
        return _benchmark_gbm(cfg)
    ds = D.load(cfg["data"])
    spec = cfg["spec"]
    jobs = E.make_jobs(ds, cfg)
    ctx = E.Context(ds, spec, cfg["seed"])
    par = cfg.get("parallel", "auto")
    P = (E.auto_parallel(len(jobs), ctx.arch, spec, ds.n, max_perf=cfg.get("perf", "max") == "max")
         if par in ("auto", 0, None) else max(1, int(par)))
    P = max(1, min(P, len(jobs)))
    st = E.StepTime()
    dev = ctx.dev
    bs = int(spec["batch"])
    for A in sorted(set([P, max(1, P // 2)]), reverse=True):
        gen = torch.Generator(device=dev).manual_seed(0)
        params = nets.init_batched(A, ctx.arch, spec["init"], gen, dev)
        opt = E.SlotOpt(params, spec, ctx.fused_fn)
        ema = [p.detach().clone() for p in params] if spec.get("ema") else None
        y = torch.as_tensor(ds.y, device=dev)
        mu = ctx.X.mean(0); sd = ctx.X.std(0).clamp(min=1e-8)
        rank_on, adv_on = bool(spec.get("rank")), bool(spec.get("adv"))
        gkey = torch.as_tensor(ds.drug_codes, device=dev) * ds.n_studies + torch.as_tensor(ds.scodes, device=dev)
        aux = E.init_aux(A, ctx.arch.feat_width, ds.n_studies, gen, dev) if adv_on else None
        aopt = E.SlotOpt(aux, spec, ctx.fused_fn) if adv_on else None

        def step(params=params, opt=opt, A=A, gen=gen, y=y, mu=mu, sd=sd, ema=ema, aux=aux, aopt=aopt):
            rows = torch.randint(0, ds.n, (A, bs), generator=gen, device=dev)
            xb = (ctx.X[rows] - mu) / sd
            mol = ctx.X.di[rows] if getattr(ctx.arch, "kind", "") == "mol" else None
            if adv_on:
                out, feat = nets.forward_batched(params, xb, ctx.arch, float(spec["dropout"]), True, return_feat=True,
                                                 mol=mol)
            else:
                out = nets.forward_batched(params, xb, ctx.arch, float(spec["dropout"]), True, mol=mol)
            mse_b = ((out - y[rows]) ** 2).mean(1)
            le = mse_b if spec.get("loss", "rmse") == "mse" else torch.sqrt(mse_b + 1e-6)
            if rank_on:                                   # (R) same cost as in training
                le = le + E.rank_loss(out, y[rows], gkey[rows], torch.ones(A, device=dev))
            if adv_on:                                    # (H)
                le = le + E.adv_loss(aux, feat, torch.full((A,), 0.1, device=dev), torch.as_tensor(ds.scodes, device=dev)[rows])
                aopt.zero_grad()
            opt.zero_grad(); le.sum().backward()
            if adv_on:
                aopt.step(float(spec["clip"]))
            opt.step(float(spec["clip"]))
            if ema is not None:
                with torch.no_grad():
                    w = E.ema_weight(opt.t, 0.999)
                    for e_, p_ in zip(ema, params):
                        e_.lerp_(p_.detach(), w.view((-1,) + (1,) * (p_.dim() - 1)))
        for _ in range(5):
            step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = n_steps if dev.type == "cuda" else max(5, n_steps // 4)
        for _ in range(n):
            step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        st.add(A, (time.perf_counter() - t0) / n * 1.08)                 # + ~8 % for the per-epoch validation pass
        del params, opt, step, ema, aux, aopt
    dist = []
    for r in S.list_runs():
        c = r["config"]
        if c.get("model") == cfg["model"] and c.get("data") == cfg["data"] and _same_recipe(c.get("spec") or {}, spec):
            f = S.folds(r["id"])
            if not f.empty:
                dist.extend(f["epochs_run"].astype(float).tolist())
    dist = dist if len(dist) >= 3 else None
    E_, pat = float(spec["epochs"]), float(spec["patience"])
    steps = math.ceil(max(len(j.tr_in) for j in jobs) / bs)
    n = len(jobs)
    exp = E.simulate_pool([], np.clip(E.expected_ends(n, dist), min(pat + 1.0, E_), E_), P, steps, st)
    mn = E.simulate_pool([], np.full(n, min(pat + 1.0, E_)), P, steps, st)
    mx = E.simulate_pool([], np.full(n, E_), P, steps, st)
    return {"device": E.gpu_info(), "optimizer": ctx.opt_name, "parallel": P, "jobs": n,
            "step_ms": {int(a): round(1000 * s, 3) for a, s in st.pts},
            "expected_s": exp, "min_s": mn, "max_s": mx,
            "epochs_basis": f"{len(dist)} earlier models (median {np.median(dist):.0f} epochs)" if dist else
            "no earlier run of this model + dataset → assumes ~140 epochs per model"}


def _benchmark_gbm(cfg):
    """LightGBM: measured seconds per boosting round on the largest fold (30 rounds) + the time to bin its data"""
    from . import gbm as G
    ds = D.load(cfg["data"])
    spec = cfg["spec"]
    jobs = E.make_jobs(ds, cfg)
    big = max(jobs, key=lambda j: len(j.tr_in))
    per, build = G.benchmark_seconds(ds.X, big, spec, int(cfg["seed"]))
    dist = []
    for r in S.list_runs():
        c = r["config"]
        if c.get("model") == cfg["model"] and c.get("data") == cfg["data"] and _same_recipe(c.get("spec") or {}, spec):
            f = S.folds(r["id"])
            if not f.empty:
                dist.extend(f["epochs_run"].astype(float).tolist())
    dist = dist if len(dist) >= 3 else None
    E_, pat = float(spec["epochs"]), float(spec["patience"])
    n = len(jobs)
    rounds = np.clip(E.expected_ends(n, dist, default=min(E_, 1000.0)), min(pat + 1.0, E_), E_)
    return {"device": {"name": f"CPU · {G.threads()} cores"}, "optimizer": f"LightGBM {G.version() or ''}",
            "parallel": 1, "jobs": n, "step_ms": {1: round(1000 * per, 3)}, "unit": "round",
            "expected_s": float(rounds.sum() * per + n * build), "min_s": float(n * (min(pat + 1.0, E_) * per + build)),
            "max_s": float(n * (E_ * per + build)),
            "epochs_basis": f"{len(dist)} earlier models (median {np.median(dist):.0f} rounds)" if dist else
            f"no earlier LightGBM run on this dataset → assumes ~{int(min(E_, 1000))} rounds per model"}
