"""Shared helpers for the pages: cached file reads (keyed on file modification time), GPU info, starting /
stopping the training process, formatting."""
import json
import os
import subprocess
import sys
import time

import numpy as np
import streamlit as st

from core import config as C
from core import data as D
from core import store as S


# ======================================================================================================================
# short-lived cache (thread-safe): Streamlit's st.cache_data with a TTL of a few seconds can raise KeyError when an
# entry expires between its own lookup steps while several fragments refresh at once
# ======================================================================================================================
def ttl_cache(seconds):
    import functools
    import threading

    def deco(fn):
        lock, store = threading.Lock(), {}

        @functools.wraps(fn)
        def wrapper(*args):
            now = time.monotonic()
            with lock:
                hit = store.get(args)
                if hit is not None and now - hit[0] < seconds:
                    return hit[1]
            val = fn(*args)
            with lock:
                store[args] = (time.monotonic(), val)
            return val

        def clear():
            with lock:
                store.clear()
        wrapper.clear = clear
        return wrapper
    return deco


# ======================================================================================================================
# formatting
# ======================================================================================================================
def hms(sec):
    if sec is None or not np.isfinite(sec):
        return "—"
    sec = int(round(max(0.0, float(sec))))
    h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d}d {h:02d}h {m:02d}m"
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def f3(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(x) else f"{x:.{nd}f}"


def pm(m, k, nd=3):
    v, s = m.get(k), m.get(k + "_sd")
    if v is None or not np.isfinite(v):
        return "—", ""
    return f"{v:.{nd}f}", (f"± {s:.{nd}f}" if s is not None and np.isfinite(s) else "")


def run_label(r):
    c = r["config"]
    tag = model_tag(c)
    data = D.dataset_name(c.get("data", "?"))
    sch = " · ".join(C.SCHEMES[s]["name"] for s in c.get("schemes", []) if s in C.SCHEMES) or "all-data model"
    mark = {"running": "●", "loading": "●", "starting": "●", "explaining": "●", "finished": "✓", "imported": "✓",
            "stopped": "■", "failed": "✕", "interrupted": "!"}.get(r["state"], "·")
    return f"{mark}  {r['id'][:15]}  ·  {tag}  ·  {data}  ·  {sch}"


def settings_chips(spec, folds=None, unit=None):
    if spec.get("arch") == C.GBM_MODEL:                     # (v5.9) LightGBM
        k = int(folds or C.N_FOLDS)
        tgt = ("target drug-wise z" if spec.get("target") == "drug_z" else
               ("target study-wise z (S)" if spec.get("target") == "study_z" else f"target raw {unit or C.UNIT}"))
        return [f"LightGBM · lr {spec.get('lr'):g}", f"max {spec.get('epochs')} rounds · early stop {spec.get('patience')}",
                f"{int(spec.get('gb_leaves', 63))} leaves · ≥ {int(spec.get('gb_min_leaf', 20))} pairs per leaf",
                f"{float(spec.get('gb_colsample', 0.5)):.0%} features · {float(spec.get('gb_subsample', 0.8)):.0%} pairs "
                f"per tree", f"L2 {float(spec.get('gb_l2', 1.0)):g}", "features raw (trees)", tgt,
                f"{spec.get('repeats')}× repeated {k}-fold" if int(spec.get("repeats", 1)) > 1 else f"{k}-fold"]
    out = [f"init {'He-normal' if spec.get('init') == 'he_normal' else 'Kaiming-uniform'}",
           f"{'AdaMax' if spec.get('optimizer') == 'adamax' else 'Adam'} · lr {spec.get('lr'):g}",
           f"batch {spec.get('batch')}", f"max {spec.get('epochs')} epochs · patience {spec.get('patience')}",
           f"features {'z-scored' if spec.get('feat_scale') else 'raw'}", f"dropout {spec.get('dropout')} · clip ±{spec.get('clip'):g}",
           f"weight decay {float(spec.get('weight_decay', 0)):g}"]
    if spec.get("arch") == "graph":
        out.append(f"pathway graph \u00b7 {spec.get('g_mode', 'hybrid')} readout \u00b7 node size {spec.get('g_dim', 32)} "
                   f"\u00b7 {spec.get('g_layers', 2)} layers \u00b7 edges Jaccard \u2265 {spec.get('g_tau', 0.05):g}")
    if spec.get("interaction"):
        out.append(f"interaction module {spec.get('d_int', 128)}")
    out.append("target drug-wise z" if spec.get("target") == "drug_z" else
               ("target study-wise z (S)" if spec.get("target") == "study_z" else f"target raw {unit or C.UNIT}"))
    k = int(folds or C.N_FOLDS)
    out.append(f"{spec.get('repeats')}× repeated {k}-fold" if int(spec.get("repeats", 1)) > 1 else f"{k}-fold")
    if spec.get("fac"):
        out.append(f"A · drug × cell factorization, rank {int(spec.get('fac_dim', 64))}")
    if spec.get("gate") and spec.get("interaction"):
        out.append("B · gated interaction")
    if spec.get("plateau"):
        pp, pd_ = int(spec.get("plateau_patience", 10)), float(spec.get("plateau_delta", 0.0))
        out.append(f"C · LR ×½ after {pp} epochs without " + (f"a {pd_:g} better val loss" if pd_ else "improvement"))
    if int(spec.get("ensemble", 1) or 1) > 1:
        out.append(f"D · {int(spec['ensemble'])}-seed ensemble")
    if spec.get("ema"):
        out.append("E · weight averaging (EMA)")
    if spec.get("loss", "rmse") == "mse":
        out.append("F · MSE training loss")
    if spec.get("val_metric", "batch") == "global":
        out.append("G · global validation RMSE")
    if spec.get("adv"):
        out.append(f"H · study-adversarial, weight {float(spec.get('adv_weight', 0.1)):g}")
    if spec.get("rank"):
        out.append(f"R · within-drug ranking loss, weight {float(spec.get('rank_weight', 0.5)):g}")
    if spec.get("mol"):
        out.append("M · learned drug graph encoder (64-d)")
    if spec.get("arch") == "graph" and spec.get("g_graph", "pid") == "random":
        out.append("control: randomly rewired pathway graph")
    return out


def model_tag(cfg):
    """PROPOSED, or PROPOSED+A·C·D×3 when upgrades are on"""
    tag = C.MODEL_SPECS.get(cfg.get("model"), {}).get("tag", "?")
    if (cfg.get("spec") or {}).get("arch") == "graph":
        tag += "\u00b7" + str((cfg.get("spec") or {}).get("g_mode", "hybrid"))
        if (cfg.get("spec") or {}).get("g_graph", "pid") == "random":
            tag += "\u00b7random-graph"
    up = C.upgrades_tag(cfg.get("spec") or {})
    return f"{tag}+{up}" if up else tag


# ======================================================================================================================
# cached reads
# ======================================================================================================================
def _mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


@st.cache_resource(show_spinner=False, max_entries=4)
def dataset(key):
    return D.load(key)


@st.cache_data(show_spinner=False)
def cell_info():
    return D.cell_info()


@st.cache_data(show_spinner=False)
def drug_info():
    return D.drug_info()


@st.cache_data(show_spinner=False, max_entries=64)
def _csv(path, mtime):
    return S.read_csv(path)


def run_csv(rid, name):
    p = S.run_dir(rid) / name
    return _csv(str(p), _mtime(p))


@st.cache_data(show_spinner=False, max_entries=32)
def _summ(rid, m1, m2, grouping):
    from core import metrics as MT
    f, p = run_csv(rid, "folds.csv"), run_csv(rid, "preds.csv")
    out = {}
    if f.empty or p.empty:
        return out
    for s in [x for x in C.SCHEMES if x in set(f["scheme"])]:
        out[s] = MT.summarize(p[p["scheme"] == s], f[f["scheme"] == s], s, grouping)
    return out


def scheme_summaries(rid):
    d = S.run_dir(rid)
    return _summ(rid, _mtime(d / "folds.csv"), _mtime(d / "preds.csv"), S.config(rid).get("grouping", "logo"))


@st.cache_data(show_spinner=False, max_entries=32)
def _conformal(rid, scheme, level, mode, mtime):
    from core import uncertainty as UQ
    p = run_csv(rid, "preds.csv")
    if p.empty:
        return p, {}
    p = p[p["scheme"] == scheme]
    iv = UQ.intervals(p, level, mode)
    return iv, UQ.summary(iv, level)


def conformal(rid, scheme, level=0.90, mode="drug"):
    """held-out predictions of one scheme with conformal intervals + their summary (cached per file version)"""
    return _conformal(rid, scheme, float(level), mode, _mtime(S.run_dir(rid) / "preds.csv"))


@st.cache_data(show_spinner=False, max_entries=16)
def _drug_widths(rid, level, mtime):
    from core import uncertainty as UQ
    p = run_csv(rid, "preds.csv")
    sch = next((s for s in ("mixed", C.CSA_SCHEME, C.LOSO_SCHEME) if not p.empty and s in set(p["scheme"])), None)
    if sch is None:
        return {}, float("nan")
    return UQ.drug_widths(p[p["scheme"] == sch], level)


def drug_widths(rid, level=0.90):
    """(q per drug, global q) for the Predict page's interval around a new prediction of this run's model (from its
    Mixed CV, else its CSA splits, else its held-out studies)"""
    return _drug_widths(rid, float(level), _mtime(S.run_dir(rid) / "preds.csv"))


@st.cache_data(show_spinner=False, max_entries=32)
def _paired(a, b, scheme, ma, mb):
    from core import compare as CP
    return CP.paired(run_csv(a, "folds.csv"), run_csv(a, "preds.csv"), run_csv(b, "folds.csv"), run_csv(b, "preds.csv"),
                     scheme)


def paired(a, b, scheme):
    """paired fold-by-fold comparison of runs a and b (B − A)"""
    return _paired(a, b, scheme, _mtime(S.run_dir(a) / "preds.csv"), _mtime(S.run_dir(b) / "preds.csv"))


@st.cache_data(show_spinner=False, max_entries=16)
def _blend(a, b, scheme, ma, mb):
    from core import blend as BL
    return BL.blend(run_csv(a, "preds.csv"), run_csv(b, "preds.csv"), scheme, S.config(a).get("grouping", "logo"))


def blend(a, b, scheme):
    """(v5.9) cross-fitted blend of runs a and b on the same held-out pairs"""
    return _blend(a, b, scheme, _mtime(S.run_dir(a) / "preds.csv"), _mtime(S.run_dir(b) / "preds.csv"))


@st.cache_resource(show_spinner=False, max_entries=6)
def _predictor(model_id, mtime):
    return S.Predictor(S.load_bundle(model_id))


def predictor(model_id):
    return _predictor(model_id, _mtime(C.MODELS / f"{model_id}.pt"))


def models():
    return S.list_models()


# ======================================================================================================================
# GPU — NVML reads the GPU without creating a CUDA context in the app process. On laptops the NVIDIA GPU powers down
# when idle and some NVML queries then fail, so every value is read separately and the name is remembered.
# ======================================================================================================================
_GPU_NAME_FILE = C.CACHE / "gpu_name.txt"


def _short(name):
    return str(name).replace("NVIDIA ", "").replace("GeForce ", "").strip()


@st.cache_resource(show_spinner=False)
def _gpu_name_fallback():
    """name of the NVIDIA GPU even when NVML cannot reach it (remembered name, then Windows' device list)"""
    try:
        if _GPU_NAME_FILE.exists():
            return _GPU_NAME_FILE.read_text(encoding="utf-8").strip() or None
    except Exception:
        pass
    if os.name == "nt":
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*NVIDIA*' } "
                                "| Select-Object -First 1).Name"], capture_output=True, text=True, timeout=8,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return r.stdout.strip() or None
        except Exception:
            return None
    return None


@ttl_cache(2)
def gpu():
    """{'name', 'total_mb', 'used_mb', 'free_mb', 'util', 'asleep'} or None when the machine has no NVIDIA GPU"""
    out = {"name": None, "total_mb": None, "used_mb": None, "free_mb": None, "util": None, "asleep": False}
    try:
        import pynvml as N
        N.nvmlInit()
        h = N.nvmlDeviceGetHandleByIndex(0)
        try:
            nm = N.nvmlDeviceGetName(h)
            out["name"] = nm.decode() if isinstance(nm, bytes) else nm
        except Exception:
            pass
        try:
            mem = N.nvmlDeviceGetMemoryInfo(h)
            out.update(total_mb=mem.total / 2 ** 20, used_mb=mem.used / 2 ** 20, free_mb=mem.free / 2 ** 20)
        except Exception:
            out["asleep"] = True
        try:
            out["util"] = int(N.nvmlDeviceGetUtilizationRates(h).gpu)
        except Exception:
            out["asleep"] = True
    except Exception:
        out["asleep"] = True
    if out["name"]:
        try:
            if not _GPU_NAME_FILE.exists() or _GPU_NAME_FILE.read_text(encoding="utf-8").strip() != out["name"]:
                _GPU_NAME_FILE.write_text(out["name"], encoding="utf-8")
        except Exception:
            pass
    else:
        out["name"] = _gpu_name_fallback()
    if not out["name"]:
        return None
    out["name"] = _short(out["name"])
    return out


def gpu_state(active_run=None):
    """('in use' | 'idle' | 'none', text)"""
    g = gpu()
    if not g:
        return "none", "No NVIDIA GPU found — training would run on the CPU"
    if active_run is not None:
        tag = C.MODEL_SPECS.get(active_run["config"].get("model"), {}).get("tag", "")
        n = (active_run["status"].get("pool") or {}).get("active")
        return "in use", f"GPU in use — training {tag}" + (f" · {n} models on the GPU" if n else "")
    if g.get("util") is not None and g["util"] >= 10:
        return "in use", f"GPU in use by another program ({g['util']} %)"
    return "idle", "GPU idle — ready to train" + (" (powered down to save energy)" if g.get("asleep") else "")


def gpu_chip():
    g = gpu()
    if not g:
        return '<span class="nt-chip dim">GPU · none found</span>'
    parts = [g["name"]]
    if g.get("used_mb") is not None:
        parts.append(f'{g["used_mb"] / 1024:.1f}/{g["total_mb"] / 1024:.1f} GB')
    if g.get("util") is not None:
        parts.append(f'{g["util"]}%')
    return f'<span class="nt-chip"><span class="nt-dot white"></span>{" · ".join(parts)}</span>'


# ======================================================================================================================
# Triton (optional fused optimizer)
# ======================================================================================================================
_TRITON_LOG = C.CACHE / "triton_install.log"
_TRITON_JOB = C.CACHE / "triton_install.json"


def triton_info():
    import importlib.metadata as md
    import importlib.util
    info = {"installed": importlib.util.find_spec("triton") is not None, "version": None, "dist": None}
    for dist in ("triton-windows", "triton"):
        try:
            info["version"], info["dist"] = md.version(dist), dist
            break
        except Exception:
            continue
    try:
        info["torch"] = md.version("torch")
    except Exception:
        info["torch"] = None
    return info


def triton_requirement():
    """pip requirement of the Triton build that matches the installed PyTorch: torch pins its Triton version in its own
    metadata; Windows uses the community 'triton-windows' build of the same Triton minor version."""
    import importlib.metadata as md
    import re
    pin = None
    try:
        for r in md.requires("torch") or []:
            m = re.match(r"^\s*triton\s*(?:==|~=|>=)\s*([0-9]+\.[0-9]+)", r)
            if m:
                pin = m.group(1)
                break
    except Exception:
        pass
    if pin is None:                                         # releases whose pairing is known
        known = {"2.6": "3.2", "2.7": "3.3", "2.8": "3.4", "2.9": "3.5", "2.14": "3.8"}
        try:
            pin = known.get(".".join(md.version("torch").split("+")[0].split(".")[:2]))
        except Exception:
            pass
    if os.name == "nt":
        if pin:
            maj, mi = pin.split(".")[:2]
            return f"triton-windows>={maj}.{mi},<{maj}.{int(mi) + 1}", pin
        return "triton-windows", None
    return (f"triton=={pin}.*" if pin else "triton"), pin


def triton_job():
    """state of a Triton install started from the app: None | {'state': 'running'|'done'|'failed', ...}"""
    job = S.read_json(_TRITON_JOB, None)
    if not job:
        return None
    if job.get("state") == "running" and not S.pid_alive(job.get("pid")):
        txt = _TRITON_LOG.read_text(encoding="utf-8", errors="replace") if _TRITON_LOG.exists() else ""
        job["state"] = "done" if "Successfully installed" in txt or "Requirement already satisfied" in txt else "failed"
        D.write_json(_TRITON_JOB, job)
    return job


def start_triton_install():
    req, _ = triton_requirement()
    log = open(_TRITON_LOG, "wb")
    kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
    p = subprocess.Popen([sys.executable, "-m", "pip", "install", "--upgrade", req], cwd=str(C.ROOT), stdout=log,
                         stderr=subprocess.STDOUT, **kw)
    log.close()
    D.write_json(_TRITON_JOB, {"state": "running", "pid": p.pid, "requirement": req, "started": time.time()})
    return req


_GBM_JOB = C.CACHE / "lightgbm_install.json"
_GBM_LOG = C.CACHE / "lightgbm_install.log"


def gbm_job():
    """(v5.9) state of a LightGBM install started from the app: None | {'state': 'running'|'done'|'failed', ...}"""
    job = S.read_json(_GBM_JOB, None)
    if not job:
        return None
    if job.get("state") == "running" and not S.pid_alive(job.get("pid")):
        txt = _GBM_LOG.read_text(encoding="utf-8", errors="replace") if _GBM_LOG.exists() else ""
        job["state"] = "done" if "Successfully installed" in txt or "Requirement already satisfied" in txt else "failed"
        D.write_json(_GBM_JOB, job)
    return job


def start_gbm_install():
    log = open(_GBM_LOG, "wb")
    kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
    p = subprocess.Popen([sys.executable, "-m", "pip", "install", "--upgrade", "lightgbm>=4.0"], cwd=str(C.ROOT),
                         stdout=log, stderr=subprocess.STDOUT, **kw)
    log.close()
    D.write_json(_GBM_JOB, {"state": "running", "pid": p.pid, "requirement": "lightgbm>=4.0", "started": time.time()})


def gbm_log_tail(n=12):
    if not _GBM_LOG.exists():
        return ""
    return "\n".join(_GBM_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def triton_log_tail(n=12):
    if not _TRITON_LOG.exists():
        return ""
    return "\n".join(_TRITON_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def check_fused():
    """runs the fused-optimizer self-test on the GPU in a separate process (≈ 20–60 s the first time)"""
    kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, "-u", str(C.ROOT / "train.py"), "--check-fused"], cwd=str(C.ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, env=env, **kw)
    for line in r.stdout.splitlines()[::-1]:
        if line.startswith("FUSED "):
            out = json.loads(line[len("FUSED "):])
            D.write_json(C.CACHE / "fused_check.json", {**out, "checked": time.time()})
            return out
    raise RuntimeError((r.stderr or r.stdout)[-1500:])


def fused_check_result():
    return S.read_json(C.CACHE / "fused_check.json", None)


# ======================================================================================================================
# training process
# ======================================================================================================================
def _spawn(args, rid):
    log = open(S.run_dir(rid) / "log.txt", "ab")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    p = subprocess.Popen([sys.executable, "-u", str(C.ROOT / "train.py"), *args], cwd=str(C.ROOT), stdout=log,
                         stderr=subprocess.STDOUT, env=env, **kw)
    log.close()                                             # the child keeps its own handle
    st_ = S.status(rid)
    st_.update(pid=p.pid, state="starting", message="Starting the training process…", updated=time.time())
    st_.setdefault("started", time.time())
    D.write_json(S.run_dir(rid) / "status.json", st_)
    return p.pid


def start_run(cfg):
    rid = S.new_run_id(cfg["model"], cfg["data"])
    cfg = {**cfg, "run_id": rid}
    S.write_config(rid, cfg)
    try:
        os.remove(S.run_dir(rid) / C.STOP_FILE)
    except OSError:
        pass
    _spawn(["--run", rid], rid)
    return rid


def resume_run(rid):
    try:
        os.remove(S.run_dir(rid) / C.STOP_FILE)
    except OSError:
        pass
    _spawn(["--resume", rid], rid)


def stop_run(rid):
    (S.run_dir(rid) / C.STOP_FILE).write_text("stop", encoding="utf-8")


def benchmark(cfg):
    p = C.CACHE / "benchmark_config.json"
    D.write_json(p, cfg)
    kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, "-u", str(C.ROOT / "train.py"), "--benchmark", "--config-json", str(p)],
                       cwd=str(C.ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=900, env=env, **kw)
    for line in r.stdout.splitlines()[::-1]:
        if line.startswith("BENCHMARK "):
            return json.loads(line[len("BENCHMARK "):])
    raise RuntimeError((r.stderr or r.stdout)[-1500:])


def log_tail(rid, n=40):
    p = S.run_dir(rid) / "log.txt"
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 20000))
        return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-n:])
