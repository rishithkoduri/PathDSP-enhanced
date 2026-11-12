"""
Final validation plan (v6.0) — runs everything in order, one command:

  1  the recipe on FRESH folds        Paper exact vs Paper exact + C·D×3·E·F·G·Z, input.txt, mixed 10-fold, seed 43
  2  leakage audit                    checks on step 1's files + four leak demonstrations (CPU)
  3  explanation check                do the explained pathways contain the drug's targets? (+ deletion, stability)
  4  unseen cell lines / drugs        grouped 10-fold LOCO and LODO: Paper exact vs the recipe, with baselines
  5  finalise                         fixes the final recipe from steps 1–4
  6  train-on-everything models       IC50: processed_data (all 40,365 pairs); AAC: CSA · all 5 studies, Run A and
                                      Run B recipes — the models Predict uses
  7  final/FINAL_SUMMARY.md + final/final_results.zip (send that zip)

  python tools/final_plan.py            run (or continue) the plan — safe to stop at any time and start again
  python tools/final_plan.py --plan     show every training run's settings, train nothing
  python tools/final_plan.py --status   show what is done

Training runs are ordinary app runs (runs/<id>/): the app's Live and Results pages show them while they train.
Stop a training with Live → Stop (or close this window); starting this script again continues where it stopped.
Do not start other trainings from the app while the plan runs.
"""
import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import train as T                     # noqa: E402  (make_config)
from core import config as C         # noqa: E402
from core import data as D           # noqa: E402
from core import store as S          # noqa: E402

import final_analysis as FA          # noqa: E402

STATE = FA.OUT / "state.json"
LOG = FA.OUT / "plan_log.txt"
INPROCESS = False                     # tests only: train in this process instead of a train.py child process
OVERRIDES = {}                        # tests only: spec overrides applied to every training run
RUNNING = ("starting", "loading", "running", "explaining")
STALE = 900                           # a run whose status.json has not changed for 15 min is not training (after a
                                      # shutdown Windows can give its old process id to another program)


def is_training(st_):
    try:
        fresh = time.time() - float(st_.get("updated", 0)) < STALE
    except (TypeError, ValueError):
        fresh = False
    return S.live_state(st_) in RUNNING and fresh

# ----------------------------------------------------------------------------------------------------------------------
# the training runs (settings as in the report: Paper exact = Table S2 recipe, batch 8; recipe = C·D×3·E·F·G·Z)
# ----------------------------------------------------------------------------------------------------------------------
RECIPE = dict(plateau=True, plateau_patience=5, plateau_delta=1e-3, ensemble=3, ema=True, loss="mse",
              val_metric="global", target="drug_z")
CSA_A = dict(epochs=300, patience=20, batch=64, target="raw", plateau=True, plateau_patience=5, plateau_delta=1e-3,
             ema=True, loss="mse", val_metric="global", g_mode="hybrid", g_dim=32, g_layers=2, g_tau=0.05,
             g_graph="pid", rank=False, mol=False, adv=False, ensemble=1, repeats=1)
CSA_B = dict(CSA_A, target="study_z", rank=True, rank_weight=0.5, mol=True)


def cfg_of(model, data, schemes, seed=42, final=False, grouping="logo", **spec):
    cfg = T.make_config(model, data, schemes, grouping, seed, final, "auto", 10, "max")
    cfg["spec"].update({"repeats": 1, "fused": True, **spec, **OVERRIDES})
    return cfg


TRAIN = {
    "1-plain": ("Step 1 · Paper exact, seed 43 (fresh folds)",
                lambda st: cfg_of("paper", "input", ["mixed"], seed=43)),
    "1-recipe": ("Step 1 · Paper exact + C·D×3·E·F·G·Z, seed 43 (fresh folds)",
                 lambda st: cfg_of("paper", "input", ["mixed"], seed=43, **RECIPE)),
    "4-plain": ("Step 4 · Paper exact, LOCO + LODO (grouped 10-fold)",
                lambda st: cfg_of("paper", "input", ["LOCO", "LODO"], grouping="kfold")),
    "4-recipe-LOCO": ("Step 4 · Paper exact + C·D×3·E·F·G·Z, LOCO (grouped 10-fold)",
                      lambda st: cfg_of("paper", "input", ["LOCO"], grouping="kfold", **RECIPE)),
    "4-recipe-LODO": ("Step 4 · Paper exact + C·D×3·E·F·G (raw target), LODO (grouped 10-fold)",
                      lambda st: cfg_of("paper", "input", ["LODO"], grouping="kfold", **dict(RECIPE, target="raw"))),
    "F1-ic50": ("Final · IC50 model on ALL of processed_data (the decided recipe)",
                lambda st: cfg_of("paper", "processed", [], final=True,
                                  **(RECIPE if (st.get("results", {}).get("finalise") or {}).get("final_ic50", {})
                                     .get("recipe") == "recipe" else {}))),
    "F2-aac-A": ("Final · AAC model on ALL 5 CSA studies, Run A recipe (raw AAC)",
                 lambda st: cfg_of("graph", "csa-ALL", [], final=True, **CSA_A)),
    "F3-aac-B": ("Final · AAC model on ALL 5 CSA studies, Run B recipe (S + R + M)",
                 lambda st: cfg_of("graph", "csa-ALL", [], final=True, **CSA_B)),
}

ORDER = [("train", "1-plain"), ("train", "1-recipe"), ("analyse", "step1"), ("analyse", "step2"),
         ("analyse", "step3"), ("train", "4-plain"), ("train", "4-recipe-LOCO"), ("train", "4-recipe-LODO"),
         ("analyse", "step4"), ("analyse", "finalise"), ("train", "F1-ic50"), ("train", "F2-aac-A"),
         ("train", "F3-aac-B"), ("analyse", "summary")]

ROUGH_HOURS = {"1-plain": "≈1–1.5 h", "1-recipe": "≈1.5–2 h", "step1": "seconds", "step2": "≈0.5 h (CPU)",
               "step3": "≈10–30 min", "4-plain": "≈2–3 h", "4-recipe-LOCO": "≈1.5–2 h", "4-recipe-LODO": "≈1.5–2 h",
               "step4": "seconds", "finalise": "seconds", "F1-ic50": "≈0.5–1 h", "F2-aac-A": "≈2–3 h",
               "F3-aac-B": "≈0.5–1.5 h", "summary": "seconds"}


# ----------------------------------------------------------------------------------------------------------------------
# state, log
# ----------------------------------------------------------------------------------------------------------------------
def load_state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {"created": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "runs": {}, "done": [], "results": {}}


def save_state(st):
    FA.OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, STATE)


def say(msg):
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        FA.OUT.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _hms(sec):
    sec = int(max(0, sec or 0))
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


class PlanStopped(Exception):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------------------------------------
def _spawn(args, rid):
    log = open(S.run_dir(rid) / "log.txt", "ab")
    kw = {}
    if os.name == "nt":                                   # own process group: Ctrl+C here does not kill the training
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    p = subprocess.Popen([sys.executable, "-u", str(ROOT / "train.py"), *args], cwd=str(ROOT), stdout=log,
                         stderr=subprocess.STDOUT, env=env, **kw)
    log.close()
    st_ = S.status(rid)
    st_.update(pid=p.pid, state="starting", message="Starting the training process…", updated=time.time())
    st_.setdefault("started", time.time())
    D.write_json(S.run_dir(rid) / "status.json", st_)
    return p


def _wait(rid, label, proc=None):
    last = 0.0
    while True:
        st_ = S.status(rid)
        state = S.live_state(st_)
        alive = proc.poll() is None if proc is not None else is_training(st_)
        if time.time() - last >= 60 or not alive:
            last = time.time()
            pool = st_.get("pool") or {}
            eta = st_.get("eta") or {}
            left = eta.get("expected") if isinstance(eta, dict) else None
            say(f"  {label}: {state} · models {st_.get('done_jobs', 0)}/{st_.get('n_jobs', '?')} · "
                f"epoch {pool.get('epoch_max', '–')} · elapsed {_hms(st_.get('elapsed'))}"
                + (f" · about {_hms(left)} left" if isinstance(left, (int, float)) else "")
                + (f" · {st_.get('message')}" if st_.get("message") and state != "running" else ""))
        if not alive:
            break
        time.sleep(5)
    time.sleep(1)
    st_ = S.status(rid)
    state = st_.get("state")
    if state == "finished" and S.summary(rid):
        return
    if state == "stopped":
        raise PlanStopped(f"{label} was stopped. Run final_plan again to continue it.")
    err = st_.get("error") or st_.get("message") or f"the training process ended in state '{state}'"
    raise RuntimeError(f"{label} failed: {err}\nSee runs/{rid}/log.txt")


def train_step(key, st):
    label, builder = TRAIN[key]
    rid = st["runs"].get(key)
    if rid and FA.run_finished(rid):
        say(f"✓ {label} — already done ({rid})")
        return rid
    if rid and S.config(rid):
        training = is_training(S.status(rid))
        try:
            os.remove(S.run_dir(rid) / C.STOP_FILE)
        except OSError:
            pass
        if training:
            say(f"… {label} is already training ({rid}) — waiting for it")
            _wait(rid, label)
            say(f"✓ {label} — finished ({_hms(S.status(rid).get('elapsed'))})")
            return rid
        say(f"▶ {label} — continuing {rid}")
        if INPROCESS:
            cfg = S.config(rid); cfg["resumed"] = True; S.write_config(rid, cfg)
            _inprocess(rid, label)
        else:
            _wait(rid, label, _spawn(["--resume", rid], rid))
        say(f"✓ {label} — finished ({_hms(S.status(rid).get('elapsed'))})")
        return rid
    from core import runner as R
    cfg = builder(st)
    err = R.validate(cfg)
    if err:
        raise RuntimeError(f"{label}: {err}")
    rid = f"{S.new_run_id(cfg['model'], cfg['data'])}_{key}"
    cfg["run_id"] = rid
    cfg["final_plan"] = key
    S.write_config(rid, cfg)
    st["runs"][key] = rid
    save_state(st)
    say(f"▶ {label} — new run {rid} ({C.upgrades_tag(cfg['spec']) or 'no upgrades'}, schemes "
        f"{', '.join(cfg['schemes']) or 'all-data model only'}; rough time {ROUGH_HOURS.get(key, '?')})")
    if INPROCESS:
        _inprocess(rid, label)
    else:
        _wait(rid, label, _spawn(["--run", rid], rid))
    say(f"✓ {label} — finished ({_hms(S.status(rid).get('elapsed'))})")
    return rid


def _inprocess(rid, label):
    from core import runner as R
    R.Run(rid).run()
    if S.status(rid).get("state") != "finished":
        raise RuntimeError(f"{label} ended in state {S.status(rid).get('state')}")


# ----------------------------------------------------------------------------------------------------------------------
# analysis steps
# ----------------------------------------------------------------------------------------------------------------------
def analyse_step(name, st, no_lgbm=False):
    if name in st["done"]:
        say(f"✓ {name} — already done")
        return
    runs = st["runs"]
    say(f"▶ {name} …")
    t0 = time.time()
    if name == "step1":
        r = FA.step1(runs["1-plain"], runs["1-recipe"])
        c = r["comparison"]
        say(f"  recipe vs plain on fresh folds: ΔRMSE {c['mean_dRMSE']:+.4f}, recipe better on {c['B_better']}/{c['n']} "
            f"folds, Wilcoxon p = {FA._p(c['p_wilcoxon'])} → {'CONFIRMED' if r['confirmed'] else 'NOT confirmed'}")
    elif name == "step2":
        r = FA.step2(runs["1-recipe"], st["results"].get("step1"), with_lgbm=not no_lgbm, progress=say)
        say(f"  checks passed: {r['checks_passed']}/{r['checks']}")
    elif name == "step3":
        r = FA.step3(runs["1-recipe"], progress=say)
        rec = r.get("recovery") or {}
        if rec:
            say(f"  target recovery: AUROC {rec['AUROC_cell']:.3f} (random 0.5), P@10 {rec['P@10']:.3f} "
                f"(random {rec['P@10_random']:.3f})")
    elif name == "step4":
        r = FA.step4(runs["4-plain"], runs["4-recipe-LOCO"], runs["4-recipe-LODO"])
        for s_ in ("LOCO", "LODO"):
            c = r[s_]["comparison"]
            say(f"  {s_}: recipe vs plain ΔRMSE {c['mean_dRMSE']:+.4f}, better on {c['B_better']}/{c['n']} folds")
    elif name == "finalise":
        r = FA.finalise(st)
        say(f"  final IC50 model: {r['final_ic50']['label']} on processed_data · GDSC")
    elif name == "summary":
        p = FA.final_summary(st)
        z = FA.pack()
        r = {"summary": str(p), "zip": str(z)}
        say(f"  {p}\n  {z}  ← send this file")
    else:
        raise ValueError(name)
    st["results"][name] = r
    st["done"].append(name)
    save_state(st)
    say(f"✓ {name} — done in {_hms(time.time() - t0)}")


# ----------------------------------------------------------------------------------------------------------------------
def show_plan(st):
    print("Final validation plan — order of work\n")
    for kind, key in ORDER:
        if kind == "train":
            label, builder = TRAIN[key]
            rid = st["runs"].get(key)
            done = "done" if rid and FA.run_finished(rid) else ("started" if rid else "to do")
            try:
                cfg = builder(st)
                sp = cfg["spec"]
                det = (f"model {cfg['model']}, data {cfg['data']}, schemes {cfg['schemes'] or ['all-data model']}"
                       f"{' (grouped 10-fold)' if cfg['grouping'] == 'kfold' else ''}, seed {cfg['seed']}, batch "
                       f"{sp['batch']}, epochs ≤ {sp['epochs']}, patience {sp['patience']}, target {sp['target']}, "
                       f"upgrades {C.upgrades_tag(sp) or 'none'}")
                if key == "F1-ic50" and "finalise" not in st["done"]:
                    det += " — the recipe is decided in step 5 (plain Paper exact shown until then)"
            except Exception as e:
                det = f"(settings decided later: {e})"
            print(f"  [{done:7}] {label}\n            {det}\n            rough time {ROUGH_HOURS.get(key)}"
                  + (f" · run {rid}" if rid else ""))
        else:
            print(f"  [{'done' if key in st['done'] else 'to do':7}] analysis: {key} ({ROUGH_HOURS.get(key)})")
    print("\nTotal roughly 12–15 h on the RTX 4060. It can be stopped and continued at any time.")


def main():
    for stream in (sys.stdout, sys.stderr):                 # ✓ ▶ — on any Windows console or redirected output
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="PathDSP final validation plan (v6.0)")
    ap.add_argument("--plan", action="store_true", help="show the plan and every run's settings; train nothing")
    ap.add_argument("--status", action="store_true", help="show what is done")
    ap.add_argument("--redo", nargs="+", default=[], help="analysis steps to compute again (step1 … summary)")
    ap.add_argument("--no-lgbm", action="store_true", help="step 2 without the LightGBM early-stopping demo")
    ap.add_argument("--allow-cpu", action="store_true", help="run even without a CUDA GPU (very slow)")
    a = ap.parse_args()
    st = load_state()
    if a.plan or a.status:
        show_plan(st)
        return
    for name in a.redo:
        if name in st["done"]:
            st["done"].remove(name)
    if a.redo:
        save_state(st)
    if not INPROCESS and not a.allow_cpu:
        import torch
        if not torch.cuda.is_available():
            sys.exit("No CUDA GPU visible to PyTorch — run start.bat (it installs the CUDA build), or pass --allow-cpu.")
        say(f"GPU: {torch.cuda.get_device_name(0)} · PyTorch {torch.__version__}")
    avail = D.available()
    for need in ("input", "processed", "csa-ALL"):
        if not avail.get(need):
            sys.exit(f"Dataset '{need}' is missing from data/ — the plan needs input.zip, processed_data_GDSC.zip "
                     f"and data/csa/.")
    say("Final validation plan started (python tools/final_plan.py --plan shows every step)")
    try:
        for kind, key in ORDER:
            if kind == "train":
                train_step(key, st)
            else:
                analyse_step(key, st, a.no_lgbm)
        say("ALL DONE. Results: final/FINAL_SUMMARY.md · send final/final_results.zip")
    except PlanStopped as e:
        say(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        say("Interrupted. A training that was running keeps running; start final_plan again to continue.")
        sys.exit(1)
    except Exception as e:
        say(f"STOPPED WITH AN ERROR: {e}")
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(traceback.format_exc() + "\n")
        sys.exit(2)


if __name__ == "__main__":
    main()
