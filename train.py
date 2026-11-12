"""
Training worker — the app starts it in the background; you can also use it directly:

  python train.py --model paper --data input --schemes mixed            new run with the model's default settings
  python train.py --model repo --data processed --schemes mixed LOCO LODO --folds 5
  python train.py --run <run_id>                                        a run prepared by the app
  python train.py --resume <run_id>                                     continue a stopped / interrupted run
  python train.py --benchmark --model proposed --data input             measured time estimate (JSON)
  python train.py --check-fused                                         test the optional Triton fused optimizer
  python train.py --model proposed --fac --gate --plateau --ensemble 3  upgrades A, B, C, D (each optional)
  python train.py --model graph --data csa-CTRPv2 --schemes csa --cross  CSA benchmark: official splits + cross-study
  python train.py --model graph --data csa-ALL --schemes loso --target study_z --rank --adv   multi-study, leave-study-out
  python train.py --model paper --data csa-ALL --schemes loso --target study_z --rank --mol   PathDSP-X v2 recipe (S + R + M)
  python train.py --model lgbm --data input --schemes mixed --target drug_z                LightGBM (CPU, v5.9)
"""
import argparse
import datetime as _dt
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", message=".*triton not found.*")

from core import config as C          # noqa: E402
from core import store as S           # noqa: E402


def make_config(model, data, schemes, grouping="logo", seed=42, final=True, parallel="auto", folds=C.N_FOLDS,
                perf="max", cross=False, **overrides):
    spec = dict(C.MODEL_SPECS[model]["defaults"])
    spec.update({k: v for k, v in overrides.items() if v is not None})
    return {"model": model, "data": data, "schemes": list(schemes), "grouping": grouping, "seed": int(seed),
            "folds": int(folds), "final": bool(final), "parallel": parallel, "perf": perf, "spec": spec,
            "cross": bool(cross),
            "created": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def high_priority():
    """max-performance mode: raise the training process's CPU priority so the GPU is never kept waiting"""
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS if os.name == "nt" else -5)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="PathDSP training worker")
    ap.add_argument("--run"); ap.add_argument("--resume"); ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--check-fused", action="store_true", help="test the Triton fused optimizer on this GPU")
    ap.add_argument("--config-json", help="config file (benchmark from the app)")
    ap.add_argument("--model", choices=C.MODEL_KEYS, default="paper")
    ap.add_argument("--data", default="input", help="input | processed | a dataset key shown on the Train page")
    ap.add_argument("--schemes", nargs="+", default=["mixed"], choices=list(C.SCHEMES))
    ap.add_argument("--grouping", choices=["logo", "kfold"], default="logo")
    ap.add_argument("--folds", type=int, default=C.N_FOLDS, help="k of k-fold CV (paper: 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-final", action="store_true", help="skip the all-data model")
    ap.add_argument("--parallel", default="auto", help="'auto' or number of fold-models trained at once")
    ap.add_argument("--perf", choices=["max", "balanced"], default="max")
    ap.add_argument("--epochs", type=int); ap.add_argument("--patience", type=int); ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float); ap.add_argument("--repeats", type=int)
    ap.add_argument("--target", choices=["raw", "drug_z", "study_z"])
    ap.add_argument("--fac", action="store_true", help="A: drug × cell factorization term (Proposed)")
    ap.add_argument("--gate", action="store_true", help="B: gated interaction module (Proposed)")
    ap.add_argument("--plateau", action="store_true", help="C: halve the LR after 10 epochs without improvement")
    ap.add_argument("--ensemble", type=int, help="D: seeds per fold (1 = off)")
    ap.add_argument("--cross", action="store_true", help="CSA studies: cross-study test of every official-split model")
    ap.add_argument("--rank", action="store_true", help="R: within-drug ranking loss")
    ap.add_argument("--adv", action="store_true", help="H: study-adversarial training (csa-ALL)")
    ap.add_argument("--random-graph", action="store_true", help="graph model on a randomly rewired pathway graph")
    ap.add_argument("--mol", action="store_true", help="M: learned drug graph encoder (CSA datasets)")
    a = ap.parse_args()

    if a.check_fused:
        import torch
        from core import engine as E
        dev = E.device()
        out = {"cuda": dev.type == "cuda", "triton": E.triton_available(), "torch": torch.__version__}
        for kind in ("adamax", "adam"):
            fn, name = E.fused_kernel(kind, dev)
            out[kind] = name
        print("FUSED " + json.dumps(out), flush=True)
        return

    from core import runner as R

    def cfg_from_args():
        return make_config(a.model, a.data, a.schemes, a.grouping, a.seed, not a.no_final, a.parallel, a.folds, a.perf,
                           cross=a.cross, epochs=a.epochs, patience=a.patience, batch=a.batch, lr=a.lr, repeats=a.repeats,
                           target=a.target, fac=(True if a.fac and a.model == "proposed" else None),
                           gate=(True if a.gate and a.model == "proposed" else None),
                           plateau=(True if a.plateau else None), ensemble=a.ensemble,
                           rank=(True if a.rank else None), adv=(True if a.adv else None),
                           mol=(True if a.mol else None),
                           g_graph=("random" if a.random_graph else None))

    if a.benchmark:
        if a.config_json:
            with open(a.config_json, encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = cfg_from_args()
        print("BENCHMARK " + json.dumps(R.benchmark(cfg), default=str), flush=True)
        return

    if a.resume:
        rid = a.resume
        cfg = S.config(rid)
        if not cfg:
            sys.exit(f"run {rid} not found")
        cfg["resumed"] = True
        S.write_config(rid, cfg)
    elif a.run:
        rid = a.run
        cfg = S.config(rid)
        if not cfg:
            sys.exit(f"run {rid} has no config.json")
    else:
        cfg = cfg_from_args()
        err = R.validate(cfg)
        if err:
            sys.exit(err)
        rid = S.new_run_id(a.model, a.data)
        cfg["run_id"] = rid
        S.write_config(rid, cfg)
        print(f"run {rid}", flush=True)
    if cfg.get("perf", "max") == "max":
        high_priority()
    R.Run(rid).run()
    print(f"{S.status(rid).get('state', 'ended')}: {rid}", flush=True)


if __name__ == "__main__":
    main()
