"""
Checks everything PathDSP-X needs and installs what is missing - run by start.bat inside .venv.

  1. packages     the app packages (requirements.txt) and PyTorch (CUDA build when an NVIDIA GPU is present)
  2. project      app files, reference tables, trained models, datasets in data/
  3. verify       imports every package once more and prints one line per item

Fast start: after a full check passes, the result is saved in .venv/pathdspx_check.json. The next start only
compares the installed package versions with that file (well under a second) and skips the slow parts
(GPU check, PyTorch test, importing every package). Any change - a package added, removed or upgraded, another
Python - runs the full check again. Force a full check with:  start.bat full

Exit code 0 = ready to start, 1 = something required is still missing.
"""
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
OK, MISS, WARN = "  [ OK ]", "  [MISS]", "  [WARN]"
STAMP = Path(sys.prefix) / "pathdspx_check.json"      # inside .venv: deleting .venv also forgets the check
STAMP_FORMAT = 1

# import name -> pip requirement (kept in step with requirements.txt)
PACKAGES = [("streamlit", "streamlit>=1.50"), ("numpy", "numpy>=1.26"), ("pandas", "pandas>=2.1"),
            ("sklearn", "scikit-learn>=1.4"), ("scipy", "scipy>=1.11"), ("altair", "altair>=5.4"),
            ("shap", "shap>=0.46"), ("psutil", "psutil>=5.9"), ("pynvml", "nvidia-ml-py>=12.535"),
            ("lightgbm", "lightgbm>=4.0")]
# pip (distribution) names whose installed versions make up the fingerprint
DISTS = ["streamlit", "numpy", "pandas", "scikit-learn", "scipy", "altair", "shap", "psutil", "nvidia-ml-py",
         "lightgbm", "torch"]
TORCH_INDEXES = ["https://download.pytorch.org/whl/cu130", "https://download.pytorch.org/whl/cu128"]


def say(text=""):
    print(text, flush=True)


def has(mod):
    return importlib.util.find_spec(mod) is not None


def pip(*args, quiet=True):
    cmd = [PY, "-m", "pip", "install", "--disable-pip-version-check", "--no-input"] + (["-q"] if quiet else []) + list(args)
    return subprocess.call(cmd) == 0


# ---------------------------------------------------------------------- fast-start memory
def fingerprint():
    """Python + installed version of every package; read from the package metadata, nothing is imported"""
    vers = {}
    for d in DISTS:
        try:
            vers[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            vers[d] = None
    return {"format": STAMP_FORMAT, "python": sys.version.split()[0], "exe": PY, "packages": vers}


def read_stamp():
    try:
        return json.loads(STAMP.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_stamp(torch_info):
    data = {"fp": fingerprint(), "torch": torch_info, "checked": time.strftime("%d-%m-%Y %H:%M")}
    try:
        STAMP.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass


def forget_stamp():
    try:
        STAMP.unlink()
    except OSError:
        pass


def fast_ok(stamp):
    """True when the last full check passed and nothing it checked has changed since"""
    if not stamp or stamp.get("fp") != fingerprint() or not isinstance(stamp.get("torch"), dict):
        return False
    return all(has(m) for m, _ in PACKAGES) and has("torch")


# ---------------------------------------------------------------------- full check
def nvidia_gpu():
    """True when an NVIDIA driver answers (decides the CUDA or the CPU build of PyTorch)"""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        return subprocess.call([smi, "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20) == 0
    except Exception:
        return False


def torch_state():
    """(installed, cuda available, version, gpu name) - checked in a fresh process"""
    code = ("import torch,sys; ok=torch.cuda.is_available(); "
            "print(torch.__version__); print(ok); print(torch.cuda.get_device_name(0) if ok else '')")
    try:
        out = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=180)
    except Exception:
        return False, False, "", ""
    if out.returncode != 0:
        return False, False, "", ""
    v, ok, name = (out.stdout.strip().splitlines() + ["", "", ""])[:3]
    return True, ok == "True", v, name


def torch_line(v, cuda, name, cpu_note):
    return f"{OK} PyTorch {v:<16}" + (f"GPU: {name}" if cuda else cpu_note)


def install_packages():
    missing = [(m, req) for m, req in PACKAGES if not has(m)]
    if not missing:
        say(f"{OK} app packages            all {len(PACKAGES)} installed")
        return True
    say(f"  installing {len(missing)} missing package(s) quietly ...")
    pip("--upgrade", "pip")
    for k, (mod, req) in enumerate(missing, 1):
        print(f"    [{k}/{len(missing)}] {req:<24}", end=" ", flush=True)
        good = pip(req)
        say("done" if good else "FAILED")
        if not good:
            return False
    importlib.invalidate_caches()
    return True


def install_torch(gpu):
    """returns {"version", "cuda", "gpu"} when PyTorch works, else None"""
    have, cuda, v, name = torch_state()
    if have and (cuda or not gpu):
        say(torch_line(v, cuda, name, "CPU only (no NVIDIA GPU found)"))
        return {"version": v, "cuda": cuda, "gpu": name}
    if not gpu:
        say("  installing PyTorch (CPU build) - download progress below ...")
        if not pip("torch", quiet=False):
            return None
    else:
        if have:
            say("  PyTorch is installed without CUDA - replacing it with the CUDA build ...")
            subprocess.call([PY, "-m", "pip", "uninstall", "-y", "-q", "torch"])
        for idx in TORCH_INDEXES:
            say(f"  installing PyTorch with CUDA (about 2-3 GB, a few minutes) from {idx.rsplit('/', 1)[-1]} ...")
            if pip("torch", "--index-url", idx, "--extra-index-url", "https://pypi.org/simple", quiet=False):
                have, cuda, v, name = torch_state()
                if cuda:
                    break
                subprocess.call([PY, "-m", "pip", "uninstall", "-y", "-q", "torch"])
    importlib.invalidate_caches()
    have, cuda, v, name = torch_state()
    if not have:
        return None
    say(torch_line(v, cuda, name, "CPU only - training is slow, Predict works"))
    return {"version": v, "cuda": cuda, "gpu": name}


def verify():
    """imports every app package (PyTorch was already tested in its own process by install_torch)"""
    bad = []
    for mod, req in PACKAGES:
        try:
            importlib.import_module(mod)
        except Exception:
            bad.append(req)
    if bad:
        say(f"{MISS} verification            cannot import: {', '.join(bad)}")
        return False
    say(f"{OK} verification            every package imports")
    return True


# ---------------------------------------------------------------------- always (fast)
def check_project():
    """required app files (fatal if missing) and optional data / models (warnings)"""
    good = True
    need = ["app.py", "core/config.py", "ui/page_predict.py", "refs/cell_info.tsv", "refs/drug_info.tsv",
            "refs/c2.cp.pid.v2024.1.Hs.symbols.gmt", ".streamlit/config.toml"]
    lost = [f for f in need if not (ROOT / f).exists()]
    if lost:
        good = False
        say(f"{MISS} app files               missing: {', '.join(lost)}")
    else:
        say(f"{OK} app files               all {len(need)} present")
    models = sorted((ROOT / "models").glob("*.pt")) if (ROOT / "models").exists() else []
    say((OK if models else WARN) + f" trained models          {len(models)} in models/"
        + ("" if models else " - Predict needs at least one (train one, or copy the demo models in)"))
    data = ROOT / "data"
    for label, names in [("input.txt dataset", ["input.zip", "input.txt", "input.7z"]),
                         ("processed_data - GDSC", ["processed_data_GDSC.zip", "processed_data.zip", "processed_data"]),
                         ("IMPROVE CSA studies", ["csa"])]:
        hit = next((n for n in names if (data / n).exists()), None)
        say((OK if hit else WARN) + f" {label:<23} " + (f"data/{hit}" if hit else f"not in data/ ({' or '.join(names)})"))
    return good


def quiet_first_run():
    """skip Streamlit's one-time e-mail question in the console"""
    cred = Path.home() / ".streamlit" / "credentials.toml"
    if not cred.exists():
        try:
            cred.parent.mkdir(parents=True, exist_ok=True)
            cred.write_text('[general]\nemail = ""\n', encoding="utf-8")
        except OSError:
            pass


def main():
    os.chdir(ROOT)
    full = any(a.lower().lstrip("-/") in ("full", "check", "recheck") for a in sys.argv[1:])
    say(f"{OK} Python {sys.version.split()[0]:<17}{PY}")
    say()
    say(" [2/4] Packages")
    stamp = None if full else read_stamp()
    if fast_ok(stamp):
        t = stamp["torch"]
        say(f"{OK} app packages            all {len(PACKAGES)} installed (unchanged)")
        say(torch_line(t.get("version", "?"), t.get("cuda"), t.get("gpu", ""), "CPU only"))
        say()
        say(" [3/4] Project files")
        ok = check_project()
        say()
        say(" [4/4] Verification")
        say(f"{OK} verification            passed on {stamp.get('checked', '?')}, nothing changed since")
        say("         (full re-check: start.bat full)")
        quiet_first_run()
        return 0 if ok else 1

    forget_stamp()
    gpu = nvidia_gpu()
    ok = install_packages()
    torch_info = install_torch(gpu) if ok else None
    ok = ok and torch_info is not None
    say()
    say(" [3/4] Project files")
    ok = check_project() and ok
    say()
    say(" [4/4] Verification")
    ok = ok and verify()
    if ok:
        write_stamp(torch_info)
    quiet_first_run()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
