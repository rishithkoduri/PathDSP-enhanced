"""
Checks everything PathDSP-X needs and installs what is missing - run by start.bat inside .venv.

  1. packages     the app packages (requirements.txt) and PyTorch (CUDA build when an NVIDIA GPU is present)
  2. project      app files, reference tables, trained models, datasets in data/
  3. verify       imports every package once more and prints one line per item

Exit code 0 = ready to start, 1 = something required is still missing.
"""
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
OK, MISS, WARN = "  [ OK ]", "  [MISS]", "  [WARN]"

# import name -> pip requirement (kept in step with requirements.txt)
PACKAGES = [("streamlit", "streamlit>=1.50"), ("numpy", "numpy>=1.26"), ("pandas", "pandas>=2.1"),
            ("sklearn", "scikit-learn>=1.4"), ("scipy", "scipy>=1.11"), ("altair", "altair>=5.4"),
            ("shap", "shap>=0.46"), ("psutil", "psutil>=5.9"), ("pynvml", "nvidia-ml-py>=12.535"),
            ("lightgbm", "lightgbm>=4.0")]
TORCH_INDEXES = ["https://download.pytorch.org/whl/cu130", "https://download.pytorch.org/whl/cu128"]


def say(text=""):
    print(text, flush=True)


def has(mod):
    return importlib.util.find_spec(mod) is not None


def pip(*args, quiet=True):
    cmd = [PY, "-m", "pip", "install", "--disable-pip-version-check", "--no-input"] + (["-q"] if quiet else []) + list(args)
    return subprocess.call(cmd) == 0


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
    have, cuda, v, name = torch_state()
    if have and (cuda or not gpu):
        say(f"{OK} PyTorch {v:<16}" + (f"GPU: {name}" if cuda else "CPU only (no NVIDIA GPU found)"))
        return True
    if not gpu:
        say("  installing PyTorch (CPU build) - download progress below ...")
        if not pip("torch", quiet=False):
            return False
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
    have, cuda, v, name = torch_state()
    if not have:
        return False
    say(f"{OK} PyTorch {v:<16}" + (f"GPU: {name}" if cuda else "CPU only - training is slow, Predict works"))
    return True


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


def verify():
    bad = []
    for mod, req in PACKAGES + [("torch", "torch")]:
        try:
            importlib.import_module(mod)
        except Exception:
            bad.append(req)
    if bad:
        say(f"{MISS} verification            cannot import: {', '.join(bad)}")
        return False
    say(f"{OK} verification            every package imports")
    return True


def main():
    os.chdir(ROOT)
    say(f"{OK} Python {sys.version.split()[0]:<17}{PY}")
    say()
    say(" [2/4] Packages")
    gpu = nvidia_gpu()
    ok = install_packages() and install_torch(gpu)
    say()
    say(" [3/4] Project files")
    ok = check_project() and ok
    say()
    say(" [4/4] Verification")
    ok = ok and verify()
    quiet_first_run()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
