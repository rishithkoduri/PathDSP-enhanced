"""
Run once from the folder that contains FNN.py:
    python apply_patches.py

Fixes:
  1. Removes the leftover `import myPlotter as myplot` line
     (myPlotter.py was deleted upstream and is never used).
  2. Replaces np.Inf with np.inf in myUtility.py (np.Inf was removed in NumPy 2).
"""
import pathlib

targets = ["FNN.py", "leaveOneGroupOut_FNN.py", "load_pretrained.py", "myUtility.py"]

for name in targets:
    p = pathlib.Path(name)
    if not p.exists():
        print(f"skip (not found): {name}")
        continue
    raw = p.read_bytes().decode("utf-8")
    new = raw
    # drop the myPlotter import line, keeping the file's own line endings
    new = "".join(l for l in new.splitlines(keepends=True)
                  if l.strip() != "import myPlotter as myplot")
    new = new.replace("np.Inf", "np.inf")
    if new != raw:
        p.write_bytes(new.encode("utf-8"))
        print(f"patched: {name}")
    else:
        print(f"no change needed: {name}")
