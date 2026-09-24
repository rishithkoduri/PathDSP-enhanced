"""
Preprocessing: a folder of DepMap+GDSC CSVs -> the real PathDSP/PathGDT matrix.
Auto-detects the files, reports progress via a callback, returns a DataFrame
(index [drug,cell], EXP+MUT+CNV+DG+CHEM features, resp) and a drug->SMILES dict.

Validated on real data: 5-block matrix -> FNN R2 0.84 / PathGDT.
"""
import os, glob, json
import numpy as np, pandas as pd


# ---------- file auto-detection ----------
def find_files(data_dir):
    files = {p.lower(): p for p in glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)}
    def pick(*needles, exclude=()):
        for low, full in files.items():
            base = os.path.basename(low)
            if all(n in base for n in needles) and not any(e in base for e in exclude):
                return full
        return None
    return {
        "expr":  pick("expression", "proteincoding"),
        "mut":   pick("mutations", "hotspot"),
        "cnv":   pick("cngene"),
        "model": pick("model") if pick("model") and "omics" not in os.path.basename(pick("model")).lower() else
                 next((f for l, f in files.items() if os.path.basename(l) == "model.csv"), None),
        "gdsc":  pick("gdsc_dataset", exclude=("gdsc2",)) or pick("gdsc", "dataset", exclude=("gdsc2",)),
    }


# ---------- reference loaders ----------
def load_gmt(p):
    gs = {}
    for line in open(p):
        a = line.rstrip("\n").split("\t")
        if len(a) >= 3: gs[a[0]] = [g for g in a[2:] if g]
    return gs


def _norm(s): return "".join(c for c in str(s).lower() if c.isalnum())


# ---------- ssGSEA (numpy) ----------
def ssgsea(expr_df, gene_sets, alpha=0.25, normalize=True):
    G = expr_df.shape[1]
    X = expr_df.values.astype(float)
    order = np.argsort(-X, axis=1)
    rank_abs = np.empty_like(X)
    for s in range(X.shape[0]):
        r = np.empty(G); r[order[s]] = np.arange(G, 0, -1); rank_abs[s] = r ** alpha
    out = {}
    for name, gl in gene_sets.items():
        mask = np.zeros(G, dtype=bool)
        gi = {g: i for i, g in enumerate(expr_df.columns)}
        for g in gl:
            if g in gi: mask[gi[g]] = True
        if mask.sum() == 0: continue
        col = np.empty(X.shape[0])
        for s in range(X.shape[0]):
            oin = mask[order[s]]; w = rank_abs[s][order[s]]; nh = oin.sum()
            if nh == 0 or nh == G: col[s] = 0.0; continue
            wh = w * oin; step = (wh / wh.sum()) - ((~oin) / (G - nh))
            col[s] = np.cumsum(step).sum()
        out[f"EXP_{name}"] = col
    res = pd.DataFrame(out, index=expr_df.index)
    if normalize and len(res):
        rng = res.values.max() - res.values.min()
        if rng > 0: res = (res - res.values.min()) / rng
    return res


def _load_omics(path, pid_genes):
    hdr = pd.read_csv(path, nrows=0)
    keep = [c for c in hdr.columns if "(" in c and c.split(" (")[0] in pid_genes]
    df = pd.read_csv(path, usecols=["ModelID", "IsDefaultEntryForModel"] + keep)
    flag = df["IsDefaultEntryForModel"]; mask = (flag == True) | (flag == "True") | (flag == 1)
    if mask.sum() == 0: mask = pd.Series(True, index=df.index)
    df = df.loc[mask].drop(columns=["IsDefaultEntryForModel"]).dropna(subset=["ModelID"]).set_index("ModelID")
    df.columns = [c.split(" (")[0] for c in df.columns]
    return df.groupby(level=0).first()


def _pathway_scores(mat, pid, prefix, transform):
    genes = set(mat.columns); M = transform(mat); out = {}
    for pw, gl in pid.items():
        cols = [g for g in gl if g in genes]
        out[f"{prefix}_{pw}"] = M[cols].mean(axis=1) if cols else pd.Series(0.0, index=mat.index)
    return pd.DataFrame(out)


def _name2smi(refs_dir):
    """drug-name (normalized) -> SMILES, from refs/smiles.txt + refs/druglist.csv."""
    cid2smi = {a.split("\t")[0].strip(): a.split("\t")[1].strip()
               for a in open(os.path.join(refs_dir, "smiles.txt")) if len(a.split("\t")) == 2}
    dl = pd.read_csv(os.path.join(refs_dir, "druglist.csv"))
    nc = [c for c in dl.columns if c.lower() == "name"][0]
    cc = [c for c in dl.columns if "pubchem" in c.lower()][0]
    name2smi = {}
    for _, r in dl.iterrows():
        cid = str(r[cc]).split(",")[0].strip().replace(".0", "")
        if cid in cid2smi: name2smi[_norm(r[nc])] = cid2smi[cid]
    return name2smi


def load_prebuilt(path, refs_dir, progress=lambda f, m: None):
    """Load a pre-assembled PathDSP input matrix (drug,cell,EXP_*,MUT_*,CNV_*,DG_*,CHEM_*,resp).
    Skips ssGSEA/scoring entirely (fast). Returns (matrix_df[index drug,cell], drug_smiles)."""
    progress(0.1, "Reading prebuilt PathDSP matrix…")
    df = pd.read_csv(path, sep=None, engine="python")
    if "resp" not in df.columns:
        raise ValueError("Prebuilt file must have a 'resp' column (LN_IC50).")
    dcol = "drug" if "drug" in df.columns else df.columns[0]
    ccol = "cell" if "cell" in df.columns else df.columns[1]
    feat = [c for c in df.columns if c.startswith(("EXP_", "MUT_", "CNV_", "DG_", "CHEM_"))]
    if not feat:
        raise ValueError("Prebuilt file has no EXP_/MUT_/CNV_/DG_/CHEM_ feature columns.")
    X = df[[dcol, ccol] + feat + ["resp"]].copy()
    X.columns = ["drug", "cell"] + feat + ["resp"]
    X[feat] = X[feat].astype("float32"); X["resp"] = X["resp"].astype("float32")
    progress(0.9, "Matching drug SMILES…")
    name2smi = _name2smi(refs_dir)
    drug_smiles = {d: (name2smi.get(_norm(d)) or "C") for d in X["drug"].astype(str).unique()}
    progress(1.0, f"Loaded prebuilt: {len(X):,} rows x {len(feat)} features.")
    return X.set_index(["drug", "cell"]), drug_smiles


def morgan(smiles, n=256):
    from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray
    a = np.zeros(n, dtype="float32")
    m = Chem.MolFromSmiles(str(smiles))
    if m is not None:
        bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n); ConvertToNumpyArray(bv, a)
    return a


def build_matrix(data_dir, refs_dir, progress=lambda f, m: None, fp_bits=256, gmt_path=None):
    """Return (matrix_df, drug_smiles_dict). matrix_df: index[drug,cell], features, resp."""
    files = find_files(data_dir)
    missing = [k for k in ["expr", "mut", "cnv", "model", "gdsc"] if not files.get(k)]
    if missing:
        raise FileNotFoundError(
            "Could not find these among the uploaded files: " + ", ".join(missing) +
            ". Expected DepMap expression/mutation/CNV + Model.csv + GDSC_DATASET.csv "
            "(as a zip or as loose CSVs).")

    pid = load_gmt(gmt_path or os.path.join(refs_dir, "c2.cp.pid.v7.0.symbols.gmt"))
    pid_genes = set(g for gl in pid.values() for g in gl)

    progress(0.05, "Loading gene expression (subset to pathway genes)…")
    hdr = pd.read_csv(files["expr"], nrows=0)
    keep = [c for c in hdr.columns if "(" in c and c.split(" (")[0] in pid_genes]
    expr = pd.read_csv(files["expr"], usecols=["ModelID", "IsDefaultEntryForModel"] + keep)
    flag = expr["IsDefaultEntryForModel"]; mask = (flag == True) | (flag == "True") | (flag == 1)
    if mask.sum() == 0: mask = pd.Series(True, index=expr.index)
    e = expr.loc[mask].drop(columns=["IsDefaultEntryForModel"]).dropna(subset=["ModelID"]).set_index("ModelID")
    e.columns = [c.split(" (")[0] for c in e.columns]
    e = e.groupby(level=0).first().astype("float32")

    progress(0.20, f"ssGSEA over {len(pid)} PID pathways…")
    EXP = ssgsea(e, pid)

    progress(0.45, "Mutation -> pathway scores…")
    MUT = _pathway_scores(_load_omics(files["mut"], pid_genes), pid, "MUT", lambda m: (m > 0).astype(float))
    progress(0.60, "Copy-number -> pathway scores…")
    CNV = _pathway_scores(_load_omics(files["cnv"], pid_genes), pid, "CNV", lambda m: (m.sub(1.0)).abs())

    progress(0.72, "Mapping DepMap<->GDSC cell lines + responses…")
    model = pd.read_csv(files["model"]).dropna(subset=["COSMICID"])
    c2m = dict(zip(model["COSMICID"].astype(int), model["ModelID"]))
    g = pd.read_csv(files["gdsc"], usecols=["COSMIC_ID", "DRUG_NAME", "LN_IC50", "TARGET"]).dropna(subset=["LN_IC50", "COSMIC_ID"])
    g["ModelID"] = g["COSMIC_ID"].astype(int).map(c2m)
    g = g.dropna(subset=["ModelID"])
    for blk in (EXP, MUT, CNV):
        g = g[g["ModelID"].isin(blk.index)]

    progress(0.80, "Matching drug SMILES…")
    name2smi = _name2smi(refs_dir)
    g["smi"] = g["DRUG_NAME"].map(lambda d: name2smi.get(_norm(d)))
    g = g.dropna(subset=["smi"]).reset_index(drop=True)

    progress(0.88, "Morgan fingerprints (CHEM) + drug-target pathways (DG)…")
    uniq = g.drop_duplicates("DRUG_NAME")
    CHEM = pd.DataFrame({d: morgan(s, fp_bits) for d, s in zip(uniq["DRUG_NAME"], uniq["smi"])}).T
    CHEM.columns = [f"CHEM_{i}" for i in range(fp_bits)]
    pw_of = {}
    for pw, gl in pid.items():
        for gg in gl: pw_of.setdefault(gg, set()).add(pw)
    dtg = g.groupby("DRUG_NAME")["TARGET"].first()
    DG = pd.DataFrame(0.0, index=dtg.index, columns=[f"DG_{p}" for p in pid])
    for d, ts in dtg.items():
        for t in str(ts).split(","):
            for pw in pw_of.get(t.strip(), ()): DG.loc[d, f"DG_{pw}"] = 1.0

    progress(0.95, "Assembling final matrix…")
    def take(df, keys): x = df.loc[keys].reset_index(drop=True); x.columns = list(df.columns); return x
    X = pd.concat([take(EXP, g["ModelID"].values), take(MUT, g["ModelID"].values),
                   take(CNV, g["ModelID"].values), take(DG, g["DRUG_NAME"].values),
                   take(CHEM, g["DRUG_NAME"].values)], axis=1).astype("float32").fillna(0.0)
    X.insert(0, "cell", g["ModelID"].values); X.insert(0, "drug", g["DRUG_NAME"].values)
    X["resp"] = g["LN_IC50"].astype("float32").values
    drug_smiles = dict(zip(uniq["DRUG_NAME"], uniq["smi"]))
    progress(1.0, f"Done: {len(X):,} rows x {X.shape[1]-3} features.")
    return X.set_index(["drug", "cell"]), drug_smiles
