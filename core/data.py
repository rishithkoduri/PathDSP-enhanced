"""
Datasets. Both sources are turned into the same layout the authors use in input.txt:

    features  CHEM_mBit_0..255 | DG_<pathway> ×196 | EXP_<pathway> ×187 | MUT_<pathway> ×196 | CNV_<pathway> ×196
    response  y = −log10 IC50 (µM)  — the unit of every number in the paper

The first load reads the archive (input.txt ≈ 30–60 s), later loads use a binary cache in data/cache (< 1 s).
"""
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

CHEM, DG, EXP, MUT, CNV = "CHEM_", "DG_", "EXP_", "MUT_", "CNV_"
BLOCKS = ("CHEM", "DG", "EXP", "MUT", "CNV")
DRUG_PREFIXES = (CHEM, DG)
CELL_PREFIXES = (EXP, MUT, CNV)
CACHE_VERSION = 3


# ======================================================================================================================
# pair matrix: rows = drug block of the row's drug + cell block of its cell line, assembled on demand
# ======================================================================================================================
class PairX:
    """Behaves like the float32 [N, F] feature matrix of a drug × cell dataset whose features are a drug block
    (CHEM, DG — first) followed by a cell block (EXP, MUT, CNV), without materialising it: a row is
    concat(D[di[row]], C[ci[row]]). Used for the large CSA studies (CTRPv2: 286k pairs × 1,031 features)."""
    ndim = 2
    dtype = np.dtype("float32")

    def __init__(self, D, C, di, ci):
        self.D = np.ascontiguousarray(D, dtype=np.float32)
        self.C = np.ascontiguousarray(C, dtype=np.float32)
        self.di = np.asarray(di, dtype=np.int64)
        self.ci = np.asarray(ci, dtype=np.int64)
        self.n_drug_feats = self.D.shape[1]
        self.shape = (len(self.di), self.D.shape[1] + self.C.shape[1])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, tuple):                                      # X[:, j] — one column
            r, c = idx
            if not isinstance(c, (int, np.integer)):
                return self[r][:, c]
            c = int(c) % self.shape[1]
            if c < self.n_drug_feats:
                return self.D[self.di[r], c]
            return self.C[self.ci[r], c - self.n_drug_feats]
        return np.concatenate([self.D[self.di[idx]], self.C[self.ci[idx]]], axis=-1)

    def stats(self, idx=None):
        """(mean, population sd) per feature over rows idx (all rows when None), float64"""
        di = self.di if idx is None else self.di[idx]
        ci = self.ci if idx is None else self.ci[idx]
        out_m, out_s = [], []
        for T, ix in ((self.D, di), (self.C, ci)):
            w = np.bincount(ix, minlength=len(T)).astype(np.float64)
            n = max(1.0, w.sum())
            m = (w @ T.astype(np.float64)) / n
            v = (w @ (T.astype(np.float64) - m) ** 2) / n
            out_m.append(m); out_s.append(np.sqrt(np.maximum(v, 0.0)))
        return np.concatenate(out_m), np.concatenate(out_s)

    def value_range(self, idx):
        """(min, max) per feature over the drugs and cell lines that occur in rows idx"""
        d = self.D[np.unique(self.di[idx])]
        c = self.C[np.unique(self.ci[idx])]
        return np.concatenate([d.min(0), c.min(0)]), np.concatenate([d.max(0), c.max(0)])

    def mean(self, axis=0, dtype=None):
        if axis != 0:
            raise ValueError("PairX.mean supports axis=0 only")
        return self.stats()[0].astype(dtype or np.float64)

    def astype(self, dtype, copy=True):
        return self[np.arange(self.shape[0])].astype(dtype)


# ======================================================================================================================
# dataset object
# ======================================================================================================================
@dataclass
class Dataset:
    key: str
    label: str
    X: np.ndarray                 # float32 [N, F] (or a PairX that behaves like one)
    y: np.ndarray                 # float32 [N]   −log10 IC50 (µM) — or AAC for the CSA studies (see unit)
    drugs: np.ndarray             # str [N]
    cells: np.ndarray             # str [N]
    features: list = field(default_factory=list)
    unit: str = "ic50"            # key of config.UNITS
    study: str = ""               # CSA study name ('ALL' = the five studies pooled, '' for every other dataset)
    study_codes: np.ndarray = None    # int [N] index into study_names (None = one study)
    study_names: list = field(default_factory=list)

    @cached_property
    def scodes(self):
        """study index of every row (zeros for a one-study dataset)"""
        if self.study_codes is None:
            return np.zeros(self.n, dtype=np.int64)
        return np.asarray(self.study_codes, dtype=np.int64)

    @property
    def n_studies(self):
        return max(1, len(self.study_names))

    @property
    def n(self):
        return int(len(self.y))

    @property
    def n_features(self):
        return int(self.X.shape[1])

    @cached_property
    def drug_list(self):
        return sorted(set(self.drugs.tolist()))

    @cached_property
    def cell_list(self):
        return sorted(set(self.cells.tolist()))

    @cached_property
    def drug_codes(self):
        """integer code per row (index into drug_list)"""
        pos = {d: i for i, d in enumerate(self.drug_list)}
        return np.fromiter((pos[d] for d in self.drugs.tolist()), dtype=np.int64, count=self.n)

    @cached_property
    def block_of(self):
        return np.array([f.split("_", 1)[0] for f in self.features])

    @cached_property
    def drug_mask(self):
        return np.array([f.startswith(DRUG_PREFIXES) for f in self.features])

    @cached_property
    def _first_row(self):
        d, c = {}, {}
        for i, (a, b) in enumerate(zip(self.drugs.tolist(), self.cells.tolist())):
            d.setdefault(a, i)
            c.setdefault(b, i)
        return d, c

    @cached_property
    def _pair_row(self):
        return {(a, b): i for i, (a, b) in enumerate(zip(self.drugs.tolist(), self.cells.tolist()))}

    def row_of(self, drug, cell):
        """row index of a measured pair, or None"""
        return self._pair_row.get((drug, cell))

    def feature_row(self, drug, cell):
        """feature vector for ANY drug × cell combination (measured or not): drug block of the drug, cell block of
        the cell line — exactly how the authors assemble a row."""
        d_rows, c_rows = self._first_row
        x = np.array(self.X[c_rows[cell]], dtype="float32", copy=True)
        x[self.drug_mask] = np.asarray(self.X[d_rows[drug]])[self.drug_mask]
        return x

    @cached_property
    def interaction_pairs(self):
        return interaction_pairs(self.features)

    def drug_responses(self, drug):
        m = self.drugs == drug
        return self.y[m], self.cells[m]


def interaction_pairs(features):
    """(dg_index, cell_index, name) for every pathway present both as DG_<p> and as EXP_/MUT_/CNV_<p>."""
    pos = {f: i for i, f in enumerate(features)}
    dg_idx, cell_idx, names = [], [], []
    for blk in (EXP, MUT, CNV):
        for f in features:
            if f.startswith(DG):
                p = f[len(DG):]
                j = pos.get(blk + p)
                if j is not None:
                    dg_idx.append(pos[f]); cell_idx.append(j); names.append(f"{blk[:-1]}×DG {p}")
    return np.array(dg_idx, dtype=np.int64), np.array(cell_idx, dtype=np.int64), names


# ======================================================================================================================
# locating the source files: the two built-in datasets + any other dataset file put into data/
# ======================================================================================================================
TABLE_EXT = (".txt", ".tsv", ".csv")
_BUILTIN_FILES = {"input.zip", "input.txt", "processed_data_GDSC.zip", "processed_data.zip", "processed_data", "cache",
                  "csa"}


def _slug(name):
    return (re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")[:40] or "data").lower()


def _looks_processed(names):
    return any(n.endswith(".RESP.negLog10IC50.txt") for n in names)


def catalog():
    """{key: meta} of every dataset: the two built-in ones (always listed) + dataset files found in data/
    (a feature table .txt/.tsv/.csv in the authors' input.txt layout, a zip holding one, or a processed_data zip/folder)."""
    out = {k: {**m, "kind": k, "builtin": True} for k, m in C.DATASETS.items()}
    from . import csa as CSA
    out.update(CSA.catalog())                                   # the five IMPROVE CSA studies (data/csa/)
    if not C.DATA.exists():
        return out
    for p in sorted(C.DATA.iterdir(), key=lambda x: x.name.lower()):
        if p.name in _BUILTIN_FILES or p.name.startswith((".", "~")) or p.name.endswith(".part"):
            continue
        kind = None
        try:
            if p.is_dir():
                names = [f for _, _, fs in os.walk(p) for f in fs]
                kind = "custom_processed" if _looks_processed(names) else None
            elif p.suffix.lower() == ".zip":
                names = zipfile.ZipFile(p).namelist()
                kind = ("custom_processed" if _looks_processed(names) else
                        ("custom_table" if any(n.lower().endswith(TABLE_EXT) for n in names) else None))
            elif p.suffix.lower() in TABLE_EXT:
                kind = "custom_table"
        except Exception:
            kind = None
        if kind:
            key = "f-" + _slug(p.stem)
            out.setdefault(key, {"name": p.name, "label": f"{p.name} (from the data folder)", "kind": kind,
                                 "path": str(p), "builtin": False})
    return out


def meta(key):
    if key == "ccle":
        return {"name": "CCLE (external test)", "label": "CCLE (external test)", "kind": "ccle", "builtin": True}
    return catalog().get(key, {"name": key, "label": key, "kind": "missing", "builtin": False})


def is_csa(key):
    return str(key).startswith("csa-")


def unit_of(key):
    """response unit of a dataset: config.UNITS entry (−log10 IC50 for everything but the CSA studies)"""
    return C.UNITS["aac" if is_csa(key) else "ic50"]


def external_for(key):
    """external test set of a training dataset: its own (processed_data GDSC -> CCLE, as in the paper), or CCLE for
    input.txt, whose 1,031 features have the same names and scale as processed_data (checked on both files).
    None when CCLE is not in the data folder."""
    ext = meta(key).get("external") or ("ccle" if key == "input" else None)
    if ext and _source(ext)[0] is None:
        return None
    return ext


def dataset_name(key):
    return meta(key).get("name", key)


def _source(key, cat=None):
    """(kind, path) of the first source found for a dataset key, or (None, None)."""
    if is_csa(key):
        from . import csa as CSA
        return ("csa", CSA.DIR) if CSA.available() and CSA.study_of(key) in CSA.STUDIES + (CSA.ALL,) else (None, None)
    if key in C.DATASETS or key == "ccle":
        spec = C.DATASETS[key if key != "ccle" else "processed"]
        candidates = [C.DATA / spec["archive"], C.DATA / spec["loose"]]
        if key in ("processed", "ccle"):
            candidates += [C.DATA / "processed_data.zip"]
    else:
        p = (cat[key] if cat and key in cat else meta(key)).get("path")
        candidates = [Path(p)] if p else []
    for p in candidates:
        if p.exists():
            return ("zip" if p.suffix == ".zip" else ("dir" if p.is_dir() else "file")), p
    return None, None


def available():
    """{dataset key: bool} — whether its file is present in data/."""
    cat = catalog()
    return {k: _source(k, cat)[0] is not None for k in cat}


def _signature(path):
    st = os.stat(path)
    return f"{os.path.basename(path)}|{st.st_size}|{int(st.st_mtime)}|v{CACHE_VERSION}"


# ======================================================================================================================
# readers
# ======================================================================================================================
def _read_input_txt(kind, path, progress):
    progress(0.05, "Reading input.txt (first time only, ~30–60 s)…")
    if kind == "zip":
        zf = zipfile.ZipFile(path)
        member = next(n for n in zf.namelist() if n.endswith("input.txt"))
        opener = lambda: zf.open(member)                                    # noqa: E731
    else:
        opener = lambda: open(path, "rb")                                   # noqa: E731
    with opener() as fh:
        header = fh.readline().decode("utf-8").rstrip("\r\n").split("\t")
    dtypes = {c: np.float32 for c in header if c not in ("drug", "cell")}
    dtypes.update(drug=str, cell=str)
    with opener() as fh:
        df = pd.read_csv(fh, sep="\t", dtype=dtypes, engine="c")
    df.columns = [("DG_" + c[len("TARGETnNEIGHBOR_"):]) if c.startswith("TARGETnNEIGHBOR_") else c for c in df.columns]
    feats = [c for c in df.columns if c.startswith((CHEM, DG, EXP, MUT, CNV))]
    X = np.ascontiguousarray(df[feats].to_numpy(dtype=np.float32))
    y = df["resp"].to_numpy(dtype=np.float32)
    return X, y, df["drug"].astype(str).to_numpy(), df["cell"].astype(str).to_numpy(), feats


class _Tree:
    """uniform file access inside a folder or a zip archive"""
    def __init__(self, kind, path):
        self.kind, self.path = kind, path
        self.zf = zipfile.ZipFile(path) if kind == "zip" else None
        self.names = (self.zf.namelist() if self.zf else
                      [os.path.relpath(os.path.join(r, f), path).replace("\\", "/")
                       for r, _, fs in os.walk(path) for f in fs])

    def find(self, *basenames):
        for b in basenames:
            hits = [n for n in self.names if n.rsplit("/", 1)[-1] == b and "GDSC_CCLE/" not in n]
            if hits:
                return hits[0]
        raise FileNotFoundError(f"{basenames[0]} not found in {self.path}")

    def table(self, *basenames):
        n = self.find(*basenames)
        if self.zf:
            with self.zf.open(n) as fh:
                return pd.read_csv(io.BytesIO(fh.read()), sep="\t", index_col=0)
        return pd.read_csv(os.path.join(self.path, n), sep="\t", index_col=0)


def _read_processed(kind, path, ds, progress):
    progress(0.1, f"Reading processed_data/{ds}…")
    t = _Tree(kind, path)
    R = t.table(f"{ds}.RESP.negLog10IC50.txt")
    R.index = R.index.astype(str); R.columns = R.columns.astype(str)

    def blk(suffix, prefix, rename_chem=False):
        B = t.table(suffix, suffix.replace(".", "..", 1))          # CCLE ships 'CCLE..CHEM.MF.txt'
        B.index = B.index.astype(str)
        B = B[~B.index.duplicated()]
        B.columns = ([f"CHEM_mBit_{i}" for i in range(B.shape[1])] if rename_chem else [prefix + str(c) for c in B.columns])
        return B.astype("float32")

    chem = blk(f"{ds}.CHEM.MF.txt", CHEM, rename_chem=True)
    dg = blk(f"{ds}.DGNet.NetPEA.txt", DG)
    exp = blk(f"{ds}.EXP.ssGSEA.txt", EXP)
    mut = blk(f"{ds}.MUT.NetPEA.txt", MUT)
    cnv = blk(f"{ds}.CNV.NetPEA.txt", CNV)
    progress(0.6, "Assembling drug × cell rows…")
    long = R.stack().reset_index()
    long.columns = ["drug", "cell", "y"]
    long = long.dropna(subset=["y"])
    cells_ok = set(exp.index) & set(mut.index) & set(cnv.index)
    long = long[long["cell"].isin(cells_ok) & long["drug"].isin(set(chem.index))].reset_index(drop=True)
    d, c = long["drug"].to_numpy(), long["cell"].to_numpy()
    parts = [chem.reindex(d).to_numpy(), dg.reindex(d).fillna(0.0).to_numpy(),          # drug without DG-Net → 0
             exp.reindex(c).to_numpy(), mut.reindex(c).to_numpy(), cnv.reindex(c).to_numpy()]
    X = np.ascontiguousarray(np.concatenate(parts, 1).astype(np.float32))
    feats = list(chem.columns) + list(dg.columns) + list(exp.columns) + list(mut.columns) + list(cnv.columns)
    return X, long["y"].to_numpy(dtype=np.float32), d.astype(str), c.astype(str), feats


def _read_table(kind, path, progress):
    """A feature table in the authors' input.txt layout: drug, cell, feature columns (CHEM_ / DG_ or TARGETnNEIGHBOR_
    / EXP_ / MUT_ / CNV_ prefixes) and a response column (resp = −log10 IC50; LN_IC50 and IC50 in µM are converted)."""
    progress(0.05, f"Reading {os.path.basename(str(path))}…")
    if kind == "zip":
        zf = zipfile.ZipFile(path)
        members = [n for n in zf.namelist() if n.lower().endswith(TABLE_EXT)]
        if not members:
            raise ValueError("The zip holds no .txt / .tsv / .csv table.")
        member = max(members, key=lambda n: zf.getinfo(n).file_size)
        opener = lambda: zf.open(member)                                    # noqa: E731
    else:
        opener = lambda: open(path, "rb")                                   # noqa: E731
    with opener() as fh:
        first = fh.readline().decode("utf-8", "replace")
    sep = "\t" if "\t" in first else ("," if first.count(",") >= first.count(";") else ";")
    header = [h.strip() for h in first.rstrip("\r\n").split(sep)]
    low = {h.lower(): h for h in header}
    dcol = low.get("drug") or low.get("drug_name") or header[0]
    ccol = low.get("cell") or low.get("cell_line") or low.get("cellline") or low.get("sample") or header[1]
    rcol, unit = None, "neglog10"
    for name, u in (("resp", "neglog10"), ("response", "neglog10"), ("neglog10_ic50", "neglog10"),
                    ("-log10ic50", "neglog10"), ("ln_ic50", "ln"), ("lnic50", "ln"), ("ic50", "um")):
        if name in low:
            rcol, unit = low[name], u
            break
    feats_raw = [h for h in header if h.startswith((CHEM, DG, EXP, MUT, CNV, "TARGETnNEIGHBOR_"))]
    if rcol is None:
        raise ValueError("No response column — name it 'resp' (−log10 IC50), 'LN_IC50' or 'IC50' (µM).")
    if not feats_raw:
        raise ValueError("No feature columns — they need the prefixes CHEM_, DG_, EXP_, MUT_ or CNV_ (as in input.txt).")
    dtypes = {h: np.float32 for h in feats_raw}
    dtypes.update({dcol: str, ccol: str})
    with opener() as fh:
        df = pd.read_csv(fh, sep=sep, dtype=dtypes, usecols=[dcol, ccol, rcol] + feats_raw, engine="c")
    progress(0.6, "Checking the table…")
    yv = pd.to_numeric(df[rcol], errors="coerce").to_numpy(dtype=np.float64)
    if unit == "ln":
        yv = -yv / C.LN10
    elif unit == "um":
        with np.errstate(divide="ignore", invalid="ignore"):
            yv = -np.log10(yv)
    ok = np.isfinite(yv)
    df, yv = df.loc[ok].reset_index(drop=True), yv[ok]
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} rows with a usable response — a dataset needs at least 100.")
    ren = {h: "DG_" + h[len("TARGETnNEIGHBOR_"):] for h in feats_raw if h.startswith("TARGETnNEIGHBOR_")}
    names = [ren.get(h, h) for h in feats_raw]
    order = sorted(range(len(names)), key=lambda i: (BLOCKS.index(names[i].split("_", 1)[0]), i))
    Xdf = df[[feats_raw[i] for i in order]].astype("float32")
    Xdf = Xdf.fillna(Xdf.mean()).fillna(0.0)
    X = np.ascontiguousarray(Xdf.to_numpy(dtype=np.float32))
    return (X, yv.astype(np.float32), df[dcol].astype(str).to_numpy(), df[ccol].astype(str).to_numpy(),
            [names[i] for i in order])


# ======================================================================================================================
# public loader
# ======================================================================================================================
def load(key, progress=None):
    """Dataset for 'input', 'processed', 'ccle' (external test set, processed_data/CCLE) or a file found in data/."""
    progress = progress or (lambda f, m: None)
    kind, path = _source(key)
    if kind == "csa":
        from . import csa as CSA
        progress(0.2, "Reading the CSA benchmark package…")
        ds = CSA.dataset(CSA.study_of(key))
        progress(1.0, "Loaded")
        return ds
    if kind is None:
        if is_csa(key):
            raise FileNotFoundError(f"The CSA benchmark package is not in the data folder ({C.DATA / 'csa'}).")
        if key in C.DATASETS or key == "ccle":
            spec = C.DATASETS[key if key != "ccle" else "processed"]
            raise FileNotFoundError(f"Put {spec['archive']} into the data folder ({C.DATA}).")
        raise FileNotFoundError(f"Dataset '{key}' not found in the data folder ({C.DATA}).")
    sig = _signature(path)
    cache = C.CACHE / f"{key}.npz"
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as z:
                if str(z["sig"]) == sig:
                    progress(1.0, "Loaded from cache")
                    return Dataset(key, _label(key), z["X"], z["y"], z["drugs"], z["cells"], z["features"].tolist())
        except Exception:
            pass
    mk = meta(key).get("kind")
    if key == "input":
        X, y, d, c, feats = _read_input_txt(kind, path, progress)
    elif key in ("processed", "ccle"):
        X, y, d, c, feats = _read_processed(kind, path, "GDSC" if key == "processed" else "CCLE", progress)
    elif mk == "custom_processed":
        X, y, d, c, feats = _read_processed(kind, path, "GDSC", progress)
    else:
        X, y, d, c, feats = _read_table(kind, path, progress)
    progress(0.9, "Writing cache…")
    tmp = C.CACHE / f"{key}.tmp.npz"
    np.savez(tmp, X=X, y=y, drugs=np.asarray(d, dtype=str), cells=np.asarray(c, dtype=str),
             features=np.asarray(feats, dtype=str), sig=np.asarray(sig))
    os.replace(tmp, cache)
    progress(1.0, "Loaded")
    return Dataset(key, _label(key), X, y, np.asarray(d, dtype=str), np.asarray(c, dtype=str), list(feats))


def _label(key):
    return dataset_name(key)


def summary(ds):
    blocks = pd.Series(ds.block_of).value_counts().reindex(list(BLOCKS)).fillna(0).astype(int).to_dict()
    return {"pairs": ds.n, "drugs": len(ds.drug_list), "cells": len(ds.cell_list), "features": ds.n_features,
            "blocks": blocks}


def align(ds, features):
    """Feature matrix of `ds` in the column order `features` (columns it lacks are NaN)."""
    pos = {f: i for i, f in enumerate(ds.features)}
    out = np.full((ds.n, len(features)), np.nan, dtype=np.float32)
    for j, f in enumerate(features):
        i = pos.get(f)
        if i is not None:
            out[:, j] = ds.X[:, i]
    return out


# ======================================================================================================================
# cell line / drug annotations (refs/)
# ======================================================================================================================
def cell_info():
    p = C.REFS / "cell_info.tsv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
    return {r.cell_id: r._asdict() for r in df.itertuples(index=False)}


def drug_info():
    p = C.REFS / "drug_info.tsv"
    out = {}
    if p.exists():
        df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
        out = {r.drug: r._asdict() for r in df.itertuples(index=False)}
    try:                                                         # CSA drugs (by their display name)
        from . import csa as CSA
        for k, v in CSA.drug_info().items():
            out.setdefault(k, v)
    except Exception:
        pass
    return out


def cell_label(cell, info):
    i = info.get(cell) or {}
    name = i.get("name") or cell
    ctype = i.get("cancer_type") or i.get("lineage") or ""
    return f"{name} — {ctype} [{cell}]" if ctype else f"{name} [{cell}]"


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
