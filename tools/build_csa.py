"""
Builds data/csa/ — the five IMPROVE CSA studies (CCLE, CTRPv2, gCSI, GDSCv1, GDSCv2) with PathDSP features — from
the benchmark's raw data. The app ships this folder ready-made; the script is here so it can be rebuilt or checked.

    python tools/build_csa.py --molecules-only [--out data/csa]        (only the drug graphs, from drugs.tsv)
    python tools/build_csa.py --csa <csa_data/raw_data> --links 9606.protein.links.v12.5.txt.gz
                              --info 9606.protein.info.v12.5.txt.gz --gdsc screened_compounds_rel_8.5.csv
                              --ctrp v20.meta.per_compound.txt [--out data/csa]

Needs (only for this script): rdkit, gseapy. About 15 minutes and 3 GB of memory.

What is built (see tools/csa_features.py for the NetPEA details):
  response   AAC = 1 − AUC of the benchmark's dose-response fit (higher = more sensitive, like −log10 IC50);
             the RMSE / R² / PCC of AAC and of AUC are identical
  CHEM       Morgan fingerprint, radius 3, 256 bits (identical to the authors' bits for 131 of 153 GDSC drugs;
             the rest differ in their SMILES)
  DG         NetPEA of the drug's targets — GDSC 'TARGET' (by GDSC id) and CTRP 'gene_symbol_of_protein_target'
             (by name, else by InChIKey); drugs without a known target in STRING get DG = 0 (the average drug)
  EXP        ssGSEA (gseapy, rank-normalised, gene sets of 15–500 genes → 187 PID pathways, as the authors')
  MUT        NetPEA of all mutated genes, walk probabilities × expression (the IMPROVE PathDSP port)
  CNV        NetPEA of all copy-number-altered genes (discretised ≠ 0), × expression (the IMPROVE PathDSP port)
  every block z-scored column-wise over all drugs / all cell lines of the five studies together, so one cell line
  or drug has the same features in every study
  splits     the benchmark's own split files (10 train / val / test splits per study + <study>_all for
             cross-study testing), as row positions into response
"""
import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csa_features as F                                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STUDIES = ("CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2")
N_SPLITS = 10
FORMAT = "pathdsp-csa-1"


def log(msg):
    print(f"[{_dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _wide(path, symbol_row):
    """CSA omics table: 3 header rows (gene ids), then one row per cell line; columns = gene symbols"""
    h = pd.read_csv(path, sep="\t", header=None, nrows=3, index_col=0, dtype=str)
    d = pd.read_csv(path, sep="\t", header=None, skiprows=3, index_col=0)
    d.columns = h.iloc[symbol_row].values
    d = d.loc[:, pd.notna(d.columns)]
    d = d.loc[:, ~pd.Index(d.columns).duplicated()]
    d.index = d.index.astype(str)
    return d


def _symbol_row(path):
    """index (0..2) of the header row holding gene symbols (the one with letters that are not ENSG ids)"""
    h = pd.read_csv(path, sep="\t", header=None, nrows=3, index_col=0, dtype=str)
    best, score = 0, -1
    for i in range(3):
        v = h.iloc[i].dropna().astype(str).head(500)
        s = sum((not x.startswith("ENSG")) and any(c.isalpha() for c in x) for x in v)
        if s > score:
            best, score = i, s
    return best


# ======================================================================================================================
def drug_targets(info, gdsc, gdsc_inchi, ctrp, network):
    """{improve_chem_id: (targets, source)} for every drug in drug_info"""
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    gdsc = gdsc.copy()
    gdsc["t"] = gdsc["TARGET"].map(lambda s: F.gdsc_targets(s, network))
    g_by_id = dict(zip(gdsc["DRUG_ID"].astype(int), gdsc["t"]))
    g_by_name = {}
    for r in gdsc.itertuples(index=False):
        for nm in [r.DRUG_NAME] + str(r.SYNONYMS).split(","):
            k = F.norm_name(nm)
            if k and k != "NAN" and r.t:
                g_by_name.setdefault(k, r.t)
    g_by_key = {}
    if gdsc_inchi is not None:
        gi = gdsc_inchi.merge(gdsc[["DRUG_ID", "t"]], on="DRUG_ID")
        for r in gi.itertuples(index=False):
            if isinstance(r.INCHI_KEY, str) and r.t:
                g_by_key.setdefault(r.INCHI_KEY, r.t)
                g_by_key.setdefault(r.INCHI_KEY[:14], r.t)
    ctrp = ctrp.copy()
    ctrp["t"] = ctrp["gene_symbol_of_protein_target"].map(lambda s: F.ctrp_targets(s, network))
    c_by_name, c_by_key = {}, {}
    for r in ctrp.itertuples(index=False):
        if not r.t:
            continue
        for nm in (r.cpd_name, r.broad_cpd_id):
            c_by_name.setdefault(F.norm_name(nm), r.t)
        k = F.inchikey(r.cpd_smiles)
        if k:
            c_by_key.setdefault(k, r.t)
            c_by_key.setdefault(k[:14], r.t)
    out = {}
    for chem, g in info.groupby("improve_chem_id"):
        tg, src = [], []

        def add(lst, s):
            new = [x for x in lst if x not in tg]
            if new:
                tg.extend(new)
                if s not in src:
                    src.append(s)
        for did in g["DrugID"]:
            if did.startswith("GDSC."):
                add(g_by_id.get(int(did[5:]), []), "GDSC")
        names = [F.norm_name(n) for n in g["NAME"].dropna()]
        for n in names:
            add(c_by_name.get(n, []), "CTRP")
            add(g_by_name.get(n, []), "GDSC")
        if not tg:                                                   # structure match only when names gave nothing
            keys = [F.inchikey(s) for s in g["SMILES"].dropna().unique()]
            for k in [k for k in keys if k]:
                for kk in (k, k[:14]):
                    add(c_by_key.get(kk, []), "CTRP")
                    add(g_by_key.get(kk, []), "GDSC")
        out[chem] = (tg, "+".join(src))
    return out


# ======================================================================================================================
def build(a):
    t0 = time.time()
    raw = Path(a.csa)
    X_, Y_, S_ = raw / "x_data", raw / "y_data", raw / "splits"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gmt_path = a.gmt or next(iter(sorted((ROOT / "refs").glob("*pid*.gmt"))), None)
    gmt = F.read_gmt(gmt_path)
    log(f"gene sets: {len(gmt)} from {Path(gmt_path).name}")

    # ---------------- response
    rsp = pd.read_csv(Y_ / "response.tsv", sep="\t", usecols=["source", "improve_sample_id", "improve_chem_id", "auc"],
                      dtype={"source": str, "improve_sample_id": str, "improve_chem_id": str})
    rsp["row"] = np.arange(len(rsp))
    log(f"response rows: {len(rsp):,}")

    # ---------------- network
    genes, A = F.load_string(a.links, a.info)
    net = set(genes.tolist())
    gidx = pd.Index(genes)
    log(f"STRING: {len(genes):,} genes, {A.nnz:,} directed edges")

    # ---------------- drugs
    smi = pd.read_csv(X_ / "drug_SMILES.tsv", sep="\t", dtype=str)
    smi.columns = ["improve_chem_id", "smiles"]
    smi = smi.drop_duplicates("improve_chem_id").set_index("improve_chem_id")["smiles"]
    info = pd.read_csv(X_ / "drug_info.tsv", sep="\t", dtype=str)
    drug_ids = sorted(set(rsp["improve_chem_id"]) & set(smi.index))
    chem = F.morgan_bits(smi.loc[drug_ids].tolist())
    ok_d = (chem >= 0).all(1)
    drug_ids = [d for d, o in zip(drug_ids, ok_d) if o]
    chem = chem[ok_d].astype(np.float32)
    log(f"drugs with a parsable SMILES: {len(drug_ids)} (CHEM radius {F.FP_RADIUS}, {F.FP_BITS} bits)")
    gdsc = pd.read_csv(a.gdsc)
    gdsc_inchi = pd.read_csv(a.gdsc_inchi) if a.gdsc_inchi else (gdsc if "INCHI_KEY" in gdsc.columns else None)
    ctrp = pd.read_csv(a.ctrp, sep="\t")
    tg = drug_targets(info[info["improve_chem_id"].isin(drug_ids)], gdsc, gdsc_inchi, ctrp, net)
    seeds = [gidx.get_indexer(tg.get(d, ([], ""))[0]) for d in drug_ids]
    has_t = np.array([len(s) > 0 for s in seeds])
    log(f"drugs with targets in STRING: {int(has_t.sum())} of {len(drug_ids)}")
    R = F.rwr(A, seeds)
    DG = F.netpea(R, genes, gmt)
    DG[~has_t] = np.nan
    del R

    # ---------------- cells
    sym = lambda p: _wide(p, _symbol_row(p))                                  # noqa: E731
    exp = sym(X_ / "cancer_gene_expression.tsv")
    mut = sym(X_ / "cancer_mutation_count.tsv")
    cnv = sym(X_ / "cancer_discretized_copy_number.tsv")
    cell_ids = sorted(set(rsp["improve_sample_id"]) & set(exp.index) & set(mut.index) & set(cnv.index))
    log(f"cell lines with expression, mutation and copy number: {len(cell_ids)}")
    import gseapy as gp
    es = gp.ssgsea(data=exp.loc[cell_ids].T, gene_sets=str(gmt_path), outdir=None, sample_norm_method="rank",
                   permutation_num=0, no_plot=True, threads=max(1, os.cpu_count() or 1), min_size=15, max_size=500)
    W = es.res2d.pivot(index="Name", columns="Term", values="NES").astype(float).loc[cell_ids]
    exp_names = [n for n, _ in gmt if n in W.columns]
    EXP = W[exp_names].to_numpy()
    log(f"EXP: ssGSEA on {len(exp_names)} pathways")
    E = exp.loc[cell_ids]
    blocks = {}
    for blk, tab, rule in (("MUT", mut, lambda v: v > 0), ("CNV", cnv, lambda v: v != 0)):
        V = np.nan_to_num(tab.loc[cell_ids].to_numpy(dtype=float))
        gcol = gidx.get_indexer(pd.Index(tab.columns))                  # network index of every column (−1 = absent)
        sl = [gcol[rule(V[i]) & (gcol >= 0)] for i in range(len(cell_ids))]
        R = F.rwr(A, sl)
        R = F.times_expression(R, genes, cell_ids, E)
        blocks[blk] = F.netpea(R, genes, gmt)
        del R
        log(f"{blk}: median {int(np.median([len(s) for s in sl]))} seed genes per cell line")

    # ---------------- z-score, assemble
    names = [n for n, _ in gmt]
    zf = lambda Z: np.nan_to_num(F.zscore_columns(Z), nan=0.0).astype(np.float32)   # noqa: E731
    drug_X = np.concatenate([chem, zf(DG)], 1)
    cell_X = np.concatenate([zf(EXP), zf(blocks["MUT"]), zf(blocks["CNV"])], 1)
    feats = ([f"CHEM_mBit_{i}" for i in range(F.FP_BITS)] + [f"DG_{n}" for n in names] +
             [f"EXP_{n}" for n in exp_names] + [f"MUT_{n}" for n in names] + [f"CNV_{n}" for n in names])

    # ---------------- rows + splits
    dpos, cpos = pd.Index(drug_ids), pd.Index(cell_ids)
    di, ci = dpos.get_indexer(rsp["improve_chem_id"]), cpos.get_indexer(rsp["improve_sample_id"])
    keep = (di >= 0) & (ci >= 0) & np.isfinite(rsp["auc"].to_numpy(float)) & rsp["source"].isin(STUDIES).to_numpy()
    kept = rsp[keep].reset_index(drop=True)
    new_of = np.full(len(rsp), -1, dtype=np.int64)
    new_of[kept["row"].to_numpy()] = np.arange(len(kept))
    study_code = pd.Index(STUDIES).get_indexer(kept["source"]).astype(np.int8)
    splits, report = {}, {}
    for s in STUDIES:
        rows_s = set(np.where(study_code == STUDIES.index(s))[0].tolist())
        for fn in [f"{s}_all.txt"] + [f"{s}_split_{k}_{p}.txt" for k in range(N_SPLITS) for p in ("train", "val", "test")]:
            idx = np.loadtxt(S_ / fn, dtype=np.int64, ndmin=1)
            bad = rsp["source"].to_numpy()[idx] != s
            if bad.any():
                raise ValueError(f"{fn}: {int(bad.sum())} rows are not from {s}")
            loc = new_of[idx]
            loc = loc[loc >= 0]
            assert set(loc.tolist()) <= rows_s
            splits[f"{s}|{fn[len(s) + 1:-4]}"] = loc.astype(np.int32)
        for k in range(N_SPLITS):
            tr, va, te = (set(splits[f"{s}|split_{k}_{p}"].tolist()) for p in ("train", "val", "test"))
            assert not (tr & va) and not (tr & te) and not (va & te), f"{s} split {k} overlaps"
        report[s] = {"rows_in_benchmark": int((rsp["source"] == s).sum()), "rows": len(rows_s),
                     "drugs": int(kept.loc[study_code == STUDIES.index(s), "improve_chem_id"].nunique()),
                     "cells": int(kept.loc[study_code == STUDIES.index(s), "improve_sample_id"].nunique()),
                     "drugs_with_targets": int(len(set(kept.loc[study_code == STUDIES.index(s), "improve_chem_id"])
                                                   & {d for d, h in zip(drug_ids, has_t) if h}))}
    log("rows kept per study: " + ", ".join(f"{s} {report[s]['rows']:,}/{report[s]['rows_in_benchmark']:,}" for s in STUDIES))

    # ---------------- names for display
    dname = {}
    for chem_id, g in info.groupby("improve_chem_id"):
        nm = [n for n in g["NAME"].dropna().tolist() if not str(n).upper().startswith(("BRD-", "CHEMBL", "NSC"))]
        dname[chem_id] = (nm or g["NAME"].dropna().tolist() or [chem_id])[0]
    drugs_tab = pd.DataFrame({"drug": drug_ids, "name": [dname.get(d, d) for d in drug_ids],
                              "targets": [", ".join(tg.get(d, ([], ""))[0]) for d in drug_ids],
                              "target_source": [tg.get(d, ([], ""))[1] for d in drug_ids],
                              "smiles": smi.loc[drug_ids].tolist()})

    # ---------------- write
    np.savez_compressed(out / "features.npz", drug_ids=np.array(drug_ids), drug_X=drug_X, cell_ids=np.array(cell_ids),
                        cell_X=cell_X, features=np.array(feats))
    np.savez_compressed(out / "response.npz", study=study_code, drug=di[keep].astype(np.int32),
                        cell=ci[keep].astype(np.int32), y=(1.0 - kept["auc"].to_numpy(np.float64)).astype(np.float32),
                        source_row=kept["row"].to_numpy(np.int32), studies=np.array(STUDIES))
    np.savez_compressed(out / "splits.npz", **splits)
    drugs_tab.to_csv(out / "drugs.tsv", sep="\t", index=False)
    write_molecules(out)
    meta = {"format": FORMAT, "built": f"{_dt.datetime.now():%Y-%m-%d %H:%M}", "seconds": round(time.time() - t0),
            "response": "AAC = 1 - AUC", "studies": report, "n_drugs": len(drug_ids), "n_cells": len(cell_ids),
            "n_features": len(feats), "blocks": {"CHEM": F.FP_BITS, "DG": len(names), "EXP": len(exp_names),
                                                   "MUT": len(names), "CNV": len(names)},
            "string": {"genes": int(len(genes)), "edges": int(A.nnz), "links": Path(a.links).name},
            "gene_sets": Path(gmt_path).name, "netpea": {"restart": F.RESTART, "tol": F.TOL,
                                                         "permutations": F.PERMUTATIONS, "seed": F.SEED},
            "drugs_with_targets": int(has_t.sum())}
    with open(out / "build.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    log(f"written to {out} in {time.time() - t0:.0f} s")
    return meta


def write_molecules(out):
    """data/csa/molecules.npz — molecular graph of every drug, in the order of features.npz drug_ids"""
    out = Path(out)
    with np.load(out / "features.npz", allow_pickle=False) as z:
        ids = [str(d) for d in z["drug_ids"]]
    tab = pd.read_csv(out / "drugs.tsv", sep="\t", dtype=str).set_index("drug")
    g = F.mol_graphs(tab.loc[ids, "smiles"].tolist())
    np.savez_compressed(out / "molecules.npz", drug_ids=np.array(ids), **g)
    log(f"molecular graphs: {len(ids)} drugs, {len(g['x']):,} atoms, {len(g['src']) // 2:,} bonds")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csa", required=True, help="csa_data/raw_data folder (x_data, y_data, splits)")
    p.add_argument("--links", required=True, help="STRING 9606.protein.links.*.txt.gz")
    p.add_argument("--info", required=True, help="STRING 9606.protein.info.*.txt.gz")
    p.add_argument("--gdsc", required=True, help="GDSC screened_compounds_rel_8.5.csv")
    p.add_argument("--gdsc-inchi", default=None, help="GDSC screened compounds with INCHI_KEY (optional)")
    p.add_argument("--ctrp", required=True, help="CTRPv2 v20.meta.per_compound.txt")
    p.add_argument("--gmt", default=None, help="MSigDB C2 CP:PID gmt (default: the one in refs/)")
    p.add_argument("--out", default=str(ROOT / "data" / "csa"))
    if "--molecules-only" in sys.argv:                  # add molecules.npz to an existing package
        i = sys.argv.index("--out") + 1 if "--out" in sys.argv else None
        write_molecules(sys.argv[i] if i else ROOT / "data" / "csa")
        return
    build(p.parse_args())


if __name__ == "__main__":
    main()
