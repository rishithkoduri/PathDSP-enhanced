"""
Assemble the Zenodo processed tables (GDSC or CCLE) into the single
tab-separated input file that PathDSP's FNN.py expects:

    drug <tab> cell <tab> feature_1 ... feature_n <tab> resp

Feature order follows the paper's Figure 1: CHEM, DG-Net, EXP, MUT-Net, CNV-Net.

Usage:
    python build_input.py --data-dir processed_data --dataset GDSC --out GDSC.input.txt
"""
import argparse
import os
import pandas as pd


def read(path):
    return pd.read_csv(path, sep="\t", index_col=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder containing GDSC/ and CCLE/")
    ap.add_argument("--dataset", default="GDSC", choices=["GDSC", "CCLE"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = os.path.join(a.data_dir, a.dataset)
    p = a.dataset
    resp = read(f"{d}/drug_response/{p}.RESP.negLog10IC50.txt")            # drugs x cells
    chem = read(f"{d}/morgan_fingerprint/{p}.CHEM.MF.txt") if p == "GDSC" \
        else read(f"{d}/morgan_fingerprint/{p}..CHEM.MF.txt")               # CCLE file name has '..'
    dg = read(f"{d}/pathway_enrichment/DGNet/{p}.DGNet.NetPEA.txt")        # drugs x pathways
    exp = read(f"{d}/pathway_enrichment/EXP/{p}.EXP.ssGSEA.txt")           # cells x pathways
    mut = read(f"{d}/pathway_enrichment/MUT/{p}.MUT.NetPEA.txt")
    cnv = read(f"{d}/pathway_enrichment/CNV/{p}.CNV.NetPEA.txt")

    # prefix columns so identical pathway names from different data types stay distinct
    dg, exp, mut, cnv = (t.add_prefix(x + "_") for t, x in
                         [(dg, "DG"), (exp, "EXP"), (mut, "MUT"), (cnv, "CNV")])

    # long format: one row per (drug, cell) with a measured response
    long = resp.stack().rename("resp").reset_index()
    long.columns = ["drug", "cell", "resp"]
    long = long.dropna(subset=["resp"])  # newer pandas keeps NaN when stacking

    # keep only pairs where every feature block exists
    long = long[long["drug"].isin(chem.index) & long["drug"].isin(dg.index)
                & long["cell"].isin(exp.index) & long["cell"].isin(mut.index)
                & long["cell"].isin(cnv.index)]

    drug_feat = chem.join(dg, how="inner")
    cell_feat = exp.join(mut, how="inner").join(cnv, how="inner")

    out = (long.merge(drug_feat, left_on="drug", right_index=True)
               .merge(cell_feat, left_on="cell", right_index=True))
    feat_cols = list(drug_feat.columns) + list(cell_feat.columns)
    out = out[["drug", "cell"] + feat_cols + ["resp"]]

    out.to_csv(a.out, sep="\t", index=False, float_format="%.6g")
    print(f"rows (drug-cell pairs): {len(out)}")
    print(f"drugs: {out['drug'].nunique()}  cells: {out['cell'].nunique()}")
    print(f"features: {len(feat_cols)}  (CHEM {chem.shape[1]}, DG {dg.shape[1]}, "
          f"EXP {exp.shape[1]}, MUT {mut.shape[1]}, CNV {cnv.shape[1]})")
    print(f"NaNs: {int(out.isna().sum().sum())}")


if __name__ == "__main__":
    main()
