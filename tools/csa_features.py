"""
PathDSP features for the IMPROVE cross-study analysis (CSA) benchmark data, built the way the authors built theirs:

  CHEM  Morgan fingerprint of the drug (RDKit, 256 bits)
  DG    NetPEA of the drug's target genes          (random walk with restart on STRING, then pathway z-scores)
  EXP   ssGSEA of the cell line's expression       (gseapy)
  MUT   NetPEA of the cell line's mutated genes
  CNV   NetPEA of the cell line's copy-number-altered genes

NetPEA follows github.com/TangYiChing/NetPEA (the code PathDSP's README points to) step for step:
  RWR      STRING graph (edge weight = combined score), made symmetric (A + Aᵀ), diagonal 0, rows L1-normalised;
           p₀ = 1 at every seed gene, p ← (1 − r)·pA + r·p₀ with r = 0.5 until Σ|Δp| ≤ 1e-5 (each group on its own)
  score    mean walk probability over the pathway genes found in the network
  null     1,000 random gene lists of the same size, drawn with replacement from the network genes
           (numpy seed 42, drawn pathway by pathway in gene-set-file order — the same draws as the original)
  z        (score − mean(null)) / sd(null)   (sd with ddof 0; NaN when the null has no spread)
Every block is finally z-scored column-wise over all drugs / all cell lines, as in the authors' files.

Only numpy / scipy / pandas are needed here; RDKit and gseapy are imported by the functions that use them.
"""
import gzip
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

RESTART = 0.5
TOL = 1e-5
PERMUTATIONS = 1000
SEED = 42
FP_BITS = 256
FP_RADIUS = 3


# ======================================================================================================================
# gene sets
# ======================================================================================================================
def read_gmt(path):
    """[(name, [genes])] in file order"""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = line.rstrip("\r\n").split("\t")
            if len(f) >= 3:
                out.append((f[0], [g for g in f[2:] if g]))
    return out


# ======================================================================================================================
# protein network
# ======================================================================================================================
def load_string(links_path, info_path, min_score=0):
    """(sorted gene names, row-normalised adjacency CSR float64) from STRING protein.links + protein.info.
    Protein ids are mapped to their preferred gene names; edges between proteins with the same name are dropped
    (they would be self-loops) and parallel edges between two names are summed, as scipy does when the original code
    builds its matrix from the name-mapped edge list."""
    info = pd.read_csv(info_path, sep="\t", usecols=[0, 1], dtype=str)
    info.columns = ["protein", "name"]
    name_of = dict(zip(info["protein"], info["name"]))
    op = gzip.open if str(links_path).endswith(".gz") else open
    with op(links_path, "rt") as fh:
        df = pd.read_csv(fh, sep=" ", dtype={"protein1": str, "protein2": str, "combined_score": np.float64})
    if min_score:
        df = df[df["combined_score"] >= min_score]
    a = df["protein1"].map(name_of)
    b = df["protein2"].map(name_of)
    ok = a.notna() & b.notna()
    a, b, w = a[ok].to_numpy(), b[ok].to_numpy(), df["combined_score"].to_numpy()[ok.to_numpy()]
    genes = np.array(sorted(set(a) | set(b)))
    pos = pd.Index(genes)
    ia, ib = pos.get_indexer(a), pos.get_indexer(b)
    n = len(genes)
    M = sp.csr_matrix((w, (ia, ib)), shape=(n, n), dtype=np.float64)
    M = (M + M.T).tocsr()
    M.setdiag(0)
    M.eliminate_zeros()
    rs = np.asarray(abs(M).sum(1)).ravel()
    rs[rs == 0] = 1.0
    M = sp.diags(1.0 / rs) @ M
    return genes, M.tocsr()


# ======================================================================================================================
# random walk with restart (batched, every group stops on its own criterion like the per-group original)
# ======================================================================================================================
def rwr(A, seed_lists, restart=RESTART, tol=TOL, chunk=256, progress=None):
    """seed_lists: list of int arrays (network indices). Returns float64 [len(seed_lists), n_genes];
    groups without a seed in the network get a row of zeros (as in the original)."""
    n = A.shape[0]
    AT = A.T.tocsr()
    out = np.zeros((len(seed_lists), n), dtype=np.float64)
    todo = [i for i, s in enumerate(seed_lists) if len(s)]
    for c0 in range(0, len(todo), chunk):
        ids = todo[c0:c0 + chunk]
        P0 = np.zeros((n, len(ids)), dtype=np.float64)
        for j, i in enumerate(ids):
            P0[seed_lists[i], j] = 1.0
        P = P0.copy()
        active = np.arange(len(ids))
        while len(active):
            new = AT @ P[:, active]
            new *= (1.0 - restart)
            new += P0[:, active] * restart
            res = np.abs(new - P[:, active]).sum(0)
            P[:, active] = new
            active = active[res > tol]
        out[ids] = P.T
        if progress:
            progress(min(1.0, (c0 + len(ids)) / max(1, len(todo))))
    return out


# ======================================================================================================================
# NetPEA z-scores
# ======================================================================================================================
def netpea(R, genes, gmt, permutations=PERMUTATIONS, seed=SEED, block=12, progress=None):
    """R [groups, n_genes] walk probabilities (columns = `genes`, the network's sorted gene list).
    Returns z [groups, len(gmt)] float64."""
    n = len(genes)
    pos = pd.Index(genes)
    rng = np.random.RandomState(seed)
    shares, draws = [], []
    for _, gl in gmt:                                     # null draws, pathway by pathway in file order
        share = sorted(set(gl) & set(genes.tolist()))
        k = len(share)
        shares.append(pos.get_indexer(share))
        draws.append(np.stack([rng.choice(n, k) for _ in range(permutations)]) if k else None)
    Z = np.full((R.shape[0], len(gmt)), np.nan)
    for p0 in range(0, len(gmt), block):
        cols, rows, vals, spans = [], [], [], []
        c = 0
        for p in range(p0, min(p0 + block, len(gmt))):
            d = draws[p]
            if d is None:
                spans.append(None)
                continue
            k = d.shape[1]
            rows.append(d.ravel())
            cols.append(np.repeat(np.arange(c, c + permutations), k))
            vals.append(np.full(d.size, 1.0 / k))
            spans.append((c, c + permutations))
            c += permutations
        if c:
            M = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, c))
            BG = np.asarray((M.T @ R.T).T)                   # [groups, c] — duplicates in a draw are summed = mean
        for p, span in zip(range(p0, min(p0 + block, len(gmt))), spans):
            if span is None:
                continue
            bg = BG[:, span[0]:span[1]]
            mu, sd = bg.mean(1), bg.std(1)
            score = R[:, shares[p]].mean(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                Z[:, p] = np.where(sd != 0, (score - mu) / sd, np.nan)
        if progress:
            progress(min(1.0, (p0 + block) / len(gmt)))
    return Z


def times_expression(R, genes, groups, exp):
    """NetPEA's optional step: walk probability × expression for the genes the expression table has (others kept)."""
    R = R.copy()
    g_idx = pd.Index(genes).get_indexer(exp.columns)
    okg = g_idx >= 0
    r_idx = pd.Index(groups).get_indexer(exp.index)
    okr = r_idx >= 0
    E = exp.to_numpy(np.float64)[np.ix_(okr, okg)]
    R[np.ix_(r_idx[okr], g_idx[okg])] *= E
    return R


def zscore_columns(Z):
    """column-wise z-score (population sd), NaN-aware; columns without spread become 0"""
    Z = np.asarray(Z, dtype=np.float64)
    mu = np.nanmean(Z, 0)
    sd = np.nanstd(Z, 0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (Z - mu) / sd


# ======================================================================================================================
# drugs
# ======================================================================================================================
def morgan_bits(smiles, n_bits=FP_BITS, radius=FP_RADIUS):
    """[n, n_bits] uint8 (rows of a SMILES that RDKit cannot parse are all -1)"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    out = np.full((len(smiles), n_bits), -1, dtype=np.int8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            out[i] = np.array(list(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)), dtype=np.int8)
    return out


def inchikey(smiles):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m) or None
    except Exception:
        return None


def norm_name(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


# GDSC 'TARGET' entries are protein names; these map the common non-symbol ones to HGNC gene symbols.
# Families are expanded to their members; entries that are not proteins (e.g. 'DNA crosslinker') are not mapped.
GDSC_ALIASES = {
    "PDGFR": ["PDGFRA", "PDGFRB"], "VEGFR": ["FLT1", "KDR", "FLT4"], "VEGFR1": ["FLT1"], "VEGFR2": ["KDR"],
    "VEGFR3": ["FLT4"], "VEGFR3/FLT4": ["FLT4"], "ABL": ["ABL1"], "BCR-ABL": ["ABL1", "BCR"],
    "ERK1": ["MAPK3"], "ERK2": ["MAPK1"], "ERK5": ["MAPK7"], "MEK1": ["MAP2K1"], "MEK2": ["MAP2K2"],
    "MEK5": ["MAP2K5"], "MTORC1": ["MTOR"], "MTORC2": ["MTOR"], "mTORC1": ["MTOR"], "mTORC2": ["MTOR"],
    "mTOR": ["MTOR"], "BCL-XL": ["BCL2L1"], "BCL-W": ["BCL2L2"], "BCL-B": ["BCL2L10"], "BFL1": ["BCL2A1"],
    "MCL-1": ["MCL1"], "BCL-2 selective": ["BCL2"],
    "PI3Kalpha": ["PIK3CA"], "PI3Kbeta": ["PIK3CB"], "PI3Kdelta": ["PIK3CD"], "PI3Kgamma": ["PIK3CG"],
    "PK3CG": ["PIK3CG"], "PI3K (class 1)": ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG"],
    "PI3K": ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG"], "PI3K (beta sparing)": ["PIK3CA", "PIK3CD", "PIK3CG"],
    "IR": ["INSR"], "IGFR1": ["IGF1R"], "HSP90": ["HSP90AA1", "HSP90AB1"], "AKT": ["AKT1", "AKT2", "AKT3"],
    "TIE2": ["TEK"], "JNK1": ["MAPK8"], "JNK2": ["MAPK9"], "JNK3": ["MAPK10"], "JNK": ["MAPK8", "MAPK9", "MAPK10"],
    "TAK1": ["MAP3K7"], "DNAPK": ["PRKDC"], "TNKS1": ["TNKS"], "TNKS2": ["TNKS2"], "PDK1 (PDPK1)": ["PDPK1"],
    "RAF": ["ARAF", "BRAF", "RAF1"], "CRAF": ["RAF1"], "c-FGR": ["FGR"], "FAK": ["PTK2"], "FAK1": ["PTK2"],
    "FAK2": ["PTK2B"], "p38": ["MAPK14", "MAPK11", "MAPK12", "MAPK13"], "p38alpha": ["MAPK14"],
    "p38beta": ["MAPK11"], "IKK": ["CHUK", "IKBKB"], "IKK1": ["CHUK"], "IKK-1": ["CHUK"], "IKK2": ["IKBKB"],
    "IKK-2": ["IKBKB"], "IKKb": ["IKBKB"], "RSK": ["RPS6KA1", "RPS6KA3", "RPS6KA2", "RPS6KA6"],
    "RSK1": ["RPS6KA1"], "RSK2": ["RPS6KA3"], "RSK3": ["RPS6KA2"], "MPS1": ["TTK"], "RON": ["MST1R"],
    "KSP": ["KIF11"], "EG5": ["KIF11"], "MNK1": ["MKNK1"], "MNK2": ["MKNK2"], "S6K1": ["RPS6KB1"],
    "KS6B1 (p70S6K)": ["RPS6KB1"], "CBP": ["CREBBP"], "G9A": ["EHMT2"], "GLP": ["EHMT1"],
    "G9a and GLP methyltransferases": ["EHMT2", "EHMT1"], "LSD1": ["KDM1A"], "KDM1": ["KDM1A"],
    "PKCB": ["PRKCB"], "ALK5": ["TGFBR1"], "ROCK": ["ROCK1", "ROCK2"], "GSK3": ["GSK3A", "GSK3B"],
    "TOP2": ["TOP2A", "TOP2B"], "ESR": ["ESR1", "ESR2"], "HDAC": ["HDAC1", "HDAC2", "HDAC3", "HDAC8"],
    "HDAC1-10": ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC5", "HDAC6", "HDAC7", "HDAC8", "HDAC9", "HDAC10"],
    "BET": ["BRD2", "BRD3", "BRD4", "BRDT"], "IAP": ["XIAP", "BIRC2", "BIRC3"], "cIAP": ["BIRC2", "BIRC3"],
    "cIAP1": ["BIRC2"], "cIAP2": ["BIRC3"], "IAP1": ["BIRC2"], "IAP2": ["BIRC3"], "PERK": ["EIF2AK3"],
    "SHP-1 (PTPN6)": ["PTPN6"], "SHP-2 (PTPN11)": ["PTPN11"], "PPARgamma": ["PPARG"], "PPARdelta": ["PPARD"],
    "Procaspase-3": ["CASP3"], "Procaspase-7": ["CASP7"], "Farnesyl-transferase (FNTA)": ["FNTA"],
    "Dihydrofolate reductase (DHFR)": ["DHFR"], "Endothelin-1 receptor (EDNRA)": ["EDNRA"],
    "eEF2K": ["EEF2K"], "MetAP2": ["METAP2"], "WIP1": ["PPM1D"], "SETD8": ["KMT5A"], "PMRT5": ["PRMT5"],
    "VSP34": ["PIK3C3"], "MRCKB": ["CDC42BPB"], "MRCKB_HUMAN": ["CDC42BPB"], "M4K2": ["MAP4K2"],
    "NIK": ["MAP3K14"], "CP11A": ["CYP11A1"], "PARP5a": ["TNKS"], "PARP5b)": ["TNKS2"],
    "Tankyrase 1/2 (PARP5a": ["TNKS"], "SIR2": ["SIRT1"], "BRPF2": ["BRD1"], "BRPF1B": ["BRPF1"],
    "PB1": ["PBRM1"], "Polybromo 1": ["PBRM1"], "KRAS (G12C)": ["KRAS"], "IDH1 (R132H)": ["IDH1"],
    "IDH2(R140Q)": ["IDH2"], "IDH2 R140Q mutant": ["IDH2"], "FGRF1": ["FGFR1"], "FGFR": ["FGFR1", "FGFR2", "FGFR3", "FGFR4"],
    "SGK2": ["SGK2"], "SGK3": ["SGK3"], "CAMK2": ["CAMK2A", "CAMK2B", "CAMK2D", "CAMK2G"],
    "RNA helicase A": ["DHX9"], "Telomerase": ["TERT"], "KSP11": ["KIF11"], "UAF1": ["WDR48"],
    "PARP7": ["TIPARP"], "PARP6": ["PARP6"], "N-myristoyltransferase 1/2": ["NMT1", "NMT2"],
    "MCT1": ["SLC16A1"], "MCT4": ["SLC16A3"], "FLT2": ["FGFR1"], "RAR": ["RARA", "RARB", "RARG"],
    "Retinioic X receptor (RXR) agonist": ["RXRA", "RXRB", "RXRG"], "LXR": ["NR1H3", "NR1H2"], "FXR": ["NR1H4"],
    "SIRT": ["SIRT1", "SIRT2"], "TEC": ["TEC"], "PKC": ["PRKCA", "PRKCB", "PRKCG", "PRKCD", "PRKCE", "PRKCQ"],
    "PKD": ["PRKD1", "PRKD2", "PRKD3"], "Sphingosine Kinase": ["SPHK1", "SPHK2"], "MDM4": ["MDM4"],
    "Stearoyl-CoA desaturase": ["SCD"], "FAS": ["FASN"], "GLS": ["GLS"], "NAE": ["NAE1", "UBA3"],
    "Proteasome": ["PSMB5"], "RNA Polymerase 1": ["POLR1A"], "gamma-secretase": ["PSEN1"],
    
    "AMPK agonist": ["PRKAA1", "PRKAA2"], 
    "TTK and microtubules": ["TTK", "TUBB"], "Microtubule stabiliser": ["TUBB"], "Microtubule destabiliser": ["TUBB"], "TP53 activation": ["TP53"], "P53 Mut specific": ["TP53"],
}


def gdsc_targets(text, network):
    """HGNC symbols in the STRING network for a GDSC 'TARGET' entry"""
    out = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t or t.lower() == "nan":
            continue
        cand = GDSC_ALIASES.get(t) or [t, t.upper()]
        for g in cand:
            if g in network and g not in out:
                out.append(g)
                if t not in GDSC_ALIASES:
                    break
    return out


def ctrp_targets(text, network):
    out = []
    for tok in str(text).split(";"):
        g = tok.strip()
        if g and g.lower() != "nan" and g in network and g not in out:
            out.append(g)
    return out


# ======================================================================================================================
# molecular graphs (Phase 4: learned drug encoder)
# ======================================================================================================================
ELEMENTS = ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "Na", "K", "Pt", "As")
DEGREES = (0, 1, 2, 3, 4, 5, 6)
CHARGES = (-2, -1, 0, 1, 2)
N_HS = (0, 1, 2, 3, 4)
HYBRID = ("SP", "SP2", "SP3", "SP3D", "SP3D2")
ATOM_FEATURES = ([f"el_{e}" for e in ELEMENTS] + ["el_other"] + [f"deg_{d}" for d in DEGREES] +
                 [f"chg_{c}" for c in CHARGES] + [f"hs_{h}" for h in N_HS] + [f"hyb_{h}" for h in HYBRID] +
                 ["hyb_other", "aromatic", "in_ring"])


def _onehot(v, choices):
    out = [0] * (len(choices) + 1)
    out[choices.index(v) if v in choices else len(choices)] = 1
    return out


def atom_features(atom):
    el = _onehot(atom.GetSymbol(), ELEMENTS)
    deg = [int(atom.GetDegree() == d) for d in DEGREES]
    chg = [int(atom.GetFormalCharge() == c) for c in CHARGES]
    hs = [int(min(atom.GetTotalNumHs(), 4) == h) for h in N_HS]
    hyb = _onehot(str(atom.GetHybridization()), HYBRID)
    return el + deg + chg + hs + hyb + [int(atom.GetIsAromatic()), int(atom.IsInRing())]


def mol_graphs(smiles):
    """molecular graph of every drug (largest fragment = the parent compound without salts / counter-ions):
    {'x' uint8 [atoms, F], 'src', 'dst' int32 (both directions of every bond), 'mol' int32 [atoms] (drug index),
     'n_mols', 'features'}"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    X, src, dst, mol = [], [], [], []
    off = 0
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            raise ValueError(f"drug {i}: SMILES not parsable")
        frags = Chem.GetMolFrags(m, asMols=True)
        m = max(frags, key=lambda f: f.GetNumHeavyAtoms())
        for a in m.GetAtoms():
            X.append(atom_features(a))
            mol.append(i)
        for b in m.GetBonds():
            u, v = b.GetBeginAtomIdx() + off, b.GetEndAtomIdx() + off
            src += [u, v]
            dst += [v, u]
        off += m.GetNumAtoms()
    assert len(X[0]) == len(ATOM_FEATURES)
    return {"x": np.array(X, dtype=np.uint8), "src": np.array(src, dtype=np.int32), "dst": np.array(dst, dtype=np.int32),
            "mol": np.array(mol, dtype=np.int32), "n_mols": len(smiles), "features": np.array(ATOM_FEATURES)}
