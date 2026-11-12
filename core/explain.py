"""
SHAP explanations (paper: SHAP DeepExplainer, 100 random training rows as background).

Paper exact / Repo exact -> shap.DeepExplainer (as the authors)
Proposed system          -> shap.GradientExplainer (expected gradients; DeepExplainer has no rule for the
                            element-wise products of the interaction module / factorization term)
Attributions are in model output units (−log10 IC50, or drug-wise z for a z-scored target):
    positive = pushes the prediction towards SENSITIVITY (higher −log10 IC50 = lower IC50)
"""
import numpy as np
import pandas as pd
import torch

from . import data as D
from . import nets


def _pathway(f):
    for b in (D.DG, D.EXP, D.MUT, D.CNV):
        if f.startswith(b):
            return f[len(b):]
    return None


def shap_values(net, bg, xs, use_deep, progress=None, chunk=100):
    method = "SHAP DeepExplainer" if use_deep else "SHAP GradientExplainer"
    try:
        import shap
        ex = shap.DeepExplainer(net, bg) if use_deep else shap.GradientExplainer(net, bg)
        parts = []
        for i in range(0, len(xs), chunk):
            xb = xs[i:i + chunk]
            sv = ex.shap_values(xb, check_additivity=False) if use_deep else ex.shap_values(xb)
            sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
            parts.append(sv.reshape(sv.shape[0], -1)[:, :xs.shape[1]])
            if progress:
                progress(min(1.0, (i + chunk) / len(xs)))
        return np.concatenate(parts), method
    except Exception as e:                                  # SHAP missing / failed -> gradient × (input − baseline)
        x = xs.clone().requires_grad_(True)
        net(x).sum().backward()
        return (x.grad * (xs - bg.mean(0))).detach().cpu().numpy(), f"gradient × input (SHAP failed: {type(e).__name__})"


def package(vals, Xz, features):
    V = np.asarray(vals, float); XV = np.asarray(Xz, float)
    vc, xc = V - V.mean(0), XV - XV.mean(0)
    den = np.sqrt((vc ** 2).sum(0) * (xc ** 2).sum(0))
    corr = np.divide((vc * xc).sum(0), den, out=np.zeros(V.shape[1]), where=den > 1e-12)
    feat = pd.DataFrame({"feature": features, "block": [f.split("_", 1)[0] for f in features],
                         "pathway": [_pathway(f) for f in features], "mean_abs": np.abs(V).mean(0),
                         "mean_signed": V.mean(0), "corr": np.round(corr, 4)})
    feat["direction"] = np.where(np.abs(corr) < 0.1, "mixed",
                                 np.where(corr > 0, "higher → more sensitive", "higher → more resistant"))
    feat = feat.sort_values("mean_abs", ascending=False).reset_index(drop=True)
    blocks = (feat.groupby("block", as_index=False)["mean_abs"].sum()
              .sort_values("mean_abs", ascending=False).reset_index(drop=True))
    blocks["share"] = 100 * blocks["mean_abs"] / max(blocks["mean_abs"].sum(), 1e-12)
    pw = feat.dropna(subset=["pathway"])
    pathways = pw.pivot_table(index="pathway", columns="block", values="mean_abs", aggfunc="sum", fill_value=0.0)
    for b in ("DG", "EXP", "MUT", "CNV"):
        if b not in pathways.columns:
            pathways[b] = 0.0
    pathways = pathways[["DG", "EXP", "MUT", "CNV"]]
    pathways["total"] = pathways.sum(axis=1)
    pathways = pathways.sort_values("total", ascending=False).reset_index()
    return feat, blocks, pathways


def attention_table(state, spec, features, mu, sd, X_te, drugs, mol_te=None):
    """Pathway-graph model: mean drug-conditioned attention per pathway over the explained pairs, the share of
    pairs where the pathway is in the top 5, and the drug that weighs it most"""
    net = nets.build(spec, len(features), None, features)
    net.load_state_dict(state)
    net.eval()
    xs = torch.as_tensor((X_te - mu) / sd, dtype=torch.float32)
    if nets.is_mol(spec):                                   # (M) the graph model sees [x ; drug embedding]
        with torch.no_grad():
            xs = net.augment(xs, torch.as_tensor(np.asarray(mol_te)))
        net = net.base
    att = net.attention(xs).cpu().numpy()                   # [N, P]
    top5 = np.argsort(-att, axis=1)[:, :5]
    in_top = np.zeros(att.shape[1]); np.add.at(in_top, top5.ravel(), 1)
    df = pd.DataFrame(att, columns=net.nodes)
    df["drug"] = np.asarray(drugs)
    by_drug = df.groupby("drug").mean()
    t = pd.DataFrame({"pathway": net.nodes, "mean_attention": att.mean(0), "top5_share": in_top / len(att),
                      "uniform": 1.0 / att.shape[1], "most_by_drug": by_drug.idxmax(axis=0).reindex(net.nodes).to_numpy()})
    return t.sort_values("mean_attention", ascending=False).reset_index(drop=True)


def explain_fold(state, spec, features, pairs, mu, sd, X_bg, X_te, dev, progress=None, mol_bg=None, mol_te=None):
    net = nets.build(spec, len(features), pairs, features)
    net.load_state_dict(state)
    net.to(dev).eval()
    bg = torch.as_tensor((X_bg - mu) / sd, dtype=torch.float32, device=dev)
    xs = torch.as_tensor((X_te - mu) / sd, dtype=torch.float32, device=dev)
    if nets.is_mol(spec):                                   # (M) explain the network on [x ; drug embedding]:
        from . import molenc as M                           # the 64 MOL_ inputs form the block 'MOL'
        with torch.no_grad():
            bg = net.augment(bg, torch.as_tensor(np.asarray(mol_bg), device=dev))
            xs = net.augment(xs, torch.as_tensor(np.asarray(mol_te), device=dev))
        net, features = net.base, M.aug_features(features, spec)
    vals, method = shap_values(net, bg, xs, use_deep=isinstance(net, nets.PathDSPNet), progress=progress)
    feat, blocks, pathways = package(vals, xs.detach().cpu().numpy(), features)
    return feat, blocks, pathways, method
