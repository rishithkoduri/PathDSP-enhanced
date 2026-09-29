"""
Summarise an FNN.py run.

    python evaluate.py out_gdsc.FNN.cv_10.Prediction.txt

Prints MAE, RMSE, R2 and PCC over all test predictions and for each fold,
plus the mean +/- standard deviation across folds (the paper reports mean +/- std).
"""
import sys
import numpy as np
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path, sep="\t", index_col=[0, 1])
print("columns:", list(df.columns), "| rows:", len(df))

y, p = df["resp"].to_numpy(), df["prediction"].to_numpy()


def scores(y, p):
    err = p - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": 1 - ss_res / ss_tot,
        "PCC": float(np.corrcoef(y, p)[0, 1]),
    }


overall = scores(y, p)
print("\nOverall:", {k: round(v, 4) for k, v in overall.items()})

per_fold = pd.DataFrame({f: scores(g["resp"].to_numpy(), g["prediction"].to_numpy())
                         for f, g in df.groupby("fold")}).T
print("\nPer fold:")
print(per_fold.round(4).to_string())
print("\nMean +/- std across folds:")
for c in per_fold.columns:
    print(f"  {c}: {per_fold[c].mean():.3f} +/- {per_fold[c].std():.3f}")

print("\nPaper (Table 1, all features): MAE 0.24, RMSE 0.35, R2 0.92, PCC 0.96")
