"""
Summarise training behaviour per fold.

    python check_loss.py out_gdsc.FNN.cv_10.Loss.txt

For each fold: epochs run, epoch with the lowest validation loss, and the
training / validation loss at that epoch (loss is RMSE-scaled by FNN.py).
"""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], sep="\t")
rows = []
for fold, g in df.groupby("fold"):
    best = g.loc[g["valid loss"].idxmin()]
    rows.append({
        "fold": int(fold),
        "epochs_run": int(g["epoch"].max()),
        "best_epoch": int(best["epoch"]),
        "train_loss_at_best": round(float(best["train loss"]), 4),
        "valid_loss_at_best": round(float(best["valid loss"]), 4),
    })
out = pd.DataFrame(rows)
print(out.to_string(index=False))
print("\nmean epochs run:", round(out["epochs_run"].mean(), 1),
      "| max allowed by FNN.py: 800")
