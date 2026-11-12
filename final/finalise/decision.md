# Final decision

Step 1: the recipe must beat plain Paper exact on fresh folds on ≥ 8 of 10 folds with Wilcoxon p < 0.05. If it does, the final IC50 model uses it; otherwise plain Paper exact.

- Step 1 confirmed the recipe: **yes**
- Final IC50 model: **Paper exact + C·D×3·E·F·G·Z**, trained on all of processed_data · GDSC
- Final AAC models: Run A recipe and Run B recipe, each trained on all 587,709 CSA pairs
