# Copyright 2025 Dickson A. Terrero
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# (full text in the LICENSE file at the project root)


# make_gamma_posonly_figs_fixed.py
import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--fig_dir", required=True)
args = ap.parse_args()

df = pd.read_csv(f"{args.run_dir}/preds_stage2.csv")

need = {"true_log1pgamma","pred_log1pgamma_cond","ybin_true"}
missing = [c for c in need if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}. I need conditional preds + ybin_true.")

pos = df.loc[df["ybin_true"] == 1].copy()
x = pos["true_log1pgamma"].to_numpy()
y = pos["pred_log1pgamma_cond"].to_numpy()

# Metrics to verify against your table
rho = spearmanr(x, y, nan_policy="omit").correlation
r2  = r2_score(x, y)
print(f"Pos-only metrics: Spearman rho={rho:.4f}, R2={r2:.4f}")

# Scatter
plt.figure(figsize=(7,5.6))
plt.scatter(x, y, s=8, alpha=0.5)
xymin, xymax = np.nanmin(np.r_[x,y]), np.nanmax(np.r_[x,y])
plt.plot([xymin, xymax], [xymin, xymax], ls="--")
plt.title(r"Stage 2: Predicted vs True $\log 1p(\gamma)$ ($\gamma>0$)")
plt.xlabel(r"True $\log 1p(\gamma)$")
plt.ylabel(r"Pred $\log 1p(\gamma)$")
plt.tight_layout()
plt.savefig(f"{args.fig_dir}/predicted_vs_actual_gamma_log1p_posonly.png", dpi=150)

# Residuals vs Pred
resid = y - x
plt.figure(figsize=(7,5.6))
plt.scatter(y, resid, s=8, alpha=0.5)
plt.axhline(0, ls="--")
plt.title(r"Stage 2: Residuals vs Pred ($\log 1p(\gamma)$, $\gamma>0$)")
plt.xlabel(r"Pred $\log 1p(\gamma)$")
plt.ylabel("Residual (pred − true)")
plt.tight_layout()
plt.savefig(f"{args.fig_dir}/residuals_vs_predicted_gamma_log1p_posonly.png", dpi=150)
print("Wrote two figures to:", args.fig_dir)
