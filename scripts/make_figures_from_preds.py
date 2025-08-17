# Copyright 2025 Dickson A. Terrero
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# (full text in the LICENSE file at the project root)


#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def scatter_diag(ax, x, y, title, xlabel, ylabel):
    ax.scatter(x, y, s=8, alpha=0.5)
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

def resid_vs_pred(ax, y_pred, resid, title, xlabel):
    ax.scatter(y_pred, resid, s=8, alpha=0.5)
    ax.axhline(0.0, linestyle="--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual (pred − true)")
    ax.grid(True, alpha=0.3)

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def find_col(actual_cols, *aliases):
    """Return the real column name matching any alias list (by normalized name)."""
    canon = {norm(c): c for c in actual_cols}
    for alias in aliases:
        a = norm(alias)
        if a in canon:
            return canon[a]
    return None

def load_stage2_as_log1p_posonly(df):
    cols = list(df.columns)

    # 1) Try direct log1p columns
    tru_l1p = find_col(cols, "true_log1pgamma", "true_log1p_gamma", "log1p_gamma_true")
    pred_l1p = find_col(cols, "pred_log1pgamma", "pred_log1p_gamma", "log1p_gamma_pred")
    if tru_l1p and pred_l1p:
        y_true = df[tru_l1p].to_numpy()
        y_pred = df[pred_l1p].to_numpy()
        pos_mask = y_true > 0  # log1p(gamma)>0  <=>  gamma>0
        return y_true[pos_mask], y_pred[pos_mask], "log1p(from file)"

    # 2) Fall back to raw gamma, convert to log1p
    tru_g = find_col(cols, "true_gamma", "gamma_true", "y_true_gamma", "true_gamma_pos", "true_gamma_posonly")
    pred_g = find_col(cols, "pred_gamma", "gamma_pred", "y_pred_gamma", "pred_gamma_pos", "pred_gamma_posonly")
    if tru_g and pred_g:
        g_true = df[tru_g].to_numpy()
        g_pred = df[pred_g].to_numpy()
        y_true = np.log1p(np.clip(g_true, 0, None))
        y_pred = np.log1p(np.clip(g_pred, 0, None))
        pos_mask = g_true > 0
        return y_true[pos_mask], y_pred[pos_mask], "log1p(from raw gamma)"

    # 3) If only log10(gamma) is present, convert: log1p(gamma) = log(1 + 10^x)
    tru_l10 = find_col(cols, "true_log10gamma", "log10_gamma_true")
    pred_l10 = find_col(cols, "pred_log10gamma", "log10_gamma_pred")
    if tru_l10 and pred_l10:
        t = df[tru_l10].to_numpy()
        p = df[pred_l10].to_numpy()
        g_true = np.power(10.0, t)
        g_pred = np.power(10.0, p)
        y_true = np.log1p(g_true)
        y_pred = np.log1p(g_pred)
        pos_mask = g_true > 0
        return y_true[pos_mask], y_pred[pos_mask], "log1p(converted from log10)"

    raise SystemExit(
        "Could not find Stage-2 gamma columns.\n"
        "Expected any of:\n"
        "  - [true_log1pgamma, pred_log1pgamma]\n"
        "  - [true_gamma, pred_gamma]\n"
        "  - [true_log10gamma, pred_log10gamma]"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Directory with preds_stage1.csv and preds_stage2.csv (e.g., runs_hurdle_clean)")
    ap.add_argument("--fig_dir", required=True,
                    help="Where to write figures (e.g., y_bridge_pipeline_supporting_files/figures)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1 figures (if available) ---
    p1 = run_dir / "preds_stage1.csv"
    if p1.exists():
        s1 = pd.read_csv(p1)
        if {"true_log10phi","pred_log10phi"}.issubset(s1.columns):
            y_true = s1["true_log10phi"].to_numpy()
            y_pred = s1["pred_log10phi"].to_numpy()

            fig, ax = plt.subplots(figsize=(5,4), dpi=150)
            scatter_diag(ax, y_true, y_pred,
                         r"Stage 1: Predicted vs. True $\log_{10}\phi_v$",
                         r"True $\log_{10}\phi_v$", r"Pred $\log_{10}\phi_v$")
            fig.tight_layout()
            fig.savefig(fig_dir / "predicted_vs_actual_phi_v_log10.png"); plt.close(fig)

            resid = y_pred - y_true
            fig, ax = plt.subplots(figsize=(5,4), dpi=150)
            resid_vs_pred(ax, y_pred, resid,
                          r"Stage 1: Residuals vs. Predicted ($\log_{10}\phi_v$)",
                          r"Pred $\log_{10}\phi_v$")
            fig.tight_layout()
            fig.savefig(fig_dir / "residuals_vs_predicted_phi_v_log10.png"); plt.close(fig)

    # --- Stage 2 figures (hurdle, magnitude on gamma>0 in log1p domain) ---
    p2 = run_dir / "preds_stage2.csv"
    s2 = pd.read_csv(p2)
    y_true_pos, y_pred_pos, how = load_stage2_as_log1p_posonly(s2)
    print(f"# Stage-2 columns resolved via: {how}. Using {y_true_pos.size} positive-γ samples.")

    # Scatter
    fig, ax = plt.subplots(figsize=(5,4), dpi=150)
    scatter_diag(ax, y_true_pos, y_pred_pos,
                 r"Stage 2: Predicted vs. True $\log1p(\gamma)$ ($\gamma>0$)",
                 r"True $\log1p(\gamma)$", r"Pred $\log1p(\gamma)$")
    fig.tight_layout()
    # Save with explicit new name…
    fig.savefig(fig_dir / "predicted_vs_actual_gamma_log1p_posonly.png")
    # …and also the legacy LaTeX name for drop-in compatibility
    fig.savefig(fig_dir / "predicted_vs_actual_gamma_log10.png")
    plt.close(fig)

    # Residuals
    resid = y_pred_pos - y_true_pos
    fig, ax = plt.subplots(figsize=(5,4), dpi=150)
    resid_vs_pred(ax, y_pred_pos, resid,
                  r"Stage 2: Residuals vs. Predicted ($\log1p(\gamma)$, $\gamma>0$)",
                  r"Pred $\log1p(\gamma)$")
    fig.tight_layout()
    fig.savefig(fig_dir / "residuals_vs_predicted_gamma_log1p_posonly.png")
    fig.savefig(fig_dir / "residuals_vs_predicted_gamma_log10.png")
    plt.close(fig)

    print("Wrote figures to:", fig_dir.resolve())

if __name__ == "__main__":
    main()
