#!/usr/bin/env python3
import argparse, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ----------------------------- utils -----------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def first_col(df, names, required=True):
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise SystemExit(f"Missing any of required columns: {names}")
    return None

def ensure_umag(df):
    # Try "Velocity Magnitude (m/s)" or "Umag"; else compute from Ux,Uy,Uz
    umag_col = first_col(df, ["Velocity Magnitude (m/s)", "Umag"], required=False)
    if umag_col:
        return df[umag_col].to_numpy()
    ux = first_col(df, ["Ux (m/s)", "Ux"], required=True)
    uy = first_col(df, ["Uy (m/s)", "Uy"], required=True)
    uz = first_col(df, ["Uz (m/s)", "Uz"], required=True)
    return np.linalg.norm(df[[ux, uy, uz]].to_numpy(), axis=1)

# ----------------------------- main ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Leak-free Stage-1 baseline on log10(Phi_v) with spatial test split.")
    ap.add_argument("--input_csv", required=True, help="Path to OpenFOAM-derived CSV (your training_table or raw).")
    ap.add_argument("--outdir", type=Path, default=Path("runs_stage1_leakfree"))
    ap.add_argument("--test_frac_x", type=float, default=0.20, help="Top fraction of X held out for TEST (default 0.20)")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    # Standardize required columns (robust to naming variants)
    col_phi = first_col(df, ["Phi_v (W/m^3)", "Phi_v"])
    col_kef = first_col(df, ["KE_Flux_Density (W/m^2)", "KE_Flux_Density"])
    col_x   = first_col(df, ["X (m)", "X"])
    col_y   = first_col(df, ["Y (m)", "Y"])
    col_z   = first_col(df, ["Z (m)", "Z"])

    # Build non-leaking features
    Umag = ensure_umag(df)
    X = df[col_x].to_numpy()
    Y = df[col_y].to_numpy()
    Z = df[col_z].to_numpy()
    r = np.sqrt(Y**2 + Z**2)
    KE = df[col_kef].to_numpy()
    FEATS = ["Umag", "KE_Flux_Density", "X", "Y", "Z", "r"]
    X_all = np.column_stack([Umag, KE, X, Y, Z, r])

    # Target in log10 domain
    eps = 1e-30
    y_all = np.log10(np.clip(df[col_phi].to_numpy(), eps, None))

    # Spatial TEST split by X (hold-out highest X slice)
    x_thr = np.quantile(X, 1.0 - args.test_frac_x)
    te_mask = X >= x_thr
    tr_mask = ~te_mask

    Xtr, ytr = X_all[tr_mask], y_all[tr_mask]
    Xte, yte = X_all[te_mask], y_all[te_mask]

    # Train model (simple, stable)
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1
    )
    rf.fit(Xtr, ytr)
    yhat = rf.predict(Xte)

    # Metrics (all in log10 domain)
    r2  = float(r2_score(yte, yhat))
    mae = float(mean_absolute_error(yte, yhat))
    rmse = float(np.sqrt(mean_squared_error(yte, yhat)))

    # Save predictions with stable row_id for audit
    te_df = df.loc[te_mask, [col_x, col_y, col_z]].reset_index(drop=False).rename(columns={"index":"row_id"})
    te_df["true_log10phi"] = yte
    te_df["pred_log10phi"] = yhat
    te_df.to_csv(outdir / "preds_stage1.csv", index=False)

    # Save metrics.json with provenance
    meta = {
        "protocol": "Stage1 leak-free baseline (RF, non-leaking features, spatial split, log-domain metrics)",
        "input_csv": str(Path(args.input_csv).resolve()),
        "input_csv_sha256": sha256(Path(args.input_csv)),
        "features": FEATS,
        "split": {"type": "spatial_by_X", "threshold": float(x_thr), "test_frac_x": args.test_frac_x},
        "model": {"type": "RandomForestRegressor", "n_estimators": args.n_estimators, "random_state": args.random_state},
        "metrics_log10_phi": {"R2": r2, "MAE": mae, "RMSE": rmse}
    }
    (outdir / "metrics.json").write_text(json.dumps(meta, indent=2))

    # Optional quick plots (log domain)
    try:
        plt.figure(figsize=(5,5))
        mn, mx = float(min(yte.min(), yhat.min())), float(max(yte.max(), yhat.max()))
        plt.scatter(yte, yhat, s=4, alpha=0.6)
        plt.plot([mn, mx], [mn, mx], "r--", lw=1)
        plt.xlabel("True log10(Phi_v)"); plt.ylabel("Pred log10(Phi_v)")
        plt.title("Predicted vs True (log10 Phi_v)")
        plt.tight_layout(); plt.savefig(outdir / "pred_vs_true_log10phi.png", dpi=200); plt.close()

        plt.figure(figsize=(5,5))
        resid = yte - yhat
        plt.scatter(yhat, resid, s=4, alpha=0.6)
        plt.axhline(0, color="r", lw=1, ls="--")
        plt.xlabel("Pred log10(Phi_v)"); plt.ylabel("Residual (true - pred)")
        plt.title("Residuals vs Pred (log10 Phi_v)")
        plt.tight_layout(); plt.savefig(outdir / "resid_vs_pred_log10phi.png", dpi=200); plt.close()
    except Exception:
        pass  # plotting optional

    # Console summary + LaTeX row
    def fmt6(x): return f"{x:.6f}"
    print("\n# STAGE-1 (leak-free) RESULTS  [log10 domain]")
    print(f"  R2  = {fmt6(r2)}")
    print(f"  MAE = {fmt6(mae)}")
    print(f"  RMSE= {fmt6(rmse)}")
    print("\n% LaTeX row (Stage 1 only; paste into your table)")
    print(r"\texttt{$\log_{10}\phi_v$ (Stage 1)} & " + f"{fmt6(r2)} & {mae:.6e} & {rmse:.6e} \\\\")
    print("\nArtifacts written to:", outdir)
    print(" -", outdir / "preds_stage1.csv")
    print(" -", outdir / "metrics.json")
    print("Input CSV SHA256:", meta["input_csv_sha256"])

if __name__ == "__main__":
    main()
