# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 Dickson A Terrero

#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- helpers ----------
def first_col(df, names, required=True):
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise SystemExit(f"Missing any of required columns: {names}")
    return None

def ensure_umag(df):
    umag = first_col(df, ["Velocity Magnitude (m/s)", "Umag"], required=False)
    if umag:
        return df[umag].to_numpy()
    ux = first_col(df, ["Ux (m/s)", "Ux"], required=True)
    uy = first_col(df, ["Uy (m/s)", "Uy"], required=True)
    uz = first_col(df, ["Uz (m/s)", "Uz"], required=True)
    return np.linalg.norm(df[[ux, uy, uz]].to_numpy(), axis=1)

def add_noise_proxies(out_df, mode, rng):
    """Add noise to proxies ONLY (not to targets)."""
    if mode is None:
        return out_df
    noisy = out_df.copy()
    cols = ["dP_dyn", "dT_proxy", "vib_proxy", "Umag"]
    if mode.startswith("gaussian_"):
        db = int(mode.split("_")[1].replace("db",""))  # 3, 6, 10
        snr_lin = 10.0**(db/10.0)  # power ratio
        for c in cols:
            x = noisy[c].to_numpy(dtype=float)
            p_sig = np.mean(x**2)
            if p_sig <= 0:
                continue
            p_noise = p_sig / snr_lin
            noisy[c] = x + rng.normal(0.0, np.sqrt(p_noise), size=x.shape)
    elif mode == "mixed":
        n = len(noisy)
        # low-freq drift
        drift = 0.5 * np.sin(np.linspace(0, 2*np.pi, n))
        for c in cols:
            noisy[c] = noisy[c].to_numpy(dtype=float) + drift
        # sparse spikes
        k = max(1, int(0.01 * n))
        idx = rng.choice(n, size=k, replace=False)
        mag = rng.uniform(2, 5, size=k) * rng.choice([-1, 1], size=k)
        for c in cols:
            arr = noisy[c].to_numpy(dtype=float)
            arr[idx] = arr[idx] + mag
            noisy[c] = arr
    return noisy

# ---------- core ----------
def build_features_for_file(path, rho, cp, noise):
    df = pd.read_csv(path)

    # standard columns
    col_x   = first_col(df, ["X (m)", "X"])
    col_y   = first_col(df, ["Y (m)", "Y"])
    col_z   = first_col(df, ["Z (m)", "Z"])
    col_ke  = first_col(df, ["KE_Flux_Density (W/m^2)", "KE_Flux_Density"])
    col_phi = first_col(df, ["Phi_v (W/m^3)", "Phi_v"])
    # NEW: optional Gamma passthrough
    col_gamma = first_col(df, ["Gamma (dimensionless)", "Gamma"], required=False)

    # velocity magnitude
    Umag = ensure_umag(df)
    X = df[col_x].to_numpy(); Y = df[col_y].to_numpy(); Z = df[col_z].to_numpy()
    r = np.sqrt(Y**2 + Z**2)
    KE = df[col_ke].to_numpy()

    # geometry length in X
    xmin, xmax = float(np.min(X)), float(np.max(X))
    Lx = xmax - xmin if xmax > xmin else 1.0

    # inlet/outlet bands for dynamic-pressure drop (non-leaky)
    band = 0.02 * Lx
    mask_in  = X <= xmin + band
    mask_out = X >= xmax - band
    # dynamic pressure q = 0.5 * rho * U^2
    q_in  = 0.5 * rho * float(np.mean(Umag[mask_in]**2))
    q_out = 0.5 * rho * float(np.mean(Umag[mask_out]**2))
    dP_dyn = q_out - q_in  # scalar proxy, broadcast to rows

    # heat proxy from KE flux (non-leaky): q'' ≈ KE_Flux_Density
    # ΔT ≈ q'' / (rho * cp * U)
    with np.errstate(divide="ignore", invalid="ignore"):
        dT_proxy = KE / (rho * cp * np.clip(Umag, 1e-9, None))
    dT_proxy = np.nan_to_num(dT_proxy, nan=0.0, posinf=0.0, neginf=0.0)

    # vibration proxy: rolling std of Umag along X (non-leaky)
    order = np.argsort(X)
    Umag_sorted = Umag[order]
    win = 25
    if len(Umag_sorted) < win:
        local_std = np.full_like(Umag_sorted, np.std(Umag_sorted))
    else:
        s = pd.Series(Umag_sorted).rolling(win, center=True, min_periods=max(3, win//5)).std()
        local_std = s.bfill().ffill().to_numpy()

    # unsort back
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    vib_proxy = local_std[inv]

    out = pd.DataFrame({
        "file": Path(path).name,
        "X": X, "Y": Y, "Z": Z,
        "Umag": Umag,
        "r": r,
        "KE_Flux_Density": KE,
        "dP_dyn": dP_dyn,                 # same value per file (broadcast)
        "dT_proxy": dT_proxy,             # per-point proxy (non-leaky)
        "vib_proxy": vib_proxy,           # per-point proxy (non-leaky)
        "Phi_v": df[col_phi].to_numpy(),  # target for Stage 1
        "Lx": np.full_like(Umag, Lx, dtype=float),
    })

    # NEW: include Gamma if present; else add NaNs so Stage-2 can detect/skip gracefully
    if col_gamma is not None:
        out["Gamma"] = df[col_gamma].to_numpy()
    else:
        out["Gamma"] = np.nan

    # optional noise on proxies
    rng = np.random.default_rng(123)
    out = add_noise_proxies(out, noise, rng)

    # logs & z-scores (computed on this file’s rows)
    eps = 1e-30
    out["log_Umag"]     = np.log10(np.clip(out["Umag"].to_numpy(), 1e-12, None))
    out["log_KE_flux"]  = np.log10(np.clip(out["KE_Flux_Density"].to_numpy(), eps, None))
    out["log_dT_proxy"] = np.log10(np.clip(out["dT_proxy"].to_numpy(), eps, None))
    for col in ["dT_proxy", "vib_proxy", "Umag", "KE_Flux_Density"]:
        mu = out[col].mean(); sd = out[col].std() + 1e-12
        out[f"{col}_z"] = (out[col] - mu) / sd

    # Stage-1 target in log10 (safe clip)
    out["log10_phi_v"] = np.log10(np.clip(out["Phi_v"].to_numpy(), eps, None))
    out["log_Lx"] = np.log10(np.clip(out["Lx"], 1e-30, None))
    return out

def main():
    ap = argparse.ArgumentParser(description="Build leak-free training_table.csv from raw OpenFOAM CSVs.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Raw CSV(s)")
    ap.add_argument("--rho", type=float, default=1000.0, help="Fluid density [kg/m^3]")
    ap.add_argument("--cp", type=float, default=4186.0, help="Specific heat [J/(kg·K)]")
    ap.add_argument("--noise", type=str, default=None,
                    help="Optional proxy noise: gaussian_3db | gaussian_6db | gaussian_10db | mixed")
    ap.add_argument("--out", type=Path, default=Path("training_table.csv"))
    args = ap.parse_args()

    parts = []
    for p in args.inputs:
        df_part = build_features_for_file(p, rho=args.rho, cp=args.cp, noise=args.noise)
        parts.append(df_part)

    full = pd.concat(parts, ignore_index=True)
    full.to_csv(args.out, index=False)

    print(f"Wrote: {args.out}  (rows={len(full)})")
    print("Columns:", ", ".join(full.columns))

if __name__ == "__main__":
    main()
