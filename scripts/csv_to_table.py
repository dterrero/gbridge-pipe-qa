#!/usr/bin/env python3

# Copyright 2025 Dickson A. Terrero
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# (full text in the LICENSE file at the project root)

"""
G-Bridge: CSV → feature table (leak-aware)

- Builds per-point proxies from raw OpenFOAM-style CSVs
- Avoids leakage (inlet/outlet bands only, per-file scalars, no future info)
- Optional, physically sensible noise:
    • per-row for signals (Umag, vib_proxy, dT_proxy)
    • per-file bias for dP_dyn
- Adds Stage-2 helpers when Gamma is present
- OPTIONAL post-processing (clipping/winsorization) for stress tests

Usage examples
--------------
# v1 (noise-free, for Tables 1–2)
python3 csv_to_table.py \
  --inputs RAW1.csv RAW2.csv \
  --out training_table_v1.csv

# noisy stress test (appendix), one-shot with PP
python3 csv_to_table.py \
  --inputs RAW1.csv RAW2.csv \
  --noise gaussian_6db --seed 123 \
  --pp-clip-nonneg --pp-winsorize Phi_v Gamma --pp-p 99.9 \
  --out training_table_noisy_pp.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ---------- helpers ----------
def first_col(df, names, required=True):
    """Return the first column present from names (case-sensitive list)."""
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise SystemExit(f"Missing any of required columns: {names}")
    return None


def ensure_umag(df):
    """Return velocity magnitude as a numpy array, or reuse provided column."""
    umag = first_col(df, ["Velocity Magnitude (m/s)", "Umag"], required=False)
    if umag:
        return df[umag].to_numpy()
    ux = first_col(df, ["Ux (m/s)", "Ux"], required=True)
    uy = first_col(df, ["Uy (m/s)", "Uy"], required=True)
    uz = first_col(df, ["Uz (m/s)", "Uz"], required=True)
    return np.linalg.norm(df[[ux, uy, uz]].to_numpy(dtype=float), axis=1)


def add_noise_proxies(out_df, mode, rng):
    """
    Add optional noise to proxies.
    - gaussian_Xdb: per-row AWGN scaled to achieve SNR=X dB (rowwise cols);
                    per-file bias for dP_dyn
    - mixed: slow drift + sparse spikes (scaled to RMS); small per-file bias on dP_dyn
    """
    if mode is None:
        return out_df
    noisy = out_df.copy()
    cols_rowwise = ["dT_proxy", "vib_proxy", "Umag"]  # per-row signals
    col_filewise = "dP_dyn"                            # file-level scalar

    if mode.startswith("gaussian_"):
        # e.g., gaussian_3db, gaussian_6db, gaussian_10db
        db = int(mode.split("_")[1].replace("db", ""))
        snr_lin = 10.0 ** (db / 10.0)

        # rowwise noise
        for c in cols_rowwise:
            if c not in noisy:
                continue
            x = noisy[c].to_numpy(dtype=float)
            p_sig = float(np.mean(x**2))
            if p_sig > 0:
                p_noise = p_sig / snr_lin
                noisy[c] = x + rng.normal(0.0, np.sqrt(p_noise), size=x.shape)

        # filewise noise (single draw, then broadcast)
        if col_filewise in noisy:
            x = float(noisy[col_filewise].iloc[0])
            p_sig = x**2
            if p_sig > 0:
                p_noise = p_sig / snr_lin
                offset = float(rng.normal(0.0, np.sqrt(p_noise)))
                noisy[col_filewise] = x + offset

    elif mode == "mixed":
        n = len(noisy)

        # distance-based signals get slow drift + sparse spikes scaled to RMS
        for c in cols_rowwise:
            if c not in noisy:
                continue
            x = noisy[c].to_numpy(dtype=float)
            rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
            drift = 0.1 * rms * np.sin(np.linspace(0.0, 2.0 * np.pi, n))  # ~10% RMS drift
            y = x + drift
            # sparse spikes (~1% of points), magnitude 1–3×RMS with random sign
            k = max(1, int(0.01 * n))
            idx = rng.choice(n, size=k, replace=False)
            mag = rng.uniform(1.0, 3.0, size=k) * rms * rng.choice([-1, 1], size=k)
            y[idx] = y[idx] + mag
            noisy[c] = y

        # small per-file bias on dP_dyn (±5%)
        if col_filewise in noisy:
            x = float(noisy[col_filewise].iloc[0])
            noisy[col_filewise] = x + 0.05 * abs(x) * float(rng.normal())

    return noisy


# ---------- core ----------
def build_features_for_file(path, rho, cp, noise, rng):
    df = pd.read_csv(path, low_memory=False)

    # standard columns
    col_x   = first_col(df, ["X (m)", "X"])
    col_y   = first_col(df, ["Y (m)", "Y"])
    col_z   = first_col(df, ["Z (m)", "Z"])
    col_ke  = first_col(df, ["KE_Flux_Density (W/m^2)", "KE_Flux_Density"])
    col_phi = first_col(df, ["Phi_v (W/m^3)", "Phi_v"])
    # optional Gamma passthrough (for Stage-2 helpers)
    col_gamma = first_col(df, ["Gamma (dimensionless)", "Gamma"], required=False)

    # geometry & base fields
    X = df[col_x].to_numpy(dtype=float)
    Y = df[col_y].to_numpy(dtype=float)
    Z = df[col_z].to_numpy(dtype=float)
    Umag = ensure_umag(df).astype(float)
    KE = df[col_ke].to_numpy(dtype=float)
    r = np.sqrt(Y**2 + Z**2)

    # geometry length in X
    xmin, xmax = float(np.min(X)), float(np.max(X))
    Lx = xmax - xmin if xmax > xmin else 1.0

    # inlet/outlet bands for dynamic-pressure drop (robust, non-leaky)
    band = 0.02 * Lx
    mask_in  = X <= xmin + band
    mask_out = X >= xmax - band
    # fallback to quantiles if empty (tiny/clustered sets)
    if not mask_in.any():
        thr = float(np.quantile(X, 0.02))
        mask_in = X <= thr
    if not mask_out.any():
        thr = float(np.quantile(X, 0.98))
        mask_out = X >= thr

    # dynamic pressure q = 0.5 * rho * U^2
    q_in  = 0.5 * rho * float(np.mean(Umag[mask_in]**2))
    q_out = 0.5 * rho * float(np.mean(Umag[mask_out]**2))
    dP_dyn_drop = q_in - q_out  # positive pressure drop

    # heat proxy from KE flux (non-leaky): q'' ≈ KE_Flux_Density; ΔT ≈ q'' / (rho * cp * U)
    with np.errstate(divide="ignore", invalid="ignore"):
        dT_proxy = KE / (rho * cp * np.clip(Umag, 1e-9, None))
    dT_proxy = np.nan_to_num(dT_proxy, nan=0.0, posinf=0.0, neginf=0.0)

    # vibration proxy: local std of Umag within a physical window ±dx in X (non-leaky)
    order = np.argsort(X)
    X_sorted = X[order]
    Umag_sorted = Umag[order]

    dx = 0.01 * max(1.0, float(np.ptp(X_sorted)))  # 1% of domain (>= 0.01 m)
    local_std = np.empty_like(Umag_sorted, dtype=float)

    j0 = 0
    n_sorted = len(X_sorted)
    for i in range(n_sorted):
        xi = X_sorted[i]
        while j0 < i and X_sorted[j0] < xi - dx:
            j0 += 1
        j1 = i
        while j1 + 1 < n_sorted and X_sorted[j1 + 1] <= xi + dx:
            j1 += 1
        window = Umag_sorted[j0:j1 + 1]
        local_std[i] = float(np.std(window)) if window.size else float(np.std(Umag_sorted))

    inv = np.empty_like(order)
    inv[order] = np.arange(n_sorted)
    vib_proxy = local_std[inv]

    # assemble features/targets
    out = pd.DataFrame({
        "file": Path(path).name,
        "X": X, "Y": Y, "Z": Z,
        "Umag": Umag,
        "r": r,
        "KE_Flux_Density": KE,
        "dP_dyn": dP_dyn_drop,            # same value per file (broadcast)
        "dT_proxy": dT_proxy,             # per-point proxy (non-leaky)
        "vib_proxy": vib_proxy,           # per-point proxy (non-leaky)
        "Phi_v": df[col_phi].to_numpy(dtype=float),  # Stage-1 target
        "Lx": np.full_like(Umag, Lx, dtype=float),
    })

    # include Gamma if present (for Stage-2 helpers)
    if col_gamma is not None:
        out["Gamma"] = df[col_gamma].to_numpy(dtype=float)
    else:
        out["Gamma"] = np.nan

    if out["Gamma"].notna().any():
        g = out["Gamma"].to_numpy(dtype=float)
        out["log1p_gamma"] = np.log1p(np.clip(g, 0.0, None))
        out["gamma_positive"] = (g > 0.0).astype(int)

    # optional noise on proxies (use passed-in RNG)
    out = add_noise_proxies(out, noise, rng)

    # logs & z-scores (computed on this file’s rows)
    eps = 1e-30
    out["log_Umag"]     = np.log10(np.clip(out["Umag"].to_numpy(), 1e-12, None))
    out["log_KE_flux"]  = np.log10(np.clip(out["KE_Flux_Density"].to_numpy(), eps, None))
    out["log_dT_proxy"] = np.log10(np.clip(out["dT_proxy"].to_numpy(), eps, None))
    for col in ["dT_proxy", "vib_proxy", "Umag", "KE_Flux_Density"]:
        mu = float(out[col].mean())
        sd = float(out[col].std()) + 1e-12
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
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for noise")
    # Optional post-processing for stress tests
    ap.add_argument("--pp-clip-nonneg", dest="pp_clip_nonneg", action="store_true",
                    help="Clip Umag/vib_proxy to >=0 (use for noisy stress tests)")
    ap.add_argument("--pp-winsorize", dest="pp_winsorize", nargs="*", default=[],
                    help="Columns to winsorize at 99.9th pct (e.g. Phi_v Gamma)")
    ap.add_argument("--pp-p", dest="pp_p", type=float, default=99.9)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # Build per-file features and combine
    parts = []
    for p in args.inputs:
        df_part = build_features_for_file(p, rho=args.rho, cp=args.cp, noise=args.noise, rng=rng)
        parts.append(df_part)
    full = pd.concat(parts, ignore_index=True)

    # Optional post-processing (for stress tests only)
    def _winsorize(arr, p):
        hi = float(np.nanpercentile(arr, p))
        return np.clip(arr, None, hi)

    if args.pp_clip_nonneg:
        for c in ["Umag", "vib_proxy"]:
            if c in full.columns:
                full[c] = np.clip(full[c].to_numpy(dtype=float), 0.0, None)

    for c in args.pp_winsorize:
        if c in full.columns:
            full[c] = _winsorize(full[c].to_numpy(dtype=float), args.pp_p)

    # --- Keep derived columns consistent after any post-processing ---
    eps = 1e-30
    if "Phi_v" in full.columns:
        full["log10_phi_v"] = np.log10(np.clip(full["Phi_v"].to_numpy(dtype=float), eps, None))
    if "Gamma" in full.columns:
        g = full["Gamma"].to_numpy(dtype=float)
        full["log1p_gamma"] = np.log1p(np.clip(g, 0.0, None))
        full["gamma_positive"] = (g > 0.0).astype(int)
    # Recompute logs impacted by possible clipping (Umag) for consistency
    if "Umag" in full.columns:
        full["log_Umag"] = np.log10(np.clip(full["Umag"].to_numpy(dtype=float), 1e-12, None))
    if "KE_Flux_Density" in full.columns:
        full["log_KE_flux"] = np.log10(np.clip(full["KE_Flux_Density"].to_numpy(dtype=float), eps, None))
    if "dT_proxy" in full.columns:
        full["log_dT_proxy"] = np.log10(np.clip(full["dT_proxy"].to_numpy(dtype=float), eps, None))
    # Recompute z-scores (columns may have changed)
    for col in ["dT_proxy", "vib_proxy", "Umag", "KE_Flux_Density"]:
        if col in full.columns:
            mu = float(full[col].mean())
            sd = float(full[col].std()) + 1e-12
            full[f"{col}_z"] = (full[col] - mu) / sd

    # Write result
    full.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}  (rows={len(full)})")
    print("Columns:", ", ".join(full.columns))


if __name__ == "__main__":
    main()
