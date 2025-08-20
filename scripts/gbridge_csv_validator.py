#!/usr/bin/env python3

# Copyright 2025 Dickson A. Terrero
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# (full text in the LICENSE file at the project root)

"""
G-Bridge CSV Validator

What it does
------------
- Opens a CSV (UTF-8 fallback to Latin-1) and reports:
  * shape, column names, raw dtypes
  * missing values per column
  * constant columns
  * duplicate rows (and duplicate keys if you pass --id-cols)
  * numeric coercion issues
- Validates common target relations if present:
  * log10_phi_v  ≈  log10(phi_v)
  * log1p_gamma  ≈  log1p(gamma)
  * gamma_positive matches (gamma > 0)
- Basic physical sanity hints:
  * nonnegative phi_v / gamma
  * SNR summary if a column like snr_db exists
- Writes a JSON report next to your CSV (or to --out)

Usage
-----
python gbridge_csv_validator.py --csv file.csv
python gbridge_csv_validator.py --csv file.csv --id-cols id,time
python gbridge_csv_validator.py --csv file.csv --out report.json
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------- Config --------------------

# Column name synonyms (case-insensitive match supported)
TARGET_SYNONYMS = {
    "phi_v": ["phi_v", "phi", "phi_visc", "visc_dissipation", "dissipation"],
    "log10_phi_v": ["log10_phi_v", "log_phi_v", "log10_phi", "log_phi"],
    "gamma": ["gamma", "gammaval", "gamma_mag"],
    "log1p_gamma": ["log1p_gamma", "logp1_gamma", "log1p_g"],
    "gamma_positive": ["gamma_positive", "gamma_pos", "gamma_gt0", "y_det", "is_positive", "gamma_label"],
}

REL_TOL = 1e-6  # tolerance for relation checks (in absolute error)
REPORT_KEYS_ORDER = [
    "path", "ok", "encoding", "shape", "columns", "dtypes_raw",
    "missing", "missing_ratio", "duplicate_rows", "duplicate_keys",
    "non_numeric_after_coerce", "numeric_summary", "detected_targets",
    "relations", "physical_sanity", "warnings", "errors"
]

# -------------------- Helpers --------------------

def try_read_csv(path: Path):
    """Try UTF-8 first then Latin-1."""
    try:
        df = pd.read_csv(path)
        return df, "utf-8", None
    except Exception as e1:
        try:
            df = pd.read_csv(path, encoding="latin-1")
            return df, "latin-1", None
        except Exception as e2:
            return None, None, f"Failed to read CSV as utf-8 ({e1}) and latin-1 ({e2})"

def pick_col(df: pd.DataFrame, names):
    """Pick the first matching column (case-insensitive) from a name list."""
    for n in names:
        if n in df.columns:
            return n
    lowmap = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lowmap:
            return lowmap[n.lower()]
    return None

def find_target_column(df: pd.DataFrame, key: str):
    """Return actual column name for a logical key (e.g., 'phi_v')."""
    return pick_col(df, TARGET_SYNONYMS.get(key, []))

def numeric_coerce(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def basic_stats(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return {"count": 0}
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std()) if s.shape[0] > 1 else 0.0,
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }

# -------------------- Core --------------------

def validate(csv_path: Path, id_cols=None):
    id_cols = id_cols or []
    report = {"path": str(csv_path), "ok": False, "warnings": [], "errors": []}

    df, enc, err = try_read_csv(csv_path)
    if err:
        report["errors"].append(err)
        return report

    report["encoding"] = enc
    report["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    report["columns"] = df.columns.tolist()
    report["dtypes_raw"] = {c: str(t) for c, t in df.dtypes.items()}

    # Missing values
    miss = df.isna().sum().astype(int).to_dict()
    miss_ratio = {k: (miss[k] / max(1, df.shape[0])) for k in miss}
    report["missing"] = miss
    report["missing_ratio"] = miss_ratio

    # Constant columns
    constants = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constants:
        report["warnings"].append(f"Constant columns: {constants}")

    # Duplicate rows
    dup_rows = int(df.duplicated().sum())
    report["duplicate_rows"] = dup_rows

    # Duplicate keys if requested
    if id_cols:
        id_cols_present = [c for c in id_cols if c in df.columns]
        if not id_cols_present:
            report["warnings"].append(f"Provided id-cols not found: {id_cols}")
        else:
            dup_keys = int(df.duplicated(subset=id_cols_present).sum())
            report["duplicate_keys"] = {"id_cols": id_cols_present, "count": dup_keys}
            if dup_keys > 0:
                report["warnings"].append(f"Duplicate keys on {id_cols_present}: {dup_keys}")

    # Coerce numeric for summaries/sanity checks
    num_df = pd.DataFrame({c: numeric_coerce(df[c]) for c in df.columns})
    non_numeric_after = [c for c in df.columns if not np.issubdtype(num_df[c].dtype, np.number)]
    report["non_numeric_after_coerce"] = non_numeric_after

    # Numeric summaries
    num_summary = {}
    for c in num_df.columns:
        if np.issubdtype(num_df[c].dtype, np.number):
            num_summary[c] = basic_stats(num_df[c])
    report["numeric_summary"] = num_summary

    # Detect targets
    found = {}
    for k in ["phi_v", "log10_phi_v", "gamma", "log1p_gamma", "gamma_positive"]:
        col = find_target_column(df, k)
        if col:
            found[k] = col
    report["detected_targets"] = found

    # Relation checks
    rel = {}
    if "phi_v" in found and "log10_phi_v" in found:
        a = numeric_coerce(df[found["phi_v"]]).replace([np.inf, -np.inf], np.nan)
        b = numeric_coerce(df[found["log10_phi_v"]]).replace([np.inf, -np.inf], np.nan)
        mask = a > 0
        diff = np.log10(a[mask]) - b[mask]
        if mask.any():
            rel["phi_vs_log10"] = {
                "n": int(mask.sum()),
                "mean_abs_diff": float(np.nanmean(np.abs(diff))),
                "p95_abs_diff": float(np.nanpercentile(np.abs(diff), 95)),
            }

    if "gamma" in found and "log1p_gamma" in found:
        g = numeric_coerce(df[found["gamma"]]).replace([np.inf, -np.inf], np.nan)
        lg = numeric_coerce(df[found["log1p_gamma"]]).replace([np.inf, -np.inf], np.nan)
        mask = g >= 0
        diff = np.log1p(g[mask]) - lg[mask]
        if mask.any():
            rel["gamma_vs_log1p"] = {
                "n": int(mask.sum()),
                "mean_abs_diff": float(np.nanmean(np.abs(diff))),
                "p95_abs_diff": float(np.nanpercentile(np.abs(diff), 95)),
            }

    if "gamma" in found and "gamma_positive" in found:
        g = numeric_coerce(df[found["gamma"]]).fillna(-1e9)
        det_raw = df[found["gamma_positive"]].astype(str).str.strip().str.lower()
        truth_table = {"1": True, "true": True, "yes": True, "y": True, "t": True,
                       "0": False, "false": False, "no": False, "n": False, "f": False}
        det = det_raw.map(truth_table)
        det = det.fillna(numeric_coerce(df[found["gamma_positive"]]) > 0)
        consistent = (det == (g > 0))
        rel["detector_consistency"] = {
            "n": int(consistent.shape[0]),
            "agree_frac": float(consistent.mean()),
            "disagreements": int((~consistent).sum()),
        }

    report["relations"] = rel

    # Physical sanity hints
    phys = {}
    for k in ["phi_v", "gamma"]:
        if k in found:
            vals = numeric_coerce(df[found[k]])
            phys[f"{k}_negatives"] = int((vals < 0).sum())

    # SNR summary if present
    snr_col = None
    for cand in ["snr_db", "snr", "SNR_dB", "snr_dB"]:
        if cand in df.columns:
            snr_col = cand
            break
    if snr_col:
        s = numeric_coerce(df[snr_col])
        phys["snr_db_summary"] = {
            "min": float(np.nanmin(s)),
            "p25": float(np.nanpercentile(s, 25)),
            "median": float(np.nanmedian(s)),
            "p75": float(np.nanpercentile(s, 75)),
            "max": float(np.nanmax(s)),
        }

    report["physical_sanity"] = phys

    # Final OK heuristic
    ok = True
    if dup_rows > 0:
        ok = False
        report["warnings"].append(f"{dup_rows} duplicate rows found")

    if "phi_vs_log10" in rel:
        p95 = rel["phi_vs_log10"].get("p95_abs_diff")
        if p95 is not None and p95 > REL_TOL:
            ok = False
            report["warnings"].append("log10(phi_v) does not match phi_v within tolerance")

    if "gamma_vs_log1p" in rel:
        p95 = rel["gamma_vs_log1p"].get("p95_abs_diff")
        if p95 is not None and p95 > REL_TOL:
            ok = False
            report["warnings"].append("log1p(gamma) does not match gamma within tolerance")

    report["ok"] = ok
    # reorder keys for readability
    report = {k: report.get(k) for k in REPORT_KEYS_ORDER if k in report} | \
             {k: v for k, v in report.items() if k not in REPORT_KEYS_ORDER}
    return report

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path, help="Path to CSV file")
    ap.add_argument("--id-cols", type=str, default="",
                    help="Comma-separated id columns for duplicate-key check, e.g., 'id,time'")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to write JSON report")
    args = ap.parse_args()

    id_cols = [s.strip() for s in args.id_cols.split(",") if s.strip()] if args.id_cols else []

    rep = validate(args.csv, id_cols=id_cols)

    # Decide output path
    out_path = args.out
    if out_path is None:
        out_path = args.csv.with_suffix(args.csv.suffix + ".gbridge_report.json")

    out_path.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nReport saved to: {out_path}")

if __name__ == "__main__":
    main()
