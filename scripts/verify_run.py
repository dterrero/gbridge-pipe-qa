# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 Dickson A Terrero

#!/usr/bin/env python3
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

# --- metrics (same as training) ---
def spearman_rho(y_true, y_pred):
    rt = pd.Series(y_true).rank(method='average').to_numpy()
    rp = pd.Series(y_pred).rank(method='average').to_numpy()
    rt = rt - rt.mean(); rp = rp - rp.mean()
    num = float((rt * rp).sum())
    den = float(np.sqrt((rt * rt).sum() * (rp * rp).sum()))
    return num / den if den else float('nan')

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    resid = y_true - y_pred
    mae  = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean())**2))
    r2  = 1.0 - ss_res/ss_tot if ss_tot > 0 else float('nan')
    rho = spearman_rho(y_true, y_pred)
    return dict(R2=r2, MAE=mae, RMSE=rmse, rho=rho)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def fmt(x): 
    return "nan" if not np.isfinite(x) else f"{x:.6g}"

def tolerant_merge(preds1: pd.DataFrame, test_df: pd.DataFrame, r=6) -> pd.DataFrame:
    """
    Try exact merge on (file,X,Y,Z); if it introduces NaNs, retry with rounded keys;
    if that still fails, fall back to positional alignment (warn).
    """
    keys = ['file','X','Y','Z']
    if not all(k in preds1.columns for k in ['file','X','Y','Z']):
        raise SystemExit("preds_stage1.csv missing one of file,X,Y,Z columns")
    if not all(k in test_df.columns for k in ['file','X','Y','Z','dT_proxy','Umag','Lx']):
        raise SystemExit("input table missing one of file,X,Y,Z,dT_proxy,Umag,Lx")

    # 1) exact merge
    m = preds1.merge(test_df[keys+['dT_proxy','Umag','Lx']], on=keys, how='left')
    if not m.isna().any().any():
        return m

    # 2) rounded merge
    p = preds1.copy(); t = test_df.copy()
    for c in ['X','Y','Z']:
        p[c] = p[c].round(r)
        t[c] = t[c].round(r)
    m2 = p.merge(t[keys+['dT_proxy','Umag','Lx']], on=keys, how='left')
    if not m2.isna().any().any():
        print(f"# WARN: merged on rounded coordinates (r={r}).")
        return m2

    # 3) positional fallback (assumes same order)
    if len(preds1) != len(test_df):
        raise SystemExit("Cannot positional-align: lengths differ between preds_stage1 and test slice.")
    print("# WARN: falling back to positional alignment (order-based).")
    out = preds1.copy()
    out[['dT_proxy','Umag','Lx']] = test_df[['dT_proxy','Umag','Lx']].to_numpy()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--test_files", nargs="+", required=True)
    ap.add_argument("--stage2_mode", choices=["cascade","oracle","hurdle"], default="cascade")
    ap.add_argument("--stage2_target", choices=["gamma","log10","log1p","signedlog1p"], default="log1p")
    ap.add_argument("--rho", type=float, default=1000.0)
    ap.add_argument("--cp",  type=float, default=4186.0)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    preds1 = pd.read_csv(outdir / "preds_stage1.csv")
    preds2 = pd.read_csv(outdir / "preds_stage2.csv")
    mj = json.loads(Path(outdir / "metrics.json").read_text())

    table = pd.read_csv(args.input_csv)
    if 'file' not in table.columns:
        table['file'] = Path(args.input_csv).name
    test_files = {Path(f).name for f in args.test_files}
    test_df = table[table['file'].isin(test_files)].copy()

    # --- Stage 1: metrics + physics sanity ---
    m1 = tolerant_merge(preds1, test_df, r=6)
    m1_calc = metrics(m1['true_log10phi'].values, m1['pred_log10phi'].values)

    phi_hat_phys = m1['dT_proxy'].to_numpy() * (args.rho * args.cp) * \
                   np.clip(m1['Umag'].to_numpy(), 1e-6, None) / np.clip(m1['Lx'].to_numpy(), 1e-12, None)
    log10_phi_phys = np.log10(np.clip(phi_hat_phys, 1e-30, None))
    m1_phys = metrics(m1['true_log10phi'].values, log10_phi_phys)

    # --- Stage 2: metrics ---
    mode = args.stage2_mode
    tgt  = args.stage2_target
    compare = {}

    if mode in ("cascade","oracle"):
        tcol = f"true_{tgt}gamma"; pcol = f"pred_{tgt}gamma"
        if tcol not in preds2.columns or pcol not in preds2.columns:
            raise SystemExit(f"Expected columns '{tcol}' and '{pcol}' in preds_stage2.csv.")
        m2_calc = metrics(preds2[tcol].values, preds2[pcol].values)
        mj_stage2 = mj.get("stage2") or mj.get("stage2_log10gamma") or mj.get("stage2_overall") or {}
        compare = {"Stage1": (mj.get("stage1", {}), m1_calc),
                   "Stage1_phys": (m1_phys, m1_phys),
                   "Stage2": (mj_stage2, m2_calc)}
    else:
        pos_mask = preds2.get('ybin_true', pd.Series(np.ones(len(preds2), int))).astype(int).values > 0
        t_all = f"true_{tgt}gamma"; p_over = f"pred_{tgt}gamma_overall"; p_cond = f"pred_{tgt}gamma_cond"
        if t_all not in preds2.columns or p_cond not in preds2.columns:
            raise SystemExit("Hurdle run missing required columns.")
        m2_pos_calc = metrics(preds2.loc[pos_mask, t_all].values,
                              preds2.loc[pos_mask, p_cond].values)
        m2_over_calc = metrics(preds2[t_all].values, preds2[p_over].values) if p_over in preds2.columns else None
        compare = {"Stage1": (mj.get("stage1", {}), m1_calc),
                   "Stage1_phys": (m1_phys, m1_phys),
                   "Stage2_pos_only": (mj.get("stage2_pos_only", {}), m2_pos_calc)}
        if m2_over_calc is not None:
            compare["Stage2_overall"] = (mj.get("stage2_overall", {}), m2_over_calc)

    # --- Print & check ---
    def show(name, ref, got):
        def row(key):
            a = float(ref.get(key, float('nan'))); b = float(got.get(key, float('nan')))
            rel = abs(a-b) / (abs(a)+1e-12)
            ok = (abs(a-b) <= args.atol) or (rel <= args.rtol) or (np.isnan(a) and np.isnan(b))
            status = "OK" if ok else "MISMATCH"
            return f"{name:>14} {key:>5}: ref={fmt(a)}  got={fmt(b)}  [{status}]"
        print(row('R2')); print(row('MAE')); print(row('RMSE')); print(row('rho'))
        return

    print("\n# AUDIT REPORT")
    print(f"Input CSV sha256: {sha256(Path(args.input_csv))}")

    ok_all = True
    for name, (ref, got) in compare.items():
        show(name, ref, got)
        for k in ('R2','MAE','RMSE','rho'):
            a = float(ref.get(k, float('nan'))); b = float(got.get(k, float('nan')))
            rel = abs(a-b) / (abs(a)+1e-12)
            if not ((abs(a-b) <= args.atol) or (rel <= args.rtol) or (np.isnan(a) and np.isnan(b))):
                ok_all = False

    print("\n# Stage-1 physics-only (sanity): "
          f"R2={fmt(m1_phys['R2'])}, RMSE={fmt(m1_phys['RMSE'])}, rho={fmt(m1_phys['rho'])}")

    if not ok_all:
        print("\nAUDIT: FAIL (mismatched metrics).", file=sys.stderr); sys.exit(1)
    print("\nAUDIT: PASS"); sys.exit(0)

if __name__ == "__main__":
    main()
