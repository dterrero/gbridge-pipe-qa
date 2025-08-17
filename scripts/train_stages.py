import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------- Metrics ----------
def spearman_rho(y_true, y_pred):
    rt = pd.Series(y_true).rank(method='average').to_numpy()
    rp = pd.Series(y_pred).rank(method='average').to_numpy()
    rt = rt - rt.mean(); rp = rp - rp.mean()
    num = float((rt * rp).sum())
    den = math.sqrt(float((rt * rt).sum()) * float((rp * rp).sum()))
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

def fmt(x):
    try:
        return "nan" if not np.isfinite(x) else f"{x:.4g}"
    except Exception:
        return str(x)

# ---------- Models (robust) ----------
def fit_predict_stage1(Xtr, ytr, Xte):
    model = GradientBoostingRegressor(
        loss='huber', alpha=0.9,
        n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
        random_state=0
    )
    model.fit(Xtr, ytr)
    return model.predict(Xte), model

# ---------- Helpers ----------
def local_std(vals, k=25):
    vals = np.asarray(vals, float)
    if len(vals) < k:
        return np.full_like(vals, vals.std())
    from numpy.lib.stride_tricks import sliding_window_view as swv
    w = swv(vals, k)
    s = w.std(axis=1)
    pad = (k-1)//2
    return np.pad(s, (pad, len(vals)-len(s)-pad), mode='edge')

def derivative_along_x(x, v, k=9):
    """
    Robust 1D derivative vs x:
      - sorts by x
      - collapses duplicate x by averaging v
      - optional k-point smoothing before gradient
      - maps derivative back to original order
    """
    x = np.asarray(x, float); v = np.asarray(v, float)
    idx = np.argsort(x); inv_sort = np.empty_like(idx); inv_sort[idx] = np.arange(len(idx))
    xs = x[idx]; vs = v[idx]
    uniq_x, inv_u, counts = np.unique(xs, return_inverse=True, return_counts=True)
    v_sum = np.zeros_like(uniq_x, dtype=float)
    np.add.at(v_sum, inv_u, vs)
    v_mean = v_sum / counts
    if len(uniq_x) >= 3:
        if len(uniq_x) >= k:
            from numpy.lib.stride_tricks import sliding_window_view as swv
            w = swv(v_mean, k).mean(axis=1)
            pad = (k - 1) // 2
            x_c = uniq_x[pad:-pad]
            if len(x_c) < 2:
                d_uniq = np.gradient(v_mean, uniq_x)
            else:
                d_core = np.gradient(w, x_c)
                d_uniq = np.pad(d_core, (pad, len(uniq_x) - len(d_core) - pad), mode='edge')
        else:
            d_uniq = np.gradient(v_mean, uniq_x)
    else:
        d_uniq = np.zeros_like(uniq_x)
    d_sorted = d_uniq[inv_u]
    d_unsort = np.empty_like(d_sorted)
    d_unsort[idx] = d_sorted
    return d_unsort

def impute_nonfinite_to_colmed(X):
    X = np.asarray(X, float)
    bad = ~np.isfinite(X)
    if not bad.any():
        return X
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    rows, cols = np.where(bad)
    X[rows, cols] = med[cols]
    return X

def recompute_derived(df):
    df = df.copy()
    eps = 1e-30
    if 'dT_proxy' in df.columns:
        df['dT_proxy'] = np.clip(df['dT_proxy'].to_numpy(), eps, None)
    df['log_dT_proxy'] = np.log10(np.clip(df['dT_proxy'].to_numpy(), eps, None))
    df['log_Umag']     = np.log10(np.clip(df['Umag'].to_numpy(), 1e-12, None))
    df['log_Lx']       = np.log10(np.clip(df['Lx'].to_numpy(), 1e-12, None))
    if 'KE_Flux_Density' in df.columns:
        df['log_KE_flux'] = np.log10(np.clip(df['KE_Flux_Density'].to_numpy(), eps, None))
    for col in ['dT_proxy','vib_proxy','Umag','KE_Flux_Density']:
        if col in df.columns:
            mu = df[col].mean(); sd = df[col].std() + 1e-12
            df[f'{col}_z'] = (df[col] - mu) / sd
    return df

def add_noise(df, snr_db, seed=123):
    """dP_dyn,vib: additive; Umag: additive+clip>0; dT_proxy: multiplicative+clip>0."""
    rng = np.random.default_rng(seed)
    noisy = df.copy()
    snr_lin = 10.0**(snr_db/10.0)

    def add_additive(col):
        if col not in noisy.columns: return
        x = noisy[col].to_numpy(dtype=float)
        p_sig = np.mean(x**2)
        if p_sig <= 0: return
        sigma = np.sqrt(p_sig / snr_lin)
        noisy[col] = x + rng.normal(0.0, sigma, size=x.shape)

    for col in ['dP_dyn','vib_proxy']:
        add_additive(col)

    if 'Umag' in noisy.columns:
        add_additive('Umag')
        noisy['Umag'] = np.clip(noisy['Umag'].to_numpy(dtype=float), 1e-6, None)

    if 'dT_proxy' in noisy.columns:
        x = noisy['dT_proxy'].to_numpy(dtype=float)
        p_sig = np.mean(x**2)
        if p_sig > 0:
            sigma_add = np.sqrt(p_sig / snr_lin)
            rms_x = np.sqrt(p_sig)
            sigma_mult = sigma_add / (rms_x + 1e-12)
            mult = 1.0 + rng.normal(0.0, sigma_mult, size=x.shape)
            noisy['dT_proxy'] = x * mult
        noisy['dT_proxy'] = np.clip(noisy['dT_proxy'].to_numpy(dtype=float), 1e-30, None)

    return recompute_derived(noisy)

# Stage-1 features
BASE_FEATS_FULL = [
    'dP_dyn','dT_proxy','vib_proxy','Umag',
    'log_dT_proxy','log_Umag','log_Lx','log_KE_flux',
    'dT_proxy_z','vib_proxy_z','Umag_z','KE_Flux_Density_z'
]
BASE_FEATS_MIN = ['log_dT_proxy','log_Umag','log_Lx']

# ---------- Target transforms for Stage 2 ----------
def make_target_transform(mode, y):
    if mode == 'gamma':  # linear
        return y.copy(), lambda z: z
    elif mode == 'log10':
        return np.log10(np.clip(y, 1e-12, None)), lambda z: np.power(10.0, z)
    elif mode == 'log1p':  # robust for nonnegative, zero-inflated
        return np.log1p(np.clip(y, 0.0, None)), lambda z: np.expm1(z)
    elif mode == 'signedlog1p':  # handles negatives if they ever appear
        s = np.sign(y); ay = np.abs(y)
        return s * np.log1p(ay), lambda z: np.sign(z) * (np.expm1(np.abs(z)))
    else:
        raise ValueError(f"Unknown stage2_target: {mode}")

# ---------- Gamma sanity stats ----------
def print_gamma_stats(df, label="ALL"):
    if 'Gamma' not in df.columns:
        print(f"# Gamma stats ({label}): 'Gamma' column not found.")
        return
    g = df['Gamma'].to_numpy()
    desc = pd.Series(g).describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    frac_nonpos = float(np.mean(g <= 0))
    print(f"# Gamma stats ({label})")
    print(desc.to_string())
    print(f"  <= 0 fraction: {frac_nonpos:.4f}")

    def corr(df_in, bname):
        if bname not in df_in.columns:
            return f"{bname}: n/a"
        b = df_in[bname].to_numpy()
        if np.std(b) < 1e-18:
            return f"{bname}: constant"
        try:
            pear = float(np.corrcoef(g, b)[0,1])
        except Exception:
            pear = float('nan')
        try:
            spr = spearman_rho(g, b)
        except Exception:
            spr = float('nan')
        return f"{bname}: Pearson={pear:.4f}, Spearman={spr:.4f}"

    for c in ['Phi_v','Umag','KE_Flux_Density','dP_dyn']:
        print(" ", corr(df, c))
    if 'Phi_v' in df.columns:
        logPhi = np.log10(np.clip(df['Phi_v'].to_numpy(), 1e-30, None))
        print("  log10(Phi_v):", corr(df.assign(logPhi=logPhi), 'logPhi'))
    if 'Umag' in df.columns:
        logU = np.log10(np.clip(df['Umag'].to_numpy(), 1e-12, None))
        print("  log10(Umag):", corr(df.assign(logU=logU), 'logU'))

# ---------- Argparse / Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--test_files', nargs='+', required=True)
    ap.add_argument('--outdir', type=Path, default=Path('runs_output'))
    ap.add_argument('--snr_test', type=float, default=0.0)
    ap.add_argument('--snr_aug', type=str, default="")
    ap.add_argument('--rho', type=float, default=1000.0)
    ap.add_argument('--cp', type=float, default=4186.0)
    ap.add_argument('--min_feats', action='store_true')
    ap.add_argument('--stage1_mode', choices=['hybrid','ml','physics'], default='hybrid')
    ap.add_argument('--stage2_mode', choices=['cascade','oracle','hurdle'], default='cascade',
                    help='Stage-2 wiring: cascade (default), oracle (use true phi_v), or hurdle (γ==0 classifier + γ>0 regressor).')
    ap.add_argument('--stage2_target', choices=['gamma','log10','log1p','signedlog1p'], default='gamma',
                    help='Target transform for Stage 2; try log1p for zero-inflated heavy tails.')
    ap.add_argument('--stage2_feats', choices=['basic','rich'], default='rich',
                    help='Feature set for Stage 2 (rich adds geometry/grad/residual).')
    ap.add_argument('--stage2_calibration', choices=['none','affine','isotonic'], default='affine',
                    help='Post-hoc calibration on Stage-2 predictions (train-space).')
    ap.add_argument('--stage2_weight', choices=['none','sqrt_inv','cap1e5','balance_zero'], default='sqrt_inv',
                    help='Sample weighting for Stage 2 (regression part).')
    ap.add_argument('--stage2_loss', choices=['huber','absolute_error','quantile','squared_error'],
                    default='huber', help='Loss for Stage-2 GBRT.')
    ap.add_argument('--stage2_alpha', type=float, default=0.9,
                    help='Alpha for quantile/huber (quantile: tau in (0,1); huber default 0.9).')
    ap.add_argument('--stage2_clip', type=float, default=None,
                    help='If set, clip linear gamma at this value before target transform.')
    ap.add_argument('--gamma_stats', action='store_true',
                    help='Print sanity stats/correlations for Gamma (overall + test split) and exit early.')
    # compatibility shim: if user passes the old flag
    ap.add_argument('--stage2_oracle', action='store_true',
                    help='Deprecated: use --stage2_mode oracle.')

    args = ap.parse_args()
    if args.stage2_oracle:
        args.stage2_mode = 'oracle'

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    if 'file' not in df.columns:
        df['file'] = Path(args.input).name

    # NEW: ensure log10_phi_v exists
    if 'log10_phi_v' not in df.columns and 'Phi_v' in df.columns:
        df['log10_phi_v'] = np.log10(np.clip(df['Phi_v'].to_numpy(), 1e-30, None))

    FEATS = BASE_FEATS_MIN if args.min_feats else BASE_FEATS_FULL

    missing = [c for c in FEATS if c not in df.columns]
    if missing:
        print(f"# WARN: dropping missing Stage-1 features: {missing}")
        FEATS = [c for c in FEATS if c in df.columns]

    test_files = {Path(f).name for f in args.test_files}
    te_mask = df['file'].isin(test_files)
    tr_mask = ~te_mask

    # physics-consistent dT_proxy on base table
    df['dT_proxy'] = (df['Phi_v'].to_numpy() * df['Lx'].to_numpy()) / (
        float(args.rho) * float(args.cp) * np.clip(df['Umag'].to_numpy(), 1e-6, None)
    )
    df = recompute_derived(df)

    # Optional: Gamma sanity stats
    if args.gamma_stats:
        print_gamma_stats(df, label="ALL")
        print_gamma_stats(df.loc[te_mask], label="TEST")
        return

    # Stage 1: train (with optional aug)
    train_df = df.loc[tr_mask].copy()
    if args.snr_aug.strip():
        aug = [train_df]
        for tok in args.snr_aug.split(','):
            tok = tok.strip()
            if not tok: continue
            try: snr = float(tok)
            except ValueError: continue
            aug.append(add_noise(train_df, snr_db=snr, seed=int(snr*10)))
        train_df = pd.concat(aug, ignore_index=True)

    # Stage 1: test (optional test-time noise)
    test_df = df.loc[te_mask].copy()
    if args.snr_test > 0:
        test_df = add_noise(test_df, snr_db=args.snr_test, seed=123)

    X1tr = train_df[FEATS].to_numpy()
    y1tr = train_df['log10_phi_v'].to_numpy()
    X1te = test_df[FEATS].to_numpy()
    y1te = test_df['log10_phi_v'].to_numpy()

    # Stage-1 physics baseline (direct)
    phi_phys_tr = train_df['dT_proxy'].to_numpy() * (float(args.rho) * float(args.cp)) * \
                  np.clip(train_df['Umag'].to_numpy(), 1e-6, None) / np.clip(train_df['Lx'].to_numpy(), 1e-12, None)
    y1_phys_tr = np.log10(np.clip(phi_phys_tr, 1e-30, None))
    phi_phys_te = test_df['dT_proxy'].to_numpy() * (float(args.rho) * float(args.cp)) * \
                  np.clip(test_df['Umag'].to_numpy(), 1e-6, None) / np.clip(test_df['Lx'].to_numpy(), 1e-12, None)
    y1_phys_te = np.log10(np.clip(phi_phys_te, 1e-30, None))
    m_phys = metrics(y1te, y1_phys_te)
    print("# Sanity (physics, no ML) on TEST split")
    print(f"  Stage1 phys-only: R2={m_phys['R2']:.4g}, RMSE={m_phys['RMSE']:.4g}, rho={m_phys['rho']:.4g}")

    # Stage-1 fit
    if args.stage1_mode == 'physics':
        y1hat_tr, y1hat_te = y1_phys_tr, y1_phys_te
    elif args.stage1_mode == 'ml':
        y1hat_te, stage1 = fit_predict_stage1(X1tr, y1tr, X1te); y1hat_tr = stage1.predict(X1tr)
    else:  # hybrid residual
        resid_tr = y1tr - y1_phys_tr
        resid_te_hat, stage1 = fit_predict_stage1(X1tr, resid_tr, X1te)
        y1hat_tr = y1_phys_tr + stage1.predict(X1tr)
        y1hat_te = y1_phys_te + resid_te_hat

    m1 = metrics(y1te, y1hat_te)

    pd.DataFrame({
        'file': test_df['file'].values,
        'X': test_df['X'].values, 'Y': test_df['Y'].values, 'Z': test_df['Z'].values,
        'true_log10phi': y1te, 'pred_log10phi': y1hat_te, 'phys_log10phi': y1_phys_te
    }).to_csv(outdir / 'preds_stage1.csv', index=False)

    # ===== Stage 2 =====
    # pick phi for Stage 2 (cascade vs oracle)
    phi_hat_tr = np.power(10.0, y1hat_tr)
    phi_hat_te = np.power(10.0, y1hat_te)
    if args.stage2_mode == 'oracle':
        phi_tr_for_stage2 = train_df['Phi_v'].to_numpy()
        phi_te_for_stage2 = test_df['Phi_v'].to_numpy()
    else:
        phi_tr_for_stage2 = phi_hat_tr
        phi_te_for_stage2 = phi_hat_te

    rho, cp = float(args.rho), float(args.cp)
    dT_hat_tr = (phi_tr_for_stage2 * train_df['Lx'].to_numpy()) / (rho * cp * np.clip(train_df['Umag'].to_numpy(), 1e-6, None))
    dT_hat_te = (phi_te_for_stage2 * test_df['Lx'].to_numpy())  / (rho * cp * np.clip(test_df['Umag'].to_numpy(),  1e-6, None))

    # local std of phi
    tr_X = train_df['X'].to_numpy(); te_X = test_df['X'].to_numpy()
    tr_sort = np.argsort(tr_X); te_sort = np.argsort(te_X)
    tr_unsort = np.empty_like(tr_sort); tr_unsort[tr_sort] = np.arange(len(tr_sort))
    te_unsort = np.empty_like(te_sort); te_unsort[te_sort] = np.arange(len(te_sort))
    phi_std_tr = local_std(phi_tr_for_stage2[tr_sort])[tr_unsort]
    phi_std_te = local_std(phi_te_for_stage2[te_sort])[te_unsort]

    # --- geometry + gradient proxies (cheap) ---
    def norm01(a):
        a = np.asarray(a, float); amin = np.min(a); amax = np.max(a)
        return (a - amin) / (amax - amin + 1e-12)

    x_tr = train_df['X'].to_numpy(); y_tr = train_df['Y'].to_numpy(); z_tr = train_df['Z'].to_numpy()
    x_te = test_df['X'].to_numpy();  y_te = test_df['Y'].to_numpy();  z_te = test_df['Z'].to_numpy()
    xnorm_tr = norm01(x_tr); xnorm_te = norm01(x_te)
    r_tr = np.sqrt(y_tr**2 + z_tr**2); r_te = np.sqrt(y_te**2 + z_te**2)
    rnorm_tr = r_tr / (np.max(r_tr) + 1e-12); rnorm_te = r_te / (np.max(r_te) + 1e-12)

    dlogU_dx_tr   = derivative_along_x(x_tr, np.log(np.clip(train_df['Umag'].to_numpy(), 1e-12, None)), k=9)
    dlogU_dx_te   = derivative_along_x(x_te, np.log(np.clip(test_df['Umag'].to_numpy(),  1e-12, None)), k=9)
    dlogPhi_dx_tr = derivative_along_x(x_tr, np.log(np.clip(phi_tr_for_stage2, 1e-30, None)), k=9)
    dlogPhi_dx_te = derivative_along_x(x_te, np.log(np.clip(phi_te_for_stage2, 1e-30, None)), k=9)

    # Stage-1 residual (log domain)
    r1_tr = y1hat_tr - y1_phys_tr
    r1_te = y1hat_te - y1_phys_te

    eps = 1e-30
    def safe_col(df_part, name):
        return df_part[name].to_numpy() if name in df_part.columns else np.zeros(len(df_part))

    # Build Stage-2 features
    if args.stage2_feats == 'basic':
        X2tr = np.column_stack([
            np.log10(np.clip(phi_tr_for_stage2, 1e-30, None)),
            np.log10(np.clip(phi_std_tr,        1e-30, None)),
            np.log10(np.clip(train_df['Umag'].to_numpy(), 1e-12, None)),
            np.log10(np.clip(train_df['dP_dyn'].to_numpy(), 1e-30, None)),
            np.log10(np.clip(dT_hat_tr,         1e-30, None))
        ])
        X2te = np.column_stack([
            np.log10(np.clip(phi_te_for_stage2, 1e-30, None)),
            np.log10(np.clip(phi_std_te,        1e-30, None)),
            np.log10(np.clip(test_df['Umag'].to_numpy(), 1e-12, None)),
            np.log10(np.clip(test_df['dP_dyn'].to_numpy(), 1e-30, None)),
            np.log10(np.clip(dT_hat_te,         1e-30, None))
        ])
    else:
        X2tr = np.column_stack([
            np.log10(np.clip(phi_tr_for_stage2, 1e-30, None)),
            np.log10(np.clip(phi_std_tr,        1e-30, None)),
            np.log10(np.clip(train_df['Umag'].to_numpy(), 1e-12, None)),
            np.log10(np.clip(train_df['dP_dyn'].to_numpy(), 1e-30, None)),
            np.log10(np.clip(dT_hat_tr,         1e-30, None)),
            np.log10(np.clip(safe_col(train_df,'KE_Flux_Density'), eps, None)),
            np.log10(np.clip(phi_tr_for_stage2 / (safe_col(train_df,'KE_Flux_Density') + 1e-12), eps, None)),
            xnorm_tr, rnorm_tr, dlogU_dx_tr, dlogPhi_dx_tr,
            r1_tr,
            safe_col(train_df,'Umag_z'), safe_col(train_df,'KE_Flux_Density_z')
        ])
        X2te = np.column_stack([
            np.log10(np.clip(phi_te_for_stage2, 1e-30, None)),
            np.log10(np.clip(phi_std_te,        1e-30, None)),
            np.log10(np.clip(test_df['Umag'].to_numpy(), 1e-12, None)),
            np.log10(np.clip(test_df['dP_dyn'].to_numpy(), 1e-30, None)),
            np.log10(np.clip(dT_hat_te,         1e-30, None)),
            np.log10(np.clip(safe_col(test_df,'KE_Flux_Density'), eps, None)),
            np.log10(np.clip(phi_te_for_stage2 / (safe_col(test_df,'KE_Flux_Density') + 1e-12), eps, None)),
            xnorm_te, rnorm_te, dlogU_dx_te, dlogPhi_dx_te,
            r1_te,
            safe_col(test_df,'Umag_z'), safe_col(test_df,'KE_Flux_Density_z')
        ])

    # Impute any non-finite to keep GBRT happy
    X2tr = impute_nonfinite_to_colmed(X2tr)
    X2te = impute_nonfinite_to_colmed(X2te)

    # Targets (linear for weights/clip; transformed for training)
    y2tr_raw = train_df['Gamma'].to_numpy()
    y2te_raw = test_df['Gamma'].to_numpy()
    if args.stage2_clip is not None:
        y2tr_raw = np.minimum(y2tr_raw, args.stage2_clip)

    y2tr, inv2 = make_target_transform(args.stage2_target, y2tr_raw)
    y2te, _    = make_target_transform(args.stage2_target, y2te_raw)

    # Sample weights for Stage 2 (on linear γ scale)
    def make_weights(mode, y_lin):
        if mode == 'none':
            return np.ones_like(y_lin, dtype=float)
        if mode == 'sqrt_inv':
            return 1.0 / np.sqrt(1.0 + np.clip(y_lin, 0.0, None))
        if mode == 'cap1e5':
            return np.minimum(1.0, 1e5 / (1e-12 + np.clip(y_lin, 0.0, None)))
        if mode == 'balance_zero':
            y0 = (y_lin <= 0).astype(float)
            p0 = y0.mean()
            w = np.ones_like(y_lin, dtype=float)
            if p0 > 0:
                w[y0==1] = 0.5 / p0
            if (1-p0) > 0:
                w[y0==0] = 0.5 / (1 - p0)
            return w
        return np.ones_like(y_lin, dtype=float)

    # ---- Stage 2: three modes ----
    print(f"# Stage 2 mode: {args.stage2_mode} | target={args.stage2_target} | feats={args.stage2_feats} | weight={args.stage2_weight}")

    def build_stage2_regressor():
        # GBRT regressor with selected loss
        loss = args.stage2_loss
        alpha = args.stage2_alpha if loss in ('quantile','huber') else None
        model = GradientBoostingRegressor(
            loss=loss, alpha=(alpha if alpha is not None else 0.9),
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
            random_state=0
        )
        return model

    if args.stage2_mode in ('cascade','oracle'):
        w2 = make_weights(args.stage2_weight, y2tr_raw)
        stage2 = build_stage2_regressor()
        stage2.fit(X2tr, y2tr, sample_weight=w2)
        y2hat_te = stage2.predict(X2te)

        # Optional calibration using training predictions
        y2hat_tr = stage2.predict(X2tr)
        if args.stage2_calibration == 'affine':
            lr = LinearRegression().fit(y2hat_tr.reshape(-1,1), y2tr)
            y2hat_te = lr.predict(y2hat_te.reshape(-1,1))
        elif args.stage2_calibration == 'isotonic':
            ir = IsotonicRegression(out_of_bounds='clip').fit(y2hat_tr, y2tr)
            y2hat_te = ir.transform(y2hat_te)

        m2 = metrics(y2te, y2hat_te)

        # Save predictions
        pd.DataFrame({
            'file': test_df['file'].values,
            'X': test_df['X'].values, 'Y': test_df['Y'].values, 'Z': test_df['Z'].values,
            f'true_{args.stage2_target}gamma': y2te,
            f'pred_{args.stage2_target}gamma': y2hat_te,
            'true_gamma': y2te_raw, 'pred_gamma': inv2(y2hat_te)
        }).to_csv(outdir / 'preds_stage2.csv', index=False)

        # Save metrics.json
        with open(outdir / 'metrics.json', 'w') as f:
            json.dump({'stage1': m1, 'stage2': m2,
                       'stage1_mode': args.stage1_mode,
                       'stage2_mode': args.stage2_mode,
                       'stage2_target': args.stage2_target,
                       'stage2_feats': args.stage2_feats,
                       'stage2_calibration': args.stage2_calibration,
                       'stage2_weight': args.stage2_weight,
                       'stage2_loss': args.stage2_loss,
                       'stage2_alpha': args.stage2_alpha,
                       'stage2_clip': args.stage2_clip,
                       'snr_test_db': args.snr_test,
                       'snr_aug_db': args.snr_aug,
                       'min_feats': args.min_feats}, f, indent=2)

        # Console table
        print("# Table 1 (paste into paper)")
        print("| Target | R² | MAE | RMSE | logRMSE | ρ | Notes |")
        print("|---|---:|---:|---:|---:|---:|---|")
        print(f"| $\\log_{{10}}\\phi_v$ | {fmt(m1['R2'])} | {fmt(m1['MAE'])} | {fmt(m1['RMSE'])} | {fmt(m1['RMSE'])} | {fmt(m1['rho'])} | mode={args.stage1_mode}; test SNR={args.snr_test} dB; train aug={args.snr_aug}; min_feats={args.min_feats} |")
        logrmse = m2['RMSE'] if args.stage2_target == 'log10' else float('nan')
        print(f"| $\\gamma$ ({args.stage2_target}) | {fmt(m2['R2'])} | {fmt(m2['MAE'])} | {fmt(m2['RMSE'])} | {fmt(logrmse)} | {fmt(m2['rho'])} | Stage2={args.stage2_mode}, feats={args.stage2_feats}, calib={args.stage2_calibration}, weight={args.stage2_weight}, loss={args.stage2_loss}, alpha={args.stage2_alpha} |")

    else:  # ---- HURDLE: classify γ==0, regress γ>0 ----
        # Classifier for zero vs positive
        ybin_tr = (y2tr_raw > 0).astype(int)
        ybin_te = (y2te_raw > 0).astype(int)
        # class-balance weights
        p1 = ybin_tr.mean()
        w_cls = np.where(ybin_tr==1, 0.5/max(p1,1e-6), 0.5/max(1-p1,1e-6))
        clf = GradientBoostingClassifier(
            learning_rate=0.05, n_estimators=300, max_depth=3, subsample=0.8, random_state=0
        )
        clf.fit(X2tr, ybin_tr, sample_weight=w_cls)
        ppos_te = clf.predict_proba(X2te)[:,1]
        # classifier metrics
        try:
            auc = float(roc_auc_score(ybin_te, ppos_te))
        except Exception:
            auc = float('nan')
        try:
            ap = float(average_precision_score(ybin_te, ppos_te))
        except Exception:
            ap = float('nan')

        # Regressor on positives only
        pos_tr = (ybin_tr == 1)
        X2tr_pos = X2tr[pos_tr]
        y2tr_pos_raw = y2tr_raw[pos_tr]
        y2tr_pos, inv2_pos = make_target_transform(args.stage2_target, y2tr_pos_raw)
        # weights for positives (balance_zero not meaningful here)
        wmode = args.stage2_weight if args.stage2_weight != 'balance_zero' else 'sqrt_inv'
        w2_pos = make_weights(wmode, y2tr_pos_raw)
        reg = GradientBoostingRegressor(
            loss=(args.stage2_loss),
            alpha=(args.stage2_alpha if args.stage2_loss in ('quantile','huber') else None) or 0.9,
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0
        )
        reg.fit(X2tr_pos, y2tr_pos, sample_weight=w2_pos)

        # Predictions:
        #  - Conditional estimate on all test points (in transform space)
        y2hat_te_cond = reg.predict(X2te)
        #  - Optional calibration (fit on positives-only train)
        if args.stage2_calibration == 'affine':
            y2hat_tr_pos = reg.predict(X2tr_pos)
            lr = LinearRegression().fit(y2hat_tr_pos.reshape(-1,1), y2tr_pos)
            y2hat_te_cond = lr.predict(y2hat_te_cond.reshape(-1,1))
        elif args.stage2_calibration == 'isotonic':
            y2hat_tr_pos = reg.predict(X2tr_pos)
            ir = IsotonicRegression(out_of_bounds='clip').fit(y2hat_tr_pos, y2tr_pos)
            y2hat_te_cond = ir.transform(y2hat_te_cond)

        # Combine to overall expectation in *linear* gamma:
        gamma_hat_cond_lin = inv2_pos(y2hat_te_cond)  # E[gamma | gamma>0, X]
        gamma_hat_overall_lin = ppos_te * gamma_hat_cond_lin  # P(gamma>0|X) * E[gamma | gamma>0, X]
        # Back to chosen transform to compute "overall" metrics fairly
        if args.stage2_target == 'log1p':
            y2hat_overall = np.log1p(gamma_hat_overall_lin)
        elif args.stage2_target == 'log10':
            y2hat_overall = np.log10(np.clip(gamma_hat_overall_lin, 1e-12, None))
        elif args.stage2_target == 'gamma':
            y2hat_overall = gamma_hat_overall_lin
        else:  # signedlog1p not expected here (γ>=0), but handle gracefully
            y2hat_overall = np.log1p(gamma_hat_overall_lin)

        # Metrics:
        # - Overall in transform space against y2te (transform of true γ):
        m2_overall = metrics(y2te, y2hat_overall)
        # - Positives-only regression metrics in transform space:
        pos_te = (y2te_raw > 0)
        m2_pos = metrics(y2te[pos_te], y2hat_te_cond[pos_te]) if pos_te.any() else dict(R2=float('nan'), MAE=float('nan'), RMSE=float('nan'), rho=float('nan'))

        # Save predictions
        out = pd.DataFrame({
            'file': test_df['file'].values,
            'X': test_df['X'].values, 'Y': test_df['Y'].values, 'Z': test_df['Z'].values,
            f'true_{args.stage2_target}gamma': y2te,
            'ybin_true': ybin_te,
            'p_pos': ppos_te,
            f'pred_{args.stage2_target}gamma_overall': y2hat_overall,
            f'pred_{args.stage2_target}gamma_cond': y2hat_te_cond,
            'true_gamma': y2te_raw,
            'pred_gamma_overall': gamma_hat_overall_lin,
            'pred_gamma_cond': gamma_hat_cond_lin
        })
        out.to_csv(outdir / 'preds_stage2.csv', index=False)

        # Save metrics.json
        with open(outdir / 'metrics.json', 'w') as f:
            json.dump({
                'stage1': m1,
                'stage2_overall': m2_overall,
                'stage2_pos_only': m2_pos,
                'stage2_cls': {'auc': auc, 'ap': ap},
                'stage1_mode': args.stage1_mode,
                'stage2_mode': args.stage2_mode,
                'stage2_target': args.stage2_target,
                'stage2_feats': args.stage2_feats,
                'stage2_calibration': args.stage2_calibration,
                'stage2_weight': args.stage2_weight,
                'stage2_loss': args.stage2_loss,
                'stage2_alpha': args.stage2_alpha,
                'stage2_clip': args.stage2_clip,
                'snr_test_db': args.snr_test,
                'snr_aug_db': args.snr_aug,
                'min_feats': args.min_feats
            }, f, indent=2)

        # Console table (report positives-only regression + classifier AUC/AP)
        print("# Table 1 (paste into paper)")
        print("| Target | R² | MAE | RMSE | logRMSE | ρ | Notes |")
        print("|---|---:|---:|---:|---:|---:|---|")
        print(f"| $\\log_{{10}}\\phi_v$ | {fmt(m1['R2'])} | {fmt(m1['MAE'])} | {fmt(m1['RMSE'])} | {fmt(m1['RMSE'])} | {fmt(m1['rho'])} | mode={args.stage1_mode}; test SNR={args.snr_test} dB; train aug={args.snr_aug}; min_feats={args.min_feats} |")
        logrmse = m2_pos['RMSE'] if args.stage2_target == 'log10' else float('nan')
        note = f"HURDLE: AUC={fmt(auc)}, AP={fmt(ap)}; feats={args.stage2_feats}, calib={args.stage2_calibration}, weight={args.stage2_weight}, loss={args.stage2_loss}, alpha={args.stage2_alpha}"
        print(f"| $\\gamma$ ({args.stage2_target}, pos-only) | {fmt(m2_pos['R2'])} | {fmt(m2_pos['MAE'])} | {fmt(m2_pos['RMSE'])} | {fmt(logrmse)} | {fmt(m2_pos['rho'])} | {note} |")
        # (If you want the "overall expectation" row instead, swap to m2_overall and y2hat_overall.)

if __name__ == '__main__':
    main()
