# G-Bridge: Pipe Manufacturing QA (noise-free)

[![status](https://joss.theoj.org/papers/acb19405f8c24865d39b92b32db63975/status.svg)](https://joss.theoj.org/papers/acb19405f8c24865d39b92b32db63975)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16890945.svg)](https://doi.org/10.5281/zenodo.16890945)

**Manufacturing QA for pipe flows:** Stage 1 predicts `log10(phi_v)` from plant-grade proxies; Stage 2 (hurdle) detects and ranks `gamma`. Reproducible scripts, audited metrics, and figures.  
**Scope (v1):** fixed geometry + high-SNR; *no runtime CFD*; noisy runs are **appendix-only** stress tests.

---

## Quickstart (v1, noise-free — used for Tables 1–2)

> RAW inputs used here:
> - `data/gamma_3D_OpenFOAM_processed_t1000_full_data.csv`
> - `data/gamma_3D_snapshot_Re2500_Fouling0.6_t5.0.csv`

```bash
# 0) (Optional) install
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt

# 1) Build the multi_snapshots training table (noise-free)
python3 csv_to_table.py \
  --inputs data/gamma_3D_OpenFOAM_processed_t1000_full_data.csv data/gamma_3D_snapshot_Re2500_Fouling0.6_t5.0.csv \
  --out data/training_table_multi_snapshots.csv

# 2) Validate integrity (pass/fail + math checks)
python3 gbridge_csv_validator.py --csv data/training_table_multi_snapshots.csv

# 3) Train (noise-free hurdle)
python3 train_stages.py --input data/training_table_multi_snapshots.csv \
  --test_files data/gamma_3D_OpenFOAM_processed_t1000_full_data.csv \
  --snr_test 0 --snr_aug "" --min_feats --stage1_mode hybrid \
  --stage2_mode hurdle --stage2_target log1p --stage2_feats rich \
  --stage2_weight sqrt_inv --stage2_calibration isotonic \
  --outdir runs_hurdle_clean

# 4) Verify + figures
python3 verify_run.py --outdir runs_hurdle_clean --input_csv data/training_table_multi_snapshots.csv \
  --test_files data/gamma_3D_OpenFOAM_processed_t1000_full_data.csv --stage2_mode hurdle --stage2_target log1p

python3 make_figures_from_preds.py --run_dir runs_hurdle_clean \
  --fig_dir y_bridge_pipeline_supporting_files/figures

python3 make_gamma_posonly_figs.py --run_dir runs_hurdle_clean \
  --fig_dir y_bridge_pipeline_supporting_files/figures
```

## Citation
If you use this software, please cite the archived release:

```bash
@software{gbridge_v1_0_0_zenodo,
  title     = {G-Bridge: Pipe Manufacturing QA (noise-free)},
  author    = {Terrero, Dickson},
  year      = {2025},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.16890945},
  url       = {https://doi.org/10.5281/zenodo.16890945},
  publisher = {Zenodo},
  note      = {Archived software release}
}
```

## Optional: Noisy stress test (appendix-only; not used for Tables 1–2)

<details>
<summary>Show commands</summary>

```bash
# Build noisy + one-shot post-processing (clipping/winsorization)
python3 csv_to_table.py \
  --inputs data/gamma_3D_OpenFOAM_processed_t1000_full_data.csv data/gamma_3D_snapshot_Re2500_Fouling0.6_t5.0.csv \
  --noise gaussian_6db --seed 123 \
  --pp-clip-nonneg --pp-winsorize Phi_v Gamma --pp-p 99.9 \
  --out data/training_table_noisy_pp.csv

# Validate noisy artifact
python3 gbridge_csv_validator.py --csv data/training_table_noisy_pp.csv

