# γ-Bridge (gamma-bridge)

**Manufacturing QA for pipe flows:** Stage 1 predicts `log10(phi_v)` from plant-grade proxies; Stage 2 (hurdle) detects and ranks `gamma`. Reproducible scripts, audited metrics, and figures.

## Quickstart
```bash
# Build training table
python3 csv_to_table.py --inputs data/snapshot.csv data/full_case.csv \
  --out data/training_table_multi_snapshots.csv

# Train (noise-free hurdle)
python3 train_stages.py --input data/training_table_multi_snapshots.csv \
  --test_files gamma_3D_OpenFOAM_processed_t1000_full_data.csv \
  --snr_test 0 --snr_aug "" --min_feats --stage1_mode hybrid \
  --stage2_mode hurdle --stage2_target log1p --stage2_feats rich \
  --stage2_weight sqrt_inv --stage2_calibration isotonic \
  --outdir runs_hurdle_clean

# Verify + figures
python3 verify_run.py --outdir runs_hurdle_clean --input_csv data/training_table_multi_snapshots.csv \
  --test_files gamma_3D_OpenFOAM_processed_t1000_full_data.csv --stage2_mode hurdle --stage2_target log1p
python3 make_figures_from_preds.py --run_dir runs_hurdle_clean --fig_dir y_bridge_pipeline_supporting_files/figures
python3 make_gamma_posonly_figs.py --run_dir runs_hurdle_clean --fig_dir y_bridge_pipeline_supporting_files/figures

## License
Source code: PolyForm Noncommercial 1.0.0. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
Commercial use requires a separate license. For licensing inquiries, contact: dterrero@theticktheory.com
