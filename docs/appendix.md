# Appendix: Validator Summary & Noise Boundaries

**Artifacts**
- Noise-free multi-snapshots table: `data/training_table_multi_snapshots.csv`
- Noisy stress-test artifact: `data/training_table_noisy_pp.csv`

**Validator (pass/fail)**
- Missing/duplicates: 0 / 0 (both)
- log10_phi_v consistency (p95 |Δ|): ~4.44e-16
- log1p_gamma consistency (p95 |Δ|): ~0
- gamma_positive == (Gamma > 0): 1.00 agreement

**Boundaries**
- Primary results use the **noise-free** artifact only.
- Noisy file is **appendix-only** for robustness illustration; clipping/winsorization applied there, not in core results.

**Reproducibility**
- Build commands + fixed seeds shown in README.
- SHA256 hashes of CSV artifacts included below.
