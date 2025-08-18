---
title: "G-Bridge: Two-stage QA for viscous dissipation and vortex-stretching from factory sensor data"
authors:
  - name: Dickson A. Terrero
    orcid: 0009-0005-4170-149X
    affiliation: 1
affiliations:
  - name: Independent Researcher, Boston, USA
    index: 1
date: 17 August 2025
bibliography: paper.bib
tags: [quality assurance, turbulence, OpenFOAM, Python, manufacturing]
repository: https://github.com/dterrero/gbridge-pipe-qa
archive_doi: 10.5281/zenodo.16890945
---

# Summary

**G-Bridge** is an open-source Python pipeline for inline quality assurance (QA) of pipe flows that bridges factory-grade proxies—pressure drop, temperature delta, vibration, and line speed—to turbulence-aware diagnostics. The software follows a two-stage design: (i) a regressor maps proxies to $\log_{10}\phi_v$ (viscous dissipation); (ii) a hurdle model first detects $\gamma>0$ events and then regresses $\log(1{+}\gamma)$ on positives to estimate vortex-stretching magnitude. The package provides containers, unit tests, pre-trained checkpoints, and an audit script that regenerates tables and figures from saved predictions with input hash checking, enabling deterministic reproduction across environments.

## Methods

Stage 1 regresses $\log_{10}\phi_v$ from factory proxies; Stage 2 uses a hurdle design that (i) detects $\gamma>0$ and (ii) regresses $\log(1{+}\gamma)$ on the positives.

![Overview of G-Bridge and representative results. 
(A) Two-stage pipeline from factory proxies to physics-aware QA signals. 
(B) Stage 1 predicted vs. true in $\log_{10}\phi_v$. 
(C) Stage 2 predicted vs. true on positives in $\log(1+\gamma)$.](fig1_overview_results.png)


# Statement of need

QA lines routinely record proxy signals, yet the flow-physics indicators most aligned with product and process health—$\phi_v$ and $\gamma$—are not directly instrumented. Offline CFD can recover them but is too slow/costly for inline decisions. **G-Bridge** offers a lightweight, reproducible path from proxies to physics-aligned QA signals: Stage 1 yields a stable dissipation trend operators can threshold/monitor; Stage 2 produces a ranked list of $\gamma$-risk locations for targeted inspection. The implementation emphasizes operational fit (CSV/HDF5 I/O, CLI verbs for training/verification/figure regeneration) and auditability (fixed seeds, metrics JSON, and SHA-256 manifests). The current release targets fixed-geometry, high-SNR settings; noise-robust operation and broader geometries are deferred to future work. The code is Apache-2.0 and leverages widely used tooling (e.g., scikit-learn) and OpenFOAM-derived data for examples [@pedregosa2011; @jasak2009].

# Acknowledgements

No external funding. We thank open-source contributors to OpenFOAM and scikit-learn.

# References
