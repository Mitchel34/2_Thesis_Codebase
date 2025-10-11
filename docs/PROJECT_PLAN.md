# Project Plan: Hydra Enhancements for Hourly NWM Corrections

Date: 2025-10-07
Owner: Mitchel Carson

## Summary
- Hybrid transformer (Hydra v2) trained on NWM 2010–2020 with 2021 validation delivers ~10% RMSE improvement versus raw NWM; serves as current control experiment.
- Next milestone focuses on stabilizing Hydra v2 across full 2010–2022 runs, tightening calibration, and strengthening the LSTM baseline for comparative analysis.
- Work on Water Resources Research (WRR) manuscript, thesis narrative, and publication-grade visuals proceeds in parallel.
- Coordination items (plain language summary, CV/material sharing) remain active alongside technical execution.

## Objectives
- Finalize Hydra v2 configuration (gain/bias head, heteroscedastic outputs, quantiles) and document reproducible training/evaluation pipelines.
- Tune and validate the LSTM baseline to provide a credible comparison point for Hydra improvements.
- Maintain a reproducible experiment stack (data prep, configs, checkpoints, metrics) that supports thesis defensibility and manuscript requirements.
- Draft the WRR-format paper, emphasizing model architecture, methodology, results, and accessible narrative.
- Produce high-quality visualizations (plots, tables, diagrams) aligned with WRR and thesis deliverables while tracking collaborative tasks.

## Current Status Snapshot
- Data pipeline produces aligned hourly datasets (2010–2022) for training/validation/testing; residual labels verified.
- Hydra v2 implementation (`modeling/train_quick_transformer_torch.py`) is operational with quick-eval tooling.
- First full-run Hydra v2 sweep (#1) trained on 2010–2022 with 2022 evaluation delivers 65% RMSE reduction versus NWM (gain_scale=0.05, logvar clamp [-4, 2], quantile weight 0.1); lower tails are under-covered (q10≈0%) while q90 over-covers (≈100%), signalling calibration work.
- LSTM baseline script exists but requires hyperparameter sweeps to avoid divergence and to match Hydra preprocessing.
- WRR LaTeX template identified; outline discussions established (four-author format, title flexibility).
- Visual assets are limited to diagnostic plots; publication-grade figures outstanding.

## Workstreams & Key Tasks

### 1. Hydra v2 Finalization & Calibration
- Run targeted sweeps over gain_scale (0.03–0.08), log-variance clamps, dropout, and quantile weights using quick and full data splits.
- Analyse calibration diagnostics (coverage, PIT, CRPS) and adjust heteroscedastic heads to achieve nominal quantile coverage.
- Document final configuration, training schedule, and metrics; archive checkpoints and logs for reproducibility.

### 2. Baseline Refresh & Comparative Benchmarks
- Reproduce Hydra v2 quick-run for sanity checks prior to full sweeps.
- Tune `modeling/train_quick_lstm_torch.py` (hidden size, depth, LR schedule, warmup) to stabilise training and beat NWM baseline.
- Compute comparative metrics (RMSE, MAE, NSE, KGE, PBIAS, correlation) across validation and holdout periods for Hydra vs. LSTM vs. NWM.
- Maintain experiment tracker summarizing hyperparameters, seeds, and outcomes.

### 3. WRR Paper Draft (LaTeX)
- Set up WRR template workspace, confirm author order, and update metadata (title, affiliations).
- Draft Plain Language Summary, Introduction, and Data/Methods sections with emphasis on the hydrologic context and transformer architecture.
- Detail model components (TCN stem, FiLM conditioning, gain/bias head, heteroscedastic loss) with equations and narrative explanation.
- Outline Results/Discussion structure using current metrics; identify gaps requiring additional experiments.
- Maintain bibliography via BibTeX and track writing tasks in a shared checklist.

### 4. Visualization & Figure Production
- Define figure list for WRR submission (architecture diagram, workflow overview, performance comparisons, residual distributions, case-study hydrographs, station maps).
- Standardize plotting style (fonts, colour palettes, labelling) to satisfy WRR/thesis guidelines.
- Extend `modeling/plot_quick_eval.py` and `modeling/generate_model_improvement_plots.py` to export publication-ready PDFs with consistent styling.
- Produce Taylor diagrams, reliability plots, and PIT histograms to support calibration analysis.
- Prepare table templates (LaTeX + CSV) for metrics and data summaries; validate formatting against WRR constraints.

### 5. Coordination & Logistics
- Follow up on CV/material exchange with collaborators; store documents in agreed location.
- Maintain meeting notes and decision log (e.g., under `docs/meetings/`) for traceability.
- Share weekly status reports covering progress on modeling, writing, and figure production.

## Deliverables
- Hydra v2 “run of record” with configuration files, checkpoints, and evaluation metrics.
- Tuned LSTM baseline with documented hyperparameters, training curves, and comparison metrics.
- WRR manuscript draft containing Plain Language Summary, Data, Methods, and initial Results sections.
- Figure bundle (PDF/PNG + generation scripts) aligned with manuscript narrative.
- Updated project documentation (this plan, status logs, experiment tracker).

## Timeline (Oct – Nov 2025)
- **Week of Oct 7 – Oct 13**
  - Complete Hydra v2 quick-run regression tests; schedule calibration sweep.
  - Draft Plain Language Summary and manuscript outline in WRR template.
- **Week of Oct 14 – Oct 20**
  - Execute Hydra v2 calibration sweeps; log coverage diagnostics.
  - Tune LSTM baseline and assemble preliminary comparison table.
  - Populate Data & Methods sections with current architecture description.
- **Week of Oct 21 – Oct 27**
  - Finalise Hydra v2 configuration on full dataset; export metrics/plots.
  - Draft architecture diagrams and performance comparison visuals.
  - Write Results section skeleton with existing metrics; note pending experiments.
- **Week of Oct 28 – Nov 3**
  - Conduct ablation/sensitivity checks (feature subsets, sequence length) if time permits.
  - Polish figures, ensure reproducible plotting scripts, and integrate into LaTeX.
  - Expand Discussion/Conclusion drafts; solicit feedback from Mohammad.
- **Week of Nov 4 – Nov 10**
  - Address feedback, finalise first full manuscript draft, and prepare submission checklist.
  - Archive experiment artefacts; update plan post-review as needed.
- **Ongoing**
  - Weekly syncs, CV/material exchanges, and incremental plan updates as milestones shift.

## Dependencies & Risks
- **Compute/GPU availability:** Hydra sweeps require reliable GPU resources; secure access for overnight runs.
- **Data completeness:** Ensure NWM/USGS coverage through 2022; monitor for missing hours or corrupt pulls.
  - Mitigation: automated QA scripts and reruns for gaps.
- **Overfitting risks:** Use validation monitoring, early stopping, and dropout/weight decay to prevent overfitting on limited station data.
- **Writing bandwidth:** Balance coding and writing tasks; schedule protected writing blocks.
- **Visualization tooling:** Confirm access to diagramming/plotting tools; budget iteration time for supervisor feedback.

## Communication & Next Touchpoints
- Share weekly progress summaries (experiments, writing, visuals) with Mohammad.
- Schedule manuscript outline review after Week of Oct 14 – Oct 20 deliverables are drafted.
- Update this plan following major milestones or timeline adjustments.
