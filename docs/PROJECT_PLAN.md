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
- Draft the WRR-format paper, emphasizing model architecture, methodology, results, and accessible narrative—complete review-ready draft for Mohammad by next Wednesday.
- Produce high-quality visualizations (plots, tables, diagrams) aligned with WRR and thesis deliverables while tracking collaborative tasks.
- Deliver diagrams and mathematical exposition of the Hydra transformer, plus hydrologic site visualizations and a companion web application that surfaces model comparisons for multiple stations.

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
- Create three-level architecture explanation (conceptual diagram → component schematic → mathematical formulas for attention, FiLM conditioning, gain/bias head, heteroscedastic loss) suitable for general reviewers.
- Outline Results/Discussion structure using current metrics; connect hypotheses to research questions and identify gaps requiring additional experiments.
- Maintain bibliography via BibTeX and track writing tasks in a shared checklist; circulate full draft to Mohammad before next Wednesday for feedback.

### 4. Visualization & Figure Production
- Define figure list for WRR submission (architecture diagram, workflow overview, performance comparisons, residual distributions, case-study hydrographs, station maps).
- Standardize plotting style (fonts, colour palettes, labelling) to satisfy WRR/thesis guidelines and broaden accessibility for non-CS reviewers.
- Extend plotting utilities into a reusable suite that: (a) exports publication-ready PDFs with consistent styling; (b) renders watershed/biome context maps for each focal station; (c) surfaces calibration diagnostics.
- Produce Taylor diagrams, reliability plots, and PIT histograms to support calibration analysis.
- Prepare table templates (LaTeX + CSV) for metrics and data summaries; validate formatting against WRR constraints.

### 5. Web Application & Repository Experience
- Scaffold a lightweight web application (e.g., Streamlit/FastAPI) that allows users to select sites, trigger data pulls, inspect saved model outputs, and view core visualizations.
- Expose model comparison dashboards (Hydra vs. LSTM vs. NWM) and link to documentation for reproducible training commands.
- Align GitHub repository structure with manuscript: high-level overview in docs, detailed instructions in README/tutorial notebooks, architecture section mirrored across mediums.
- Capture walkthrough/demo plan to accompany thesis defense and stakeholder reviews.

### 6. Coordination & Logistics
- Follow up on CV/material exchange with collaborators; store documents in agreed location.
- Maintain meeting notes and decision log (e.g., under `docs/meetings/`) for traceability.
- Share weekly status reports covering progress on modeling, writing, and figure production.

## Deliverables
- Hydra v2 “run of record” with configuration files, checkpoints, and evaluation metrics.
- Tuned LSTM baseline with documented hyperparameters, training curves, and comparison metrics.
- WRR manuscript draft containing Plain Language Summary, Data, Methods (with mathematical exposition), and initial Results sections; ready for advisor review by next Wednesday.
- Figure bundle (PDF/PNG + generation scripts) aligned with manuscript narrative, including watershed context maps and architecture diagrams.
- Interactive web application demonstrating site selection, model comparison, and visualization capabilities.
- Updated project documentation (this plan, status logs, experiment tracker).

## Timeline (Oct – Nov 2025)
- **Week of Oct 7 – Oct 13**
  - Complete Hydra v2 quick-run regression tests; schedule calibration sweep.
  - Draft Plain Language Summary and manuscript outline in WRR template.
  - Sketch three-level architecture explanation (conceptual, schematic, mathematical) and begin LaTeX diagram integration.
  - Stand up web app scaffold with site selection + cached data loading.
- **Week of Oct 14 – Oct 20**
  - Execute Hydra v2 calibration sweeps; log coverage diagnostics.
  - Tune LSTM baseline and assemble preliminary comparison table.
  - Populate Data & Methods sections with mathematical exposition and literature framing; ensure narrative threads bridge hypotheses → methods → results.
  - Generate watershed overview figures for Watauga plus two additional focal stations; prototype map styling.
- **Week of Oct 21 – Oct 27**
  - Finalise Hydra v2 configuration on full dataset; export metrics/plots for focal stations.
  - Flesh out Results section skeleton with cross-biome comparisons and calibration analysis.
  - Refine architecture diagrams, add equation appendix, and draft accessible explanation for general reviewers.
  - Extend web app with model comparison dashboards and visualization tabs; capture demo script.
- **Week of Oct 28 – Nov 3**
  - Conduct ablation/sensitivity checks (feature subsets, sequence length) if time permits.
  - Polish figure bundle (including watershed maps), ensure reproducible plotting scripts, and integrate visuals into LaTeX.
  - Expand Discussion/Conclusion drafts; align narrative transitions and emphasize site-selection rationale.
  - Prepare advisor hand-off package (manuscript PDF, figures, web app instructions) ahead of review meeting.
- **Week of Nov 4 – Nov 10**
  - Address advisor feedback, finalise manuscript and appendices, and prepare submission checklist.
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
