# Project Plan: Hydra Enhancements for Hourly NWM Corrections

Date: 2025-10-07
Owner: Mitchel Carson

## Summary
- Hybrid transformer (Hydra v2) targeting NWM v2 data (train 2010–2018, val 2019, test 2020) will replace the earlier 2010–2022 control experiment once the refreshed run completes.
- Trial 1 from the latest Optuna sweep is the new “run of record” for Watauga (RMSE 5.137, NSE 0.644, KGE 0.715); all writing/plotting now references this configuration.
- Immediate milestone is completing the thesis/WRR draft—including publication-ready visualizations that compare NWM, LSTM, and Hydra for Watauga—and locking reproducible assets for that narrative.
- Near-term scope is intentionally limited to Watauga; multi-site exploration is logged as future work once the single-site story is finalized.
- Coordination items (plain language summary, CV/material sharing) remain active alongside technical execution.

## Objectives
- Finalize Hydra v2 configuration (gain/bias head, heteroscedastic outputs, quantiles) and document reproducible training/evaluation pipelines for Watauga.
- Tune and validate the Watauga LSTM baseline to provide a credible comparison point for Hydra improvements.
- Maintain a reproducible experiment stack (data prep, configs, checkpoints, metrics) that supports thesis defensibility and manuscript requirements.
- Complete the WRR/thesis draft—including Plain Language Summary, Methods, Results, and Discussion—highlighting the new Trial 1 metrics and comparisons.
- Produce high-quality visualizations (hydrographs, residual distributions, reliability diagnostics, skill bar charts) for the Watauga runs and integrate them into LaTeX.
- Deliver diagrams and mathematical exposition of the Hydra transformer, plus Watauga-focused visualizations and a companion web application mock for that site.
- Document a clear outline for scaling the residual-correction workflow to additional USGS sites; execution is deferred to future work.

## Current Status Snapshot
- Data pipeline produces aligned hourly datasets (2010–2020) for training/validation/testing; residual labels verified.
- Hydra v2 implementation (`modeling/train_quick_transformer_torch.py`) is operational with quick-eval tooling.
- First full-run Hydra v2 sweep (#1) will retrain on 2010–2018 with 2019 validation and 2020 evaluation (NWM v2 only) to establish the refreshed baseline before iterating on calibration tweaks.
- LSTM baseline script exists but requires hyperparameter sweeps to avoid divergence and to match Hydra preprocessing.
- WRR LaTeX template identified; outline discussions established (four-author format, title flexibility).
- Visual assets are limited to diagnostic plots; publication-grade figures outstanding.

## Workstreams & Key Tasks

### 1. Hydra v2 Finalization & Calibration
- Run targeted sweeps over gain_scale (0.03–0.08), log-variance clamps, dropout, and quantile weights using quick and full data splits.
- Analyse calibration diagnostics (coverage, PIT, CRPS) and adjust heteroscedastic heads to achieve nominal quantile coverage.
- Document final configuration, training schedule, and metrics; archive checkpoints and logs for reproducibility.

### 2. Baseline Refresh & Comparative Benchmarks
- Reproduce Hydra v2 quick-run for Watauga sanity checks prior to full sweeps.
- Tune `modeling/train_quick_lstm_torch.py` (hidden size, depth, LR schedule, warmup) to stabilise training and beat NWM baseline on Watauga.
- Compute comparative metrics (RMSE, MAE, NSE, KGE, PBIAS, correlation) across validation and holdout periods for Hydra vs. LSTM vs. NWM (Watauga-only dataset).
- Maintain experiment tracker summarizing hyperparameters, seeds, and outcomes.

### 3. WRR Paper Draft (LaTeX)
- Set up WRR template workspace, confirm author order, and update metadata (title, affiliations).
- Draft Plain Language Summary, Introduction, and Data/Methods sections with emphasis on the hydrologic context and transformer architecture.
- Create three-level architecture explanation (conceptual diagram → component schematic → mathematical formulas for attention, FiLM conditioning, gain/bias head, heteroscedastic loss) suitable for general reviewers.
- Write Results/Discussion sections anchored on Trial 1 (Hydra) vs. LSTM vs. NWM metrics; highlight calibration findings and residual behaviour.
- Maintain bibliography via BibTeX and track writing tasks in a shared checklist; circulate full draft to Mohammad before next Wednesday for feedback.

### 4. Visualization & Figure Production
- Define figure list for WRR submission (architecture diagram, workflow overview, performance comparisons, residual distributions, case-study hydrographs, station maps) with Watauga as the single focal basin.
- Standardize plotting style (fonts, colour palettes, labelling) to satisfy WRR/thesis guidelines and broaden accessibility for non-CS reviewers.
- Extend plotting utilities into a reusable suite that: (a) exports publication-ready PDFs with consistent styling; (b) renders the Watauga watershed context map; (c) surfaces calibration diagnostics driven by Trial 1 outputs. Document how multi-site support will be added later.
- Produce Taylor diagrams, reliability plots, and PIT histograms to support calibration analysis.
- Prepare table templates (LaTeX + CSV) for metrics and data summaries; validate formatting against WRR constraints.

### 5. Web Application & Repository Experience
- Scaffold a lightweight web application (e.g., Streamlit/FastAPI) that allows users to inspect the Watauga site, trigger data pulls, inspect saved model outputs, and view core visualizations; note multi-site selection as a later enhancement.
- Expose model comparison dashboards (Hydra vs. LSTM vs. NWM) for Watauga and link to documentation for reproducible training commands.
- Align GitHub repository structure with manuscript: high-level overview in docs, detailed instructions in README/tutorial notebooks, architecture section mirrored across mediums.
- Capture walkthrough/demo plan to accompany thesis defense and stakeholder reviews.

### 6. Coordination & Logistics
- Follow up on CV/material exchange with collaborators; store documents in agreed location.
- Maintain meeting notes and decision log (e.g., under `docs/meetings/`) for traceability.
- Share weekly status reports covering progress on modeling, writing, and figure production.

## Deliverables
- Hydra v2 “run of record” for Watauga with configuration files, checkpoints, and evaluation metrics (Trial 1 snapshot documented and archived).
- Tuned Watauga LSTM baseline with documented hyperparameters, training curves, and comparison metrics.
- WRR/thesis draft containing Plain Language Summary, Data/Methods, and full Results/Discussion referencing the latest metrics; ready for advisor review by next Wednesday.
- Figure bundle (PDF/PNG + generation scripts) aligned with manuscript narrative, including Watauga hydrographs, residual plots, reliability diagnostics, watershed map, and architecture diagrams.
- Interactive web application demonstrating Watauga model comparison and visualization capabilities (multi-site features deferred).
- Updated project documentation (this plan, status logs, experiment tracker).

## Dependencies & Risks
- **Compute/GPU availability:** Hydra sweeps require reliable GPU resources; secure access for overnight runs.
- **Data completeness:** Ensure NWM/USGS coverage through 2020; monitor for missing hours or corrupt pulls.
  - Mitigation: automated QA scripts and reruns for gaps.
- **Overfitting risks:** Use validation monitoring, early stopping, and dropout/weight decay to prevent overfitting on limited station data.
- **Writing bandwidth:** Balance coding and writing tasks; schedule protected writing blocks.
- **Visualization tooling:** Confirm access to diagramming/plotting tools; budget iteration time for supervisor feedback.

## Communication & Next Touchpoints
- Share weekly progress summaries (experiments, writing, visuals) with Mohammad.
- Schedule manuscript outline review after Week of Oct 14 – Oct 20 deliverables are drafted.
- Update this plan following major milestones or timeline adjustments.
