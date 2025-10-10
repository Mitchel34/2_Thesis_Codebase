# Project Plan: Foundation Model Extension for Hourly NWM Corrections

Date: 2025-10-07
Owner: Mitchel Carson

## Summary
- Hybrid transformer baseline trained on NWM 2010–2020 with 2021 validation achieves ~10% RMSE improvement against raw NWM; serves as control experiment.
- Next milestone is integrating a Hugging Face foundation model to push accuracy further and benchmark against LSTM and other baselines.
- Parallel effort required to draft Water Resources Research (WRR) manuscript, document architecture decisions, and produce publication-ready visuals.
- Coordination items (plain language summary, CV/material sharing) remain active alongside technical execution.

## Objectives
- Adapt and fine-tune a suitable time series foundation model on the hourly NWM/USGS residual dataset and quantify gains versus the hybrid transformer and LSTM baselines.
- Maintain a reproducible experiment stack (data prep, configs, checkpoints, metrics) to support thesis defensibility and manuscript requirements.
- Draft the WRR-format paper, emphasizing the model architecture, methodology, results, and plain-language summary.
- Create high-quality visualizations (plots, tables, architecture diagrams) that align with WRR submission standards and thesis deliverables.
- Track collaborative tasks (e.g., CV exchange, template alignment) to keep stakeholders informed.

## Current Status Snapshot
- Data pipeline produces aligned hourly datasets (2010–2022) for training/validation/testing; residual labels verified.
- Hybrid transformer implementation operational in PyTorch (`modeling/train_quick_transformer_torch.py`) with quick-eval tooling.
- Early analysis shows ~10% RMSE improvement on 2021 validation relative to raw NWM baselines.
- WRR LaTeX template identified; outline discussions established (four-author format, title flexibility).
- Visual assets currently limited to diagnostic plots; publication-grade figures outstanding.

## Workstreams & Key Tasks

### 1. Foundation Model Integration
- Identify candidate Hugging Face models (e.g., TimeGPT, Chronos, Temporal Fusion Transformer variants) and assess licensing, input requirements, and computational demands.
- Prototype data adapters to map hourly residual sequences into the foundation model format (windowing, feature scaling, static metadata handling).
- Fine-tune chosen model on 2010–2020 training data with 2021 validation; log checkpoints, configuration files, and training curves.
- Run comparative inference on 2022 test data; capture RMSE/NSE/KGE/PBias and qualitative diagnostics.
- Document findings and decision rationale for inclusion in thesis and paper.

### 2. Baseline Refresh & Comparative Benchmarks
- Reproduce hybrid transformer training runs to establish control metrics with up-to-date data splits and logging.
- Implement LSTM (and any other reference models already scoped) with consistent preprocessing and evaluation scripts.
  - New script: `modeling/train_quick_lstm_torch.py --data <path>` mirrors the transformer CLI; outputs saved under `data/clean/modeling/` for side-by-side metrics.
- Integrate a Hugging Face foundation transformer for residual correction.
  - New script: `modeling/train_hf_foundation.py --data <path>` loads a pre-trained `TimeSeriesTransformer` checkpoint (default `kashif/timeseries-transformer-tourism-hourly`) and fine-tunes it on the hourly residual dataset.
- Wrap the foundation backbone with Hydra hybrid heads for dual-output corrections.
  - New script: `modeling/train_hydra_foundation_torch.py --data <path>` composes `HydraFoundationModel` with the same loss stack while optionally freezing the pre-trained encoder.
- Ship Hydra v2 upgrades (TCN stem, FiLM conditioning, gain/bias + heteroscedastic heads) and validate on the January 2023 quick dataset.
  - Updated script: `modeling/train_quick_transformer_torch.py --output-prefix hydra_v2_quick` logs Gaussian-NLL losses and writes metrics to `hydra_v2_quick_metrics.json`.
- Create a comparison matrix summarizing performance across models and sites; flag statistically significant gains.
- Archive artefacts (models, logs, configs) in versioned storage for reproducibility.

### 3. WRR Paper Draft (LaTeX)
- Set up WRR template repository/workspace, confirm author order, and update metadata (title, affiliations).
- Draft Plain Language Summary, Introduction, and Data/Methods sections, incorporating current dataset description.
- Develop Model Architecture section with detailed diagrams, component descriptions (embedding, attention blocks, loss), and justification for foundation model integration.
- Outline Results section structure (baseline vs. foundation model, ablations, sensitivity) and populate with preliminary metrics.
- Capture Discussion, Conclusion, and Future Work placeholders; maintain bibliography in BibTeX.
- Track writing tasks in shared checklist; schedule internal review cadence with Mohammad.

### 4. Visualizations & Figure Production
- Define figure list for WRR submission (architecture diagram, training workflow, performance comparisons, error distributions, case-study hydrographs).
- Standardize plotting style (fonts, color palettes, labeling) to meet WRR guidelines and thesis formatting.
- Automate generation of core plots via `modeling/plot_quick_eval.py` extensions; save to `figures/` with versioning.
- Draft architecture diagram (e.g., using draw.io or Python graph libs) illustrating hybrid transformer and foundation model components.
- Prepare table templates (LaTeX + CSV) for metrics and data summaries; validate against WRR column width limits.

### 5. Coordination & Logistics
- Follow up on CV/material exchange with collaborators; store shared documents in project repo or agreed drive.
- Maintain meeting notes and decision log (e.g., in `docs/meetings/`) for traceability.
- Update weekly status reports covering model progress, writing, and figure readiness.

## Deliverables
- Fine-tuned foundation model checkpoint(s) with accompanying configuration, evaluation reports, and comparison plots.
- Refreshed hybrid transformer and LSTM benchmark packages (scripts, metrics, inference outputs).
- Draft WRR manuscript (LaTeX) with completed Abstract, Plain Language Summary, and Methods sections by end of October.
- Figure bundle (PNG/PDF + source scripts) aligned with manuscript narrative.
- Updated project documentation (this plan, status logs, experiment tracker).

## Timeline (Oct – Nov 2025)
- **Week of Oct 7 – Oct 13**
  - Finalize foundation model candidate selection and resource estimates.
  - Build data adapters/windowing pipelines; rerun hybrid transformer baseline for reference metrics.
  - Draft Plain Language Summary and manuscript outline in WRR template.
- **Week of Oct 14 – Oct 20**
  - Fine-tune foundation model on training set; begin hyperparameter sweeps.
  - Implement LSTM benchmark refresh and assemble preliminary comparison table.
  - Populate Data & Methods sections, including detailed architecture description draft.
- **Week of Oct 21 – Oct 27**
  - Complete foundation model tuning, run validation/test evaluations, and capture plots.
  - Draft architecture figure(s) and performance comparison visuals.
  - Write Results section skeleton with current metrics; note gaps for pending experiments.
- **Week of Oct 28 – Nov 3**
  - Conduct ablation/sensitivity checks (e.g., feature subsets, window lengths) if time permits.
  - Polish figures, ensure reproducible plotting scripts, and integrate into LaTeX.
  - Expand Discussion/Conclusion drafts; solicit feedback from Mohammad.
- **Week of Nov 4 – Nov 10**
  - Address feedback, finalize first full manuscript draft, and prepare submission checklist.
  - Archive experiment artefacts; update plan based on review outcomes.
- **Ongoing**
  - Weekly syncs, CV/material exchanges, and incremental updates to this plan as milestones shift.

## Dependencies & Risks
- **Compute/GPU availability:** Foundation model fine-tuning may require higher VRAM; secure access to cloud or lab resources.
- **Data completeness:** Ensure NWM/USGS data coverage through 2022; monitor for missing hours or corrupt pulls.
  - Mitigation: automated QA scripts and reruns for gaps.
- **Licensing/compliance:** Verify Hugging Face model licensing for academic publication use.
- **Writing bandwidth:** Balance coding and writing tasks; schedule protected writing blocks.
- **Visualization tooling:** Confirm access to preferred diagram/plotting tools; budget time for iteration based on supervisor feedback.

## Communication & Next Touchpoints
- Share weekly progress summary (experiments, writing, visuals) with Mohammad.
- Notify collaborators once foundation model candidate and compute plan are finalized.
- Schedule manuscript outline review after Week of Oct 14 – Oct 20 deliverables are drafted.
- Update this plan after major milestones or if timeline adjustments become necessary.
