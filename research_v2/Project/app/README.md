# Dashboard v2 Workspace

This folder contains the v2 research workbench dashboard for the continual
learning study. The dashboard is:

- read-only
- sourced only from generated v2 artifacts
- intended for local exploration with Streamlit

## Run

From the repository root:

```powershell
streamlit run ".\research_v2\Project\app\main.py"
```

## Required artifact roots

- `research_v2/results/analysis/epoch_1`
- `research_v2/results/analysis/ablations/epoch_1`
- `research_v2/results/figures/epoch_1/analysis`

## Notes

- The dashboard prefers interactive Plotly charts rebuilt from CSV artifacts.
- Static Phase 7 PNGs are kept as archival fallback visuals.
- Phase 6 ablations are shown only as contextual evidence, not as the main study leaderboard.
