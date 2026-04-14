# Phase 6 Dashboard

Local launch:

```powershell
streamlit run Project/app/main.py
```

The dashboard reads precomputed Phase 5 artifacts from `results/analysis/phase5/`.
If those files are missing, regenerate them with:

```powershell
conda run -n genai python Project/experiments/run_phase5.py
```
