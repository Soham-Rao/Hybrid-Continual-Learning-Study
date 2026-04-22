# Resource Summary Notes

- `runtime_sensitivity_summary.csv` is derived from per-task `time_sec` rows in run metrics CSVs.
- `memory_sensitivity_summary.csv` is proxy-based. It summarizes memory-relevant config knobs such as backbone, batch size, and replay buffer size.
- Peak VRAM and host RAM are not instrumented in the current v2 trainer, so those values are intentionally not claimed here.
