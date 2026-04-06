# Post-Report Notes

This file tracks findings and deferred cleanup items that should be revisited after the Phase 4 run + report deadline.

## Audit Findings to Revisit

- [ ] Re-run all `agem` results after the projection bug fix in [agem.py](/f:/Temp/cf/Catastrophic%20Forgetting/Project/src/methods/baselines/agem.py).
- [ ] Re-run all `agem_distill` results after the projection fix and the replay-policy correction in [agem_distill.py](/f:/Temp/cf/Catastrophic%20Forgetting/Project/src/methods/hybrid/agem_distill.py).
- [ ] Treat all pre-fix AGEM-family epoch-1 numbers as stale until rerun.
- [ ] Re-run `der` and `xder` families after the float32 replay-logit stabilization change in [der.py](/f:/Temp/cf/Catastrophic%20Forgetting/Project/src/methods/hybrid/der.py) and [xder.py](/f:/Temp/cf/Catastrophic%20Forgetting/Project/src/methods/hybrid/xder.py).
- [ ] Investigate `xder` numerical instability on Split CIFAR-100 before using it for strong claims.
- [ ] Keep old Split CIFAR-100 `si_der` pre-fix logs only as historical artifacts; use the fixed rerun summary for current reporting.

## Metric / Interpretation Notes

- [ ] The current FWT implementation uses a zero-shot score minus a trainer-supplied chance baseline, not a strict random-init baseline. Keep this wording consistent in future docs and paper drafts.
- [ ] Negative FWT values in Class-IL should be interpreted cautiously; do not over-index on them for rankings.

## Method-Faithfulness Notes

- [ ] `progress_compress` is a simplified shared-backbone variant, not a fully faithful reproduction of the original architectural formulation. Keep claims narrow.
- [ ] Re-check `icarl` and `progress_compress` wording in paper/report to ensure “project implementation variant” language is used where appropriate.

## Result-Table Hygiene

- [ ] Standardize seed counts in later paper tables. Current summaries include 3-seed and 4-seed exceptions.
- [ ] Clearly separate “report-deadline Phase 4 results” from “post-report full rerun campaign” in later writeups.
- [ ] Remove or quarantine stale intermediate logs/results once full reruns exist and replacements are confirmed.

## Tooling / Code Cleanup

- [ ] Upgrade deprecated AMP calls (`torch.cuda.amp.*`) to the newer `torch.amp.*` API after the report deadline.
- [ ] Expand automated tests beyond the current smoke coverage into method-specific behavioral checks where feasible.
- [ ] Add a result-validation script for NaNs, seed-count mismatches, and missing runs before the full rerun campaign.
