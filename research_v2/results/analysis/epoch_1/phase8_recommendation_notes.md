# Phase 8 Recommendation Notes

The recommendation engine consumes only Phase 7 v2 primary-matrix summaries.
Memory is reported as a proxy estimate derived from dataset shape and configured replay storage.
Scoring is empirical-first: average accuracy, forgetting, runtime fit, and memory fit dominate the score.
Task similarity contributes only a small tie-breaker through method traits.
Joint training is excluded only when the request disallows joint retraining.

## Fixed Case Studies

- `permuted_mnist`: memory=32.0 MB, compute=low, forgetting<=40.0, similarity=high, joint_allowed=False
- `split_cifar10`: memory=128.0 MB, compute=low, forgetting<=35.0, similarity=high, joint_allowed=False
- `split_cifar100`: memory=256.0 MB, compute=medium, forgetting<=25.0, similarity=medium, joint_allowed=False
- `split_mini_imagenet`: memory=512.0 MB, compute=medium, forgetting<=20.0, similarity=low, joint_allowed=False
- `split_mini_imagenet`: memory=4096.0 MB, compute=high, forgetting<=15.0, similarity=low, joint_allowed=True
