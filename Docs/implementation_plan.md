# Implementation Plan: CL Comparative Study — Catastrophic Forgetting

A rigorous comparative study of **hybrid continual learning (CL) methods** to identify which combinations of regularization, replay, and distillation best mitigate catastrophic forgetting across datasets of increasing complexity. The output is a reproducible benchmark, visual results, and a decision-support dashboard.

---

## Scope Summary

- **Research focus**: Hybrid CL methods (combinations of ≥2 strategies)
- **Standalone methods**: Included only as baseline reference rows in tables/plots
- **Scenarios**: Class-Incremental Learning (Class-IL) primarily; Permuted MNIST uses Domain-IL naturally
- **Compute split**: Local runs for small/medium datasets and Phase 4 Mini-ImageNet ResNet18; optional fallback cloud use for heavier ViT experiments
- **Frontend**: Streamlit dashboard built *after* all experiments conclude
- **Compute-bound dataset rationale**: The benchmark set is intentionally constrained by limited hardware. Mini-ImageNet ResNet18 is now planned locally on the RTX 4050 laptop, while heavier ViT/Tiny-ImageNet work may still use a fallback cloud path if needed.

---

## Compute & Platform Strategy

| Dataset | Model | Platform | Machine |
|---|---|---|---|
| Permuted MNIST, Split CIFAR-10 | ResNet-8 / Slim ResNet-18 | **Local** | Both systems |
| Split CIFAR-100 | Slim ResNet-18 | **Local** | Both systems, FP16 |
| Split Mini-ImageNet | Slim ResNet-18, ViT-Small\* | **Local RTX 4050 first** | Resume-enabled local runs; cloud fallback optional |
| Sequential Tiny-ImageNet | ViT-Small\* / Slim ResNet-18 | **Local feasibility first** | Selected top methods only |

\*ViT-Small uses gradient checkpointing + FP16 + gradient accumulation to stay within the local RTX 4050 memory budget where feasible, with cloud fallback still possible if needed.

---

## Proposed Changes — Full File Tree

```
d:\Catastrophic Forgetting\Project\
├── src/
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── permuted_mnist.py       # Domain-IL, 10 tasks
│   │   ├── split_cifar10.py        # Class-IL, 5 tasks × 2 classes
│   │   ├── split_cifar100.py       # Class-IL, 20 tasks × 5 classes
│   │   ├── split_mini_imagenet.py  # Class-IL, 20 tasks × 5 classes
│   │   ├── seq_tiny_imagenet.py    # Class-IL, 10 tasks × 20 classes
│   │   └── base_dataset.py         # Abstract base class for all dataset wrappers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet.py               # ResNet-8, Slim ResNet-18 (custom lightweight)
│   │   ├── vit_small.py            # ViT-Small wrapper (timm), gradient checkpointing
│   │   └── classifier_head.py      # Expandable multi-head classifier for Class-IL
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── base_method.py          # Abstract CL method interface
│   │   ├── baselines/
│   │   │   ├── fine_tune.py        # Naive SGD, no protection (lower bound)
│   │   │   ├── joint_train.py      # Full data replay (upper bound)
│   │   │   ├── ewc.py              # Elastic Weight Consolidation
│   │   │   ├── agem.py             # Averaged GEM
│   │   │   └── lwf.py              # Learning without Forgetting
│   │   └── hybrid/
│   │       ├── der.py              # Dark Experience Replay (Replay + Distill)
│   │       ├── xder.py             # Extended DER (Replay + Distill + Revision)
│   │       ├── icarl.py            # iCaRL (Replay + Distill + Nearest-Mean)
│   │       ├── er_ewc.py           # ER-Reservoir + EWC (Replay + Regularization)
│   │       ├── progress_compress.py# Progress & Compress (Arch + EWC + Distill)
│   │       ├── agem_distill.py     # Novel: A-GEM + Knowledge Distillation
│   │       └── si_der.py           # Novel: Synaptic Intelligence + DER
│   ├── trainers/
│   │   ├── __init__.py
│   │   └── cl_trainer.py           # Core training loop: tasks, eval, buffer mgmt
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── continual_metrics.py    # AA, F, BWT, FWT, memory, compute time
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py                # All chart generators (matplotlib/seaborn)
│   └── utils/
│       ├── __init__.py
│       ├── seed.py                 # Deterministic seeding (torch, numpy, random)
│       ├── logger.py               # CSV + console logging per run
│       └── checkpoint.py           # Save/load model + buffer state
├── experiments/
│   ├── configs/
│   │   ├── base_config.yaml        # Default hyperparams (overridden per run)
│   │   ├── permuted_mnist_*.yaml   # One config per method × dataset combo
│   │   └── split_cifar100_*.yaml
│   └── run_experiment.py           # CLI entrypoint: --config, --seed, --device
├── notebooks/
│   ├── local_mini_imagenet.py      # Local: Mini-ImageNet multi-method runner
│   └── local_tiny_imagenet.py      # Local: Tiny-ImageNet top-method runner
├── results/
│   ├── raw/                        # Per-run CSVs (method, dataset, seed, metrics)
│   └── figures/                    # PNG plots per method × dataset
├── requirements.txt
└── app/                            # Planned for Phase 6 (not created yet)
```

---

## Component Details

### 1. Dataset Loaders (`src/datasets/`)

All datasets implement a common interface:
```python
class BaseCLDataset:
    def get_task_loaders(task_id: int) -> (train_loader, test_loader)
    def get_all_test_loaders() -> list[DataLoader]   # for full matrix eval
    @property n_tasks: int
    @property n_classes_per_task: int
```

**Split strategy for Class-IL**: Classes sorted by label, split sequentially into equal-sized task groups. Task identity **not** provided at test time (hard Class-IL).

**Permuted MNIST**: Fixed random permutation per task applied to all 784 pixels. 10 permutations = 10 tasks.

---

### 2. Models (`src/models/`)

**Slim ResNet-18** (local workhorse):
- Standard ResNet-18 architecture but with reduced channel widths (width multiplier 0.5)
- Outputs a feature vector; classifier head is separate and expandable
- ~2M parameters. FP16 training via `torch.cuda.amp`

**ViT-Small** (optional higher-memory run):
- Loaded from `timm` library (`vit_small_patch16_224`)
- Gradient checkpointing enabled (`model.set_grad_checkpointing(True)`)
- FP16 via `torch.cuda.amp.autocast`
- Gradient checkpointing is enabled; batch size should stay conservative on local hardware

**Expandable Classifier Head** (Class-IL):
- Multi-head design: one output head per task group
- At test time in Class-IL, all heads active, argmax over all outputs

---

### 3. Method Interface (`src/methods/base_method.py`)

All methods implement:
```python
class BaseCLMethod:
    def before_task(task_id, train_loader)   # e.g. compute Fisher for EWC
    def observe(x, y, task_id) -> loss        # single forward+backward step
    def after_task(task_id, train_loader)    # e.g. update buffer, distill
    def get_buffer() -> ReplayBuffer | None
```

This uniform interface means `cl_trainer.py` is method-agnostic and calls the same hooks for every method.

---

### 4. Replay Buffer (`src/methods/base_method.py`)

```python
class ReplayBuffer:
    strategy: "reservoir" | "herding" (iCaRL uses herding)
    capacity: int                        # total samples across all tasks
    store(x, y, logits=None)             # DER also stores logits
    sample(batch_size) -> (x, y, logits?)
```

- **Reservoir sampling**: uniform random replacement, O(1) per update
- **Herding** (iCaRL): selects exemplars closest to class mean in feature space

---

### 5. Training Harness (`src/trainers/cl_trainer.py`)

```
for task_id in 0..N-1:
    method.before_task(task_id, train_loader)
    for epoch in 0..E-1:
        for (x, y) in train_loader:
            loss = method.observe(x, y, task_id)
    method.after_task(task_id, train_loader)
    # Evaluate on ALL tasks seen so far → fills accuracy matrix A[t][i]
    evaluate_all_tasks(model, all_test_loaders[:task_id+1])
```

The **accuracy matrix** `A` where `A[t][i]` = accuracy on task `i` after training on task `t` is the raw data from which all 4 metrics are computed.

---

### 6. Metrics (`src/metrics/continual_metrics.py`)

All computed from the accuracy matrix `A`:

| Metric | Formula | Meaning |
|---|---|---|
| **Avg Accuracy** | `(1/T) Σ A[T-1][i]` | Final performance across all tasks |
| **Forgetting** | `(1/T-1) Σ (max_j A[j][i] − A[T-1][i])` | Average peak-to-final drop |
| **BWT** | `(1/T-1) Σ (A[T-1][i] − A[i][i])` | Did new tasks help or hurt old ones? |
| **FWT** | `(1/T-1) Σ (A[i-1][i] − b_i)` | Did prior knowledge help new task start? |

---

### 7. Visualization (`src/visualization/plots.py`)

All plots saved as PNG to `results/figures/`. Generated automatically after each full run.

| Plot | Type | X-axis | Y-axis |
|---|---|---|---|
| Accuracy heatmap | Seaborn heatmap | Task trained on | Task evaluated on |
| Forgetting bar | Grouped bar | Method | Forgetting score |
| BWT & FWT | Grouped bar | Method | Transfer score |
| Training curves | Line plot | Task index | Accuracy on each past task |
| Pareto frontier | Scatter | Buffer memory (KB) | Final avg accuracy |
| Compute cost | Grouped bar | Method | Time per task (s) |

---

### 8. Experiment Runner (`experiments/run_experiment.py`)

```bash
python run_experiment.py \
  --config configs/split_cifar100_der.yaml \
  --seed 42 \
  --device cuda
```

Config YAML example:
```yaml
dataset: split_cifar100
n_tasks: 20
model: slim_resnet18
method: der
buffer_size: 500
lr: 0.03
n_epochs: 1        # CL typically 1 epoch per task
batch_size: 32
grad_accum_steps: 1
fp16: true
lambda_distill: 0.5
seeds: [42, 123, 456, 789, 1024]
```

5 seeds per config. Results averaged ± std.

---

### 9. Local Phase 4 Runners

**`Project/notebooks/local_mini_imagenet.py`**:
- runs Mini-ImageNet locally
- supports multiple methods and seeds
- enables task-boundary resume
- can delete checkpoints automatically after successful completion

**`Project/notebooks/local_tiny_imagenet.py`**:
- runs Tiny-ImageNet locally with a top-method subset
- supports resumable task-boundary checkpoints
- keeps local-first execution while preserving a fallback path for heavier runs

**Checkpoint strategy**: Save `{run_name}_task{N}.pt` after each completed task. Resume continues from the latest completed task checkpoint. Completed successful runs may delete checkpoints automatically to save space.

---

### 10. Statistical Analysis (`notebooks/analysis.ipynb`)

- Paired t-tests between each hybrid method and baselines (per metric per dataset)
- Bonferroni correction for multiple comparisons
- Cohen's d effect sizes
- Friedman test for overall ranking across methods
- Pareto frontier identification (accuracy vs. memory, accuracy vs. compute)

---

### 11. Decision Tree (`Phase 5`)

Built from empirical results:

**Input features:**
- Available memory budget (MB)
- Compute budget (GPU hours, as low/med/high)
- Expected task similarity (high/low)
- Acceptable forgetting threshold (%)
- Dataset complexity (small/medium/large)

**Output:** Recommended hybrid CL method

Implementation: `sklearn.tree.DecisionTreeClassifier` trained on (experiment configs → best method) pairs from results. Exported as both a visual flowchart (graphviz) and as a Python function for the dashboard.

---

### 12. Streamlit Dashboard (`Project/app/main.py`, planned) — Phase 6

Sections:
1. **Inputs panel**: Sliders + dropdowns for hardware constraints + task params
2. **Recommendation panel**: Decision tree output + explanation
3. **Method comparison**: Embedded Pareto chart, metric tables
4. **Accuracy heatmap viewer**: Select any method + dataset → view heatmap
5. **About**: Project summary, links to paper/code

---

## Methods: Implementation Details

### Baselines

| Method | Key mechanism | Implementation notes |
|---|---|---|
| Fine-tune | Standard SGD, no protection | Simplest baseline |
| Joint | Concatenate all task data | Only feasible for small datasets |
| EWC | Fisher-weighted L2 penalty on weights | Compute Fisher after each task end via diagonal approximation |
| A-GEM | Project gradient onto old-task constraint | Store fixed episodic memory (200 samples); project if constraint violated |
| LwF | Distillation from snapshot of old model | Save model snapshot before each new task |

### Hybrid Methods

| Method | Mechanism A | Mechanism B | Mechanism C | Buffer |
|---|---|---|---|---|
| **DER** | Replay | Distillation (logit MSE) | — | Reservoir, stores (x, y, logits) |
| **X-DER** | Replay | Distillation (logit MSE) | Buffer revision (update stored logits) | Reservoir |
| **iCaRL** | Replay | Distillation (KD loss) | Nearest-Mean classifier | Herding exemplars |
| **ER+EWC** | Replay | EWC regularization | — | Reservoir |
| **P&C** | Online distillation | EWC on knowledge base | Lateral connections | None |
| **A-GEM+Distill** | Gradient projection | Distillation from old snapshot | — | Episodic memory (fixed) |
| **SI+DER** | Online weight importance (SI) | Replay + logit distillation | — | Reservoir |

---

## Hyperparameter Grid

| Hyperparameter | Values | Notes |
|---|---|---|
| Buffer size | 200, 500, 1000 | For replay methods |
| EWC λ | 1, 10, 100, 1000 | For EWC-based methods |
| Distill α/λ | 0.1, 0.5, 1.0 | For distillation term weight |
| LR | 0.01, 0.03, 0.1 | SGD with momentum 0.9 |
| Epochs/task | 1, 5 | 1 = standard CL setting |
| Batch size | 32 (local ResNet), small conservative batches for local ViT | |
| Seeds | 5 per config (42, 123, 456, 789, 1024) | |

---

## Verification Plan

### Automated Sanity Checks

1. **Metric correctness test** — run `pytest src/metrics/` with a handcrafted accuracy matrix where forgetting and BWT are known analytically. Pass if computed values match expected within 1e-4.

2. **Fine-tune regression test** — after training on 3 tasks of Split CIFAR-10, verify that fine-tune accuracy on Task 1 drops by ≥30 percentage points from peak. This confirms catastrophic forgetting is actually occurring.

3. **Joint training bound test** — joint training final accuracy must be ≥ fine-tune final accuracy on all datasets. If not, implementation bug.

4. **Buffer capacity test** — after N tasks with buffer size K, verify `len(buffer) == K` for reservoir sampling.

5. **Reproducibility test** — two runs with identical seed produce identical accuracy matrices (bitwise identical on CPU).

### Manual Validation

6. **Published number check** — compare DER results on Split CIFAR-10 against Table 1 of Buzzega et al. (2020). Avg Accuracy should be within ±3% of paper's reported value (paper: ~44.7% on S-CIFAR-10 with 200 buffer).

7. **Visual sanity check** — inspect the accuracy heatmap for fine-tune: it must show a clear "diagonal" pattern (high accuracy only on the most recent task, near-zero on all earlier tasks).

8. **Pareto frontier check** — confirm that Joint Training sits at the top-right of the Pareto chart (high accuracy) and fine-tune at bottom-left. Methods should fall between.

### Commands to Run Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest src/ -v

# Run a single quick experiment (MNIST, fine-tune, 1 seed, CPU)
python experiments/run_experiment.py \
  --config experiments/configs/permuted_mnist_finetune.yaml \
  --seed 42 --device cpu

# View output plot
# results/figures/permuted_mnist_finetune_seed42_heatmap.png
```

---

## Dependencies (`requirements.txt`)

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0           # ViT-Small
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0   # Decision tree
pyyaml>=6.0
tqdm>=4.65.0
pytest>=7.4.0
streamlit>=1.29.0     # Phase 6 only
```

---

## Key Design Decisions

> [!IMPORTANT]
> All methods implement the same `BaseCLMethod` interface. The trainer is method-agnostic. Adding a new hybrid method only requires creating a new file in `methods/hybrid/` — no changes to the trainer.

> [!IMPORTANT]
> Class-IL (no task ID at test time) is the default scenario for all image datasets. This is the hardest and most scientifically rigorous setting. The expandable multi-head classifier with argmax over all heads handles this correctly.

> [!NOTE]
> ViT-Small experiments are optional and should use conservative local settings first; cloud fallback remains available if local memory becomes the bottleneck.

> [!NOTE]
> Novel hybrid methods (A-GEM+Distill, SI+DER) are original contributions of this project. Their results will be discussed comparatively alongside known methods.

> [!NOTE]
> The frontend (Streamlit dashboard) is strictly Phase 6 and will not be started until Phases 1–5 are complete and results are finalized.
