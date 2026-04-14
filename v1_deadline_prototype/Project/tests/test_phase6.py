import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.dashboard_data import comparison_table, datasets_from_summary, missing_artifacts, top_findings


def test_missing_artifacts_empty_for_current_repo() -> None:
    missing = missing_artifacts()
    assert missing == []


def test_datasets_from_summary_prefers_deadline_dataset_order() -> None:
    df = pd.DataFrame(
        {
            "dataset": ["split_cifar10", "split_mini_imagenet", "permuted_mnist"],
        }
    )
    assert datasets_from_summary(df) == [
        "split_mini_imagenet",
        "split_cifar10",
        "permuted_mnist",
    ]


def test_comparison_table_filters_primary_runs() -> None:
    df = pd.DataFrame(
        [
            {
                "dataset": "split_mini_imagenet",
                "method": "icarl",
                "method_variant": "icarl",
                "source_group": "phase4_local_mini",
                "model": "slim_resnet18",
                "seeds": 5,
                "avg_accuracy_mean": 3.7,
                "forgetting_mean": 8.5,
                "backward_transfer_mean": -8.0,
                "forward_transfer_mean": -2.7,
                "runtime_hours_mean": 0.12,
                "buffer_size_mean": 2000,
                "data_quality_note": "",
                "is_primary_run": True,
            },
            {
                "dataset": "split_mini_imagenet",
                "method": "icarl",
                "method_variant": "icarl_variant",
                "source_group": "interaction_ablations",
                "model": "slim_resnet18",
                "seeds": 3,
                "avg_accuracy_mean": 2.8,
                "forgetting_mean": 12.0,
                "backward_transfer_mean": -10.0,
                "forward_transfer_mean": -2.7,
                "runtime_hours_mean": 0.10,
                "buffer_size_mean": 2000,
                "data_quality_note": "variant",
                "is_primary_run": False,
            },
        ]
    )
    table = comparison_table(df, "split_mini_imagenet", primary_only=True)
    assert len(table) == 1
    assert table.iloc[0]["Variant"] == "icarl"


def test_top_findings_surfaces_icarl_message() -> None:
    df = pd.DataFrame(
        [
            {
                "dataset": "split_mini_imagenet",
                "method": "icarl",
                "avg_accuracy_mean": 3.72,
                "forgetting_mean": 8.53,
                "is_primary_run": True,
                "data_quality_note": "",
            },
            {
                "dataset": "split_cifar10",
                "method": "joint_training",
                "avg_accuracy_mean": 64.5,
                "forgetting_mean": 10.1,
                "is_primary_run": True,
                "data_quality_note": "",
            },
        ]
    )
    findings = top_findings(df)
    assert any("iCaRL" in item["body"] or "iCaRL" in item["title"] for item in findings)
