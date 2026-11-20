from pathlib import Path

import pandas as pd
import pytest
import yaml
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from fatigued_face_space.nodes.feature_select.v2.nodes import filtered_features_table
from fatigued_face_space.pipelines.feature_select.v2.pipelines import (
    create_feature_select_v2,
)


def test_selection_retains_baseline_and_does_not_mutate_input():
    table = pd.DataFrame(
        {
            "fatigue_level": [1, 2, None, None],
            "class": [1, 2, None, 2],
            "is_baseface": [0, 0, 1, 0],
            "feature": [0, 1, 2, 3],
        },
        index=[4, 5, 6, 7],
    )
    original = table.copy(deep=True)
    filtered = filtered_features_table(table, [2], [2])
    assert filtered.index.tolist() == [5, 6]
    filtered.loc[5, "feature"] = 99
    pd.testing.assert_frame_equal(table, original)


def test_empty_selection_retains_only_baseline():
    table = pd.DataFrame(
        {"fatigue_level": [1, None], "class": [1, None], "is_baseface": [0, 1]}
    )
    pd.testing.assert_frame_equal(
        filtered_features_table(table, [], [1, 2, 3, 4]), table.iloc[[1]]
    )


def test_missing_filter_column_is_an_error():
    with pytest.raises(ValueError, match="fatigue_level"):
        filtered_features_table(pd.DataFrame({"feature": [1]}), [1], [1])


@pytest.mark.parametrize("selected", [[1, 2, 3], [2, 3], []])
@pytest.mark.parametrize("classes", [[1, 2, 3, 4], [2]])
def test_v2_pipeline_filters_before_medians_and_persists(tmp_path, selected, classes):
    table = pd.DataFrame(
        {
            "person": [1] * 4 + [2] * 4,
            "class": [None, 1, 2, 3] * 2,
            "is_baseface": [1, 0, 0, 0] * 2,
            "fatigue_level": [None, 1, 2, 3] * 2,
            "feature": [0.0, -100.0, 2.0, -2.0] * 2,
        }
    )
    parameters = {
        "persons": [1, 2, 3],
        "languid_classes": [1, 2],
        "tense_classes": [3, 4],
        "min_median_difference": 0.5,
        "selected_fatigue_levels": selected,
        "selected_classes": classes,
    }
    pipeline = create_feature_select_v2()
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(
        (root / "conf/base/catalog.yml").read_text(encoding="utf-8")
    )
    output_config = {}
    for name in pipeline.all_outputs():
        entry = config[name].copy()
        entry["filepath"] = (tmp_path / entry["filepath"]).as_posix()
        if name == "fatigue_direction_matches":
            entry["load_args"] = {"index_col": 0}
        output_config[name] = entry
    catalog = DataCatalog.from_config(output_config)
    catalog["features_table"] = MemoryDataset(table)
    for key, value in parameters.items():
        catalog[f"params:feature_select.{key}"] = MemoryDataset(value)
    SequentialRunner().run(pipeline, catalog)
    filtered = catalog.load("filtered_features_table")
    assert len(filtered) == 2 + 2 * len(set(selected) & set(classes))
    up_level = 2
    compared = catalog.load("languid_compared_features_table")
    assert compared["feature"].tolist() == [
        *(
            ["down"] * 2
            if 1 in selected and 1 in classes
            else ["up"] * 2
            if up_level in selected and up_level in classes
            else ["missing"] * 2
        ),
        "missing",
    ]
    assert catalog.load("fatigue_direction_matches")["feature"].tolist() == [
        "missing",
        "missing",
    ]
    for entry in output_config.values():
        path = Path(entry["filepath"])
        assert len(list(path.glob(f"*/{path.name}"))) == 1


@pytest.mark.parametrize(
    ("levels", "classes", "expected"),
    [
        ([2], [2], [0, 2]),
        ([2], [3], [0, 3]),
        ([3], [2], [0, 4]),
        ([2, 3], [2, 3], [0, 2, 3, 4]),
        ([2, 3], [], [0]),
    ],
)
def test_level_and_class_filters_are_combined(levels, classes, expected):
    table = pd.DataFrame(
        {
            "is_baseface": [1, 0, 0, 0, 0, 0, 0],
            "fatigue_level": [None, 1, 2, 2, 3, 2, None],
            "class": [None, 2, 2, 3, 2, None, 2],
        }
    )
    assert filtered_features_table(table, levels, classes).index.tolist() == expected


def test_missing_class_column_is_an_error():
    with pytest.raises(ValueError, match="class"):
        filtered_features_table(
            pd.DataFrame({"fatigue_level": [2], "is_baseface": [0]}), [2], [2]
        )
