from pathlib import Path

import pandas as pd
import pytest
import yaml
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from fatigued_face_space.nodes.feature_select.v1.nodes import (
    compare_fatigue_group_medians,
    get_fatigue_features_tables,
    judge_direction,
    summarize_direction_matches,
)
from fatigued_face_space.pipelines.feature_select.v1.pipelines import (
    create_feature_select,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["up"] * 3, "up-match"),
        (["down"] * 3, "down-match"),
        (["unchanged"] * 3, "unchanged-match"),
        (["up", "down", "unchanged"], "unchanged-mixed"),
        (["up", "up", "unchanged"], "unchanged-mixed"),
        (["up", "down", "up"], "nomatch"),
        (["missing", "up", "up"], "missing"),
        ([None, "up", "up"], "missing"),
        ([], "nomatch"),
    ],
)
def test_direction_summary(values, expected):
    table = pd.DataFrame({"person": range(len(values)), "feature": values})
    assert summarize_direction_matches(table)["feature"] == expected


def test_comparison_aligns_people_and_features():
    base = pd.DataFrame({"person": [1, 2, 3], "a": [0, 2, 1], "b": [1, 1, 1]})
    fatigue = pd.DataFrame({"person": [3, 1], "b": [2, 1], "a": [0, 0]})
    actual = compare_fatigue_group_medians(base, fatigue).set_index("person")
    assert actual["a"].tolist() == ["unchanged", "missing", "down"]
    assert actual["b"].tolist() == ["unchanged", "missing", "up"]


def test_absent_group_retains_subject():
    table = pd.DataFrame(
        {
            "person": [1, 1, 2],
            "class": [1, None, None],
            "is_baseface": [0, 1, 1],
            "AU01_trend_mean": [2, 1, 0],
        }
    )
    languid, _, base = get_fatigue_features_tables(table, [1, 2, 3], [1, 2], [3, 4])
    assert languid["person"].tolist() == [1, 2, 3]
    compared = compare_fatigue_group_medians(base, languid)
    assert compared["AU01_trend_mean"].tolist() == ["up", "missing", "missing"]


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("threshold", [0.0, 1.5])
def test_pipeline_with_synthetic_data(tmp_path, persist, threshold):
    """Exercise every node using artificial observations only."""
    parameters = {
        "min_median_difference": threshold,
        "persons": [1, 2],
        "languid_classes": [1, 2],
        "tense_classes": [3, 4],
    }
    table = pd.DataFrame(
        {
            "person": [1, 1, 1, 2, 2, 2],
            "class": [None, 1, 3, None, 2, 4],
            "is_baseface": [1, 0, 0, 1, 0, 0],
            "AU01_trend_mean": [1, 2, 0, 1, 2, 0],
            "AU01_trend_var": [0, 0, 1, 0, 1, -1],
            "AU01_peakfreq": [0, 0, 0, 0, 0, 0],
            "new_feature": [0, 1, -1, 0, 1, -1],
            "Name": ["sample"] * 6,
            "check_date": ["2000-01-01"] * 6,
            "fatigue_level": [0, 1, 2, 0, 1, 2],
            "Unnamed: 0": list(range(6)),
        }
    )
    catalog = DataCatalog(
        {
            "features_table": MemoryDataset(table),
            **{
                f"params:feature_select.{key}": MemoryDataset(value)
                for key, value in parameters.items()
            },
        }
    )
    pipeline = create_feature_select()
    if persist:
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
        saved_catalog = DataCatalog.from_config(output_config)
        for name in output_config:
            catalog[name] = saved_catalog[name]
    outputs = SequentialRunner().run(pipeline, catalog)
    actual = (
        catalog.load("fatigue_direction_matches")
        if persist
        else outputs["fatigue_direction_matches"].load()
    )
    expected = pd.DataFrame(
        {
            "AU01_trend_mean": ["up-match", "down-match"],
            "AU01_trend_var": ["unchanged-mixed", "nomatch"],
            "AU01_peakfreq": ["unchanged-match", "unchanged-match"],
            "new_feature": ["up-match", "down-match"],
        },
        index=["languid", "tense"],
        dtype="object",
    )
    if threshold > 0:
        expected.loc[:, :] = "unchanged-match"
    pd.testing.assert_frame_equal(actual, expected, check_dtype=not persist)
    if persist:
        for name, entry in output_config.items():
            dataset_path = Path(entry["filepath"])
            files = list(dataset_path.glob(f"*/{dataset_path.name}"))
            assert len(files) == 1
            if name != "fatigue_direction_matches":
                assert dataset_path.suffix == ".parquet"
                pd.testing.assert_frame_equal(
                    pd.read_parquet(files[0]), catalog.load(name)
                )


@pytest.mark.parametrize("threshold", [0.0, 0.5])
def test_threshold_boundaries_and_missing(threshold):
    base = pd.DataFrame({"person": range(8), "feature": [0.0] * 7 + [None]})
    fatigue = pd.DataFrame(
        {
            "person": range(8),
            "feature": [0.0, 0.25, -0.25, 0.5, -0.5, 1.0, None, 1.0],
        }
    )
    expected = (
        ["unchanged", "up", "down", "up", "down", "up", "missing", "missing"]
        if threshold == 0
        else [
            "unchanged",
            "unchanged",
            "unchanged",
            "up",
            "down",
            "up",
            "missing",
            "missing",
        ]
    )
    actual = compare_fatigue_group_medians(base, fatigue, threshold)
    assert actual["feature"].tolist() == expected
    assert [
        judge_direction(b, f, threshold)
        for b, f in zip(base["feature"], fatigue["feature"], strict=True)
    ] == expected


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf"), "0.1", True])
def test_invalid_threshold_is_rejected(threshold):
    table = pd.DataFrame({"person": [1], "feature": [0.0]})
    with pytest.raises(ValueError, match="finite nonnegative"):
        compare_fatigue_group_medians(table, table, threshold)
    with pytest.raises(ValueError, match="finite nonnegative"):
        judge_direction(0.0, 0.0, threshold)
