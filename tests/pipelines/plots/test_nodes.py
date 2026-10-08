import numpy as np
import pandas as pd
from kedro.io import Version
from kedro_datasets.matplotlib import MatplotlibDataset
from matplotlib.colors import to_rgba

from fatigued_face_space.pipelines.au_map import au_map_int
from fatigued_face_space.pipelines.plots.nodes import (
    create_peak_plots,
    create_trend_noise_plots,
)


def _sample_au_data() -> pd.DataFrame:
    signal = np.zeros(40)
    signal[10] = 1.0
    signal[25] = 1.5
    columns = {" timestamp": np.arange(40) / 10}
    columns.update({f" AU{au:02}_r": signal.copy() for au in au_map_int})
    columns[" AU45_r"] = np.zeros(40)
    return pd.DataFrame(columns)


def test_trend_noise_plots_use_metadata_movie_and_expected_labels(tmp_path):
    df = _sample_au_data()
    partitions = {"selected": lambda: df, "other": lambda: df}

    plots = create_trend_noise_plots(partitions, ["selected"], frac=0.2)

    assert len(plots) == len(au_map_int)
    assert all(key.startswith("selected/") for key in plots)
    figure = plots["selected/AU01.png"]
    axes = figure.axes[0]
    assert "selected | AU01_r | Trend and noise" == axes.get_title()
    assert axes.get_xlabel() == "Time (s)"
    assert axes.get_ylabel() == "AU intensity"
    assert [line.get_color() for line in axes.lines[:3]] == [
        "#808080",
        "#1f77b4",
        "#2ca02c",
    ]
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "Raw data",
        "Trend (LOWESS)",
        "Noise (residual)",
    ]

    dataset = MatplotlibDataset(
        filepath=(tmp_path / "trend_noise").as_posix(),
        version=Version(load=None, save="2026-10-08T00.00.00.000Z"),
        save_args={"format": "png"},
    )
    dataset.save({"selected/AU01.png": figure})
    assert (
        tmp_path
        / "trend_noise"
        / "2026-10-08T00.00.00.000Z"
        / "trend_noise"
        / "selected"
        / "AU01.png"
    ).is_file()


def test_peak_plots_mark_detected_samples():
    df = _sample_au_data()

    plots = create_peak_plots(
        {"selected": lambda: df},
        ["selected"],
        height=0.1,
        distance=5,
        prominence=0.1,
    )

    assert len(plots) == len(au_map_int)
    axes = plots["selected/AU01.png"].axes[0]
    assert axes.get_title() == "selected | AU01_r | Detected peaks"
    assert axes.lines[0].get_color() == "#1f77b4"
    np.testing.assert_allclose(
        axes.collections[0].get_facecolors()[0], to_rgba("#d62728")
    )
    np.testing.assert_allclose(
        axes.collections[0].get_offsets(), [[1.0, 1.0], [2.5, 1.5]]
    )
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "Raw data",
        "Detected peaks (2)",
    ]

    no_peak_axes = plots["selected/AU45.png"].axes[0]
    assert len(no_peak_axes.collections[0].get_offsets()) == 0
    assert any(text.get_text() == "No peaks detected" for text in no_peak_axes.texts)
