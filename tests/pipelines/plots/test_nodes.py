import numpy as np
import pandas as pd
from kedro.io import Version
from kedro_datasets.matplotlib import MatplotlibDataset
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle

from fatigued_face_space.common.au_map import au_map_int
from fatigued_face_space.nodes.plots.v1.nodes import (
    create_peak_plots,
    create_trend_residual_plots,
)


def _sample_au_data() -> pd.DataFrame:
    signal = np.zeros(40)
    signal[10] = 1.0
    signal[25] = 1.5
    columns = {" timestamp": np.arange(40) / 10}
    columns.update({f" AU{au:02}_r": signal.copy() for au in au_map_int})
    columns[" AU45_r"] = np.zeros(40)
    return pd.DataFrame(columns)


def test_trend_residual_plots_use_metadata_movie_and_expected_labels(tmp_path):
    df = _sample_au_data()
    partitions = {"selected": lambda: df, "other": lambda: df}
    trend = np.linspace(0.0, 0.3, len(df))
    trend_residual_columns = {"timestamp": df[" timestamp"].to_numpy()}
    for au in au_map_int:
        trend_residual_columns[f"AU{au:02}_trend"] = trend
        trend_residual_columns[f"AU{au:02}_residual"] = (
            df[f" AU{au:02}_r"].to_numpy() - trend
        )
    trend_residual_df = pd.DataFrame(trend_residual_columns)
    trend_residual_partitions = {
        "selected": lambda: trend_residual_df,
        "other": lambda: trend_residual_df,
    }

    plots = create_trend_residual_plots(
        partitions, trend_residual_partitions, ["selected"]
    )

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
    np.testing.assert_array_equal(axes.lines[0].get_ydata(), df[" AU01_r"])
    np.testing.assert_array_equal(axes.lines[0].get_xdata(), trend_residual_df["timestamp"])
    np.testing.assert_array_equal(axes.lines[1].get_ydata(), trend)
    np.testing.assert_array_equal(
        axes.lines[2].get_ydata(), trend_residual_df["AU01_residual"]
    )
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
    residual_signal = np.zeros(40)
    residual_signal[5] = 0.6
    residual_signal[30] = 1.2
    residual_columns = {"timestamp": df[" timestamp"].to_numpy()}
    peak_columns = {}
    no_peak_au = 45
    for au in au_map_int:
        residual_columns[f"AU{au:02}_residual"] = (
            residual_signal.copy() if au != no_peak_au else np.zeros(40)
        )
        peak_columns[f"AU{au:02}_peakpoints"] = pd.Series(
            [0.5, 3.0] if au != no_peak_au else [], dtype="float64"
        )
    residual_df = pd.DataFrame(residual_columns)
    peak_df = pd.DataFrame(peak_columns)

    plots = create_peak_plots(
        {"selected": lambda: df},
        {"selected": lambda: residual_df},
        {"selected": lambda: peak_df},
        ["selected"],
    )

    assert len(plots) == len(au_map_int)
    axes = plots["selected/AU01.png"].axes[0]
    assert axes.get_title() == "selected | AU01_r | Detected peaks"
    assert axes.lines[0].get_color() == "#808080"
    raw_alpha = 0.35
    assert axes.lines[0].get_alpha() == raw_alpha
    assert axes.lines[1].get_color() == "#1f77b4"
    np.testing.assert_array_equal(axes.lines[0].get_ydata(), df[" AU01_r"])
    np.testing.assert_array_equal(
        axes.lines[1].get_ydata(), residual_df["AU01_residual"]
    )
    np.testing.assert_allclose(
        axes.collections[0].get_facecolors()[0], to_rgba("#d62728")
    )
    np.testing.assert_allclose(
        axes.collections[0].get_offsets(), [[0.5, 0.6], [3.0, 1.2]]
    )
    circle = MarkerStyle("o")
    np.testing.assert_allclose(
        axes.collections[0].get_paths()[0].vertices,
        circle.get_path().transformed(circle.get_transform()).vertices,
    )
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "Raw data",
        "Residual",
        "Detected peaks (2)",
    ]

    no_peak_axes = plots["selected/AU45.png"].axes[0]
    assert len(no_peak_axes.collections[0].get_offsets()) == 0
    assert any(text.get_text() == "No peaks detected" for text in no_peak_axes.texts)
