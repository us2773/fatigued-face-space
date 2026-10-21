"""Create diagnostic plots from OpenFace AU time series."""

from collections.abc import Callable, Mapping

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..au_map import au_map_int


def _plot_output_key(movie_name: str, au_number: int) -> str:
    """Keep each movie in its own directory below the catalog output path."""
    if (
        not movie_name
        or movie_name in {".", ".."}
        or any(separator in movie_name for separator in ("/", "\\", ":"))
    ):
        raise ValueError(f"Invalid movie name for plot output: {movie_name!r}")
    return f"{movie_name}/AU{au_number:02}.png"


def _create_au_figure(
    movie_name: str, au_number: int, plot_kind: str
) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(10, 4.5), layout="constrained")
    axes = figure.subplots()
    axes.set_title(f"{movie_name} | AU{au_number:02}_r | {plot_kind}")
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("AU intensity")
    axes.grid(alpha=0.25)
    return figure, axes


def create_trend_residual_plots(
    partitions: Mapping[str, Callable[[], pd.DataFrame]],
    trend_residual_partitions: Mapping[str, Callable[[], pd.DataFrame]],
    movie_list: list[str],
) -> dict[str, Figure]:
    """Plot the raw AU signal, stored LOWESS trend, and residual for each movie."""
    trend_residual_figures: dict[str, Figure] = {}

    for movie_name in movie_list:
        df = partitions[movie_name]()
        trend_residual_df = trend_residual_partitions[movie_name]()
        times = trend_residual_df["timestamp"].to_numpy()

        for au_number in au_map_int:
            signal = df[f" AU{au_number:02}_r"].to_numpy()
            trend = trend_residual_df[f"AU{au_number:02}_trend"].to_numpy()
            residual = trend_residual_df[f"AU{au_number:02}_residual"].to_numpy()

            figure, axes = _create_au_figure(movie_name, au_number, "Trend and noise")
            axes.plot(times, signal, color="#808080", linewidth=1, label="Raw data")
            axes.plot(
                times, trend, color="#1f77b4", linewidth=2, label="Trend (LOWESS)"
            )
            axes.plot(
                times,
                residual,
                color="#2ca02c",
                linewidth=1.2,
                label="Noise (residual)",
            )
            axes.axhline(0, color="#808080", linewidth=0.7, alpha=0.5)
            axes.legend()
            trend_residual_figures[_plot_output_key(movie_name, au_number)] = figure

    return trend_residual_figures


def create_peak_plots(
    partitions: Mapping[str, Callable[[], pd.DataFrame]],
    trend_residual_partitions: Mapping[str, Callable[[], pd.DataFrame]],
    peak_partitions: Mapping[str, Callable[[], pd.DataFrame]],
    movie_list: list[str],
) -> dict[str, Figure]:
    """Plot raw AU signals, residuals, and stored residual peaks."""
    peak_figures: dict[str, Figure] = {}

    for movie_name in movie_list:
        df = partitions[movie_name]()
        trend_residual_df = trend_residual_partitions[movie_name]()
        peak_df = peak_partitions[movie_name]()
        times = trend_residual_df["timestamp"].to_numpy()
        time_index = pd.Index(times)

        for au_number in au_map_int:
            signal = df[f" AU{au_number:02}_r"].to_numpy()
            residual = trend_residual_df[f"AU{au_number:02}_residual"].to_numpy()
            peak_times = peak_df[f"AU{au_number:02}_peakpoints"].dropna().to_numpy()
            peak_indices = time_index.get_indexer(peak_times)
            if (peak_indices < 0).any():
                raise ValueError(
                    f"Stored peak timestamp not found for {movie_name} AU{au_number:02}"
                )

            figure, axes = _create_au_figure(movie_name, au_number, "Detected peaks")
            axes.plot(
                times, signal, color="#808080", alpha=0.35, linewidth=1,
                label="Raw data",
            )
            axes.plot(
                times, residual, color="#1f77b4", linewidth=1.5,
                label="Residual",
            )
            axes.scatter(
                peak_times,
                residual[peak_indices],
                color="#d62728",
                marker="o",
                edgecolors="white",
                linewidths=0.8,
                s=55,
                zorder=3,
                label=f"Detected peaks ({len(peak_times)})",
            )
            if len(peak_times) == 0:
                axes.text(
                    0.98,
                    0.95,
                    "No peaks detected",
                    transform=axes.transAxes,
                    horizontalalignment="right",
                    verticalalignment="top",
                )
            axes.legend()
            peak_figures[_plot_output_key(movie_name, au_number)] = figure

    return peak_figures
