"""Create diagnostic plots from cleansed OpenFace AU time series."""

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..au_map import au_map_int
from ..nodes import find_AU_peaks, separate_AU_trend_noise


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


def create_trend_noise_plots(
    partitions: Mapping[str, Callable[[], pd.DataFrame]],
    movie_list: list[str],
    frac: float,
) -> dict[str, Figure]:
    """Plot the raw AU signal, its LOWESS trend, and the residual for each movie."""
    trend_noise_figures: dict[str, Figure] = {}

    for movie_name in movie_list:
        df = partitions[movie_name]()
        times = df[" timestamp"].to_numpy()

        for au_index, au_number in enumerate(au_map_int):
            signal = df[f" AU{au_number:02}_r"].to_numpy()
            trend, residual = separate_AU_trend_noise(df, au_index, frac)

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
            trend_noise_figures[_plot_output_key(movie_name, au_number)] = figure

    return trend_noise_figures


def create_peak_plots(
    partitions: Mapping[str, Callable[[], pd.DataFrame]],
    movie_list: list[str],
    height: float,
    distance: float,
    prominence: float,
) -> dict[str, Figure]:
    """Plot the raw AU signal and peaks found with the feature pipeline settings."""
    peak_figures: dict[str, Figure] = {}

    for movie_name in movie_list:
        df = partitions[movie_name]()

        for au_index, au_number in enumerate(au_map_int):
            signal = df[f" AU{au_number:02}_r"].to_numpy()
            peaks, times = find_AU_peaks(df, au_index, height, distance, prominence)

            figure, axes = _create_au_figure(movie_name, au_number, "Detected peaks")
            axes.plot(times, signal, color="#1f77b4", linewidth=1.2, label="Raw data")
            axes.scatter(
                np.asarray(times)[peaks],
                signal[peaks],
                color="#d62728",
                edgecolors="white",
                linewidths=0.8,
                s=55,
                zorder=3,
                label=f"Detected peaks ({len(peaks)})",
            )
            if len(peaks) == 0:
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
