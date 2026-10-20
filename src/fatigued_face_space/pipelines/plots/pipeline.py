"""Kedro pipeline for AU time-series visualisation."""

from kedro.pipeline import Pipeline, node

from ..nodes import get_movie_list
from .nodes import create_peak_plots, create_trend_noise_plots


def create_au_plots() -> Pipeline:
    """Plot every AU for the movies selected by the metadata file."""
    return Pipeline(
        [
            node(
                func=get_movie_list,
                inputs="metadata_list",
                outputs="plot_movie_list",
                name="plot_get_movie_list",
            ),
            node(
                func=create_trend_noise_plots,
                inputs=["cleansed_openface", "plot_movie_list", "params:trend.frac"],
                outputs="trend_noise_plots",
                name="create_trend_noise_plots",
            ),
            node(
                func=create_peak_plots,
                inputs=[
                    "cleansed_openface",
                    "plot_movie_list",
                    "params:peak.height",
                    "params:peak.distance",
                    "params:peak.prominence",
                ],
                outputs="peak_plots",
                name="create_peak_plots",
            ),
        ]
    )
