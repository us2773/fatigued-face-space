"""Kedro pipeline for AU time-series visualisation."""

from kedro.pipeline import Pipeline, node

from fatigued_face_space.nodes.metadata.v1.nodes import get_movie_list
from fatigued_face_space.nodes.plots.v1.nodes import (
    create_peak_plots,
    create_trend_residual_plots,
)


# AUの時系列グラフを作成するPipelineを構築
def create_au_plots() -> Pipeline:
    """Plot every AU for the movies selected by the metadata file."""
    return Pipeline(
        [
            node(
                func=get_movie_list,
                inputs="movie_metadata",
                outputs="plot_movie_list",
                name="plot_get_movie_list",
            ),
            node(
                func=create_trend_residual_plots,
                inputs=["cleansed_openface_timeseries", "trend_residual_timeseries", "plot_movie_list"],
                outputs="trend_residual_plots",
                name="create_trend_residual_plots",
            ),
            node(
                func=create_peak_plots,
                inputs=[
                    "cleansed_openface_timeseries",
                    "trend_residual_timeseries",
                    "peak_timeseries",
                    "plot_movie_list",
                ],
                outputs="peak_plots",
                name="create_peak_plots",
            ),
        ]
    )
