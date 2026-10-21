from kedro.pipeline import Node, Pipeline

from ..nodes import *
from .nodes import *


def create_features_report_v2(**kwargs) -> Pipeline :
    return Pipeline(
        [
            Node(
                func=cleansing_dataframe,
                inputs=["openface_result"],
                outputs="cleansed_openface",
                name="cleansing_dataframe"
            ),
            Node(
                func=get_movie_list,
                inputs="metadata_list",
                outputs="movie_list",
                name="get_movie_list"
            ),
            Node(
                func=sepalate_trend_residual,
                inputs=["cleansed_openface", "movie_list", "params:trend.frac", "params:trend.it", "params:trend.delta", "params:trend.return_sorted"],
                outputs="trend_residual_table",
                name="sepalate_trend_residual"
            ),
            Node(
                func=trend_mean_table_sepalated,
                inputs=["trend_residual_table", "movie_list"],
                outputs="trend_mean_table",
                name="trend_mean_table"
                ),
            Node(
                func=trend_var_table_sepalated,
                inputs=["trend_residual_table", "movie_list"],
                outputs="trend_var_table",
                name="trend_var_table"
                ),
            Node(
                func=get_peak_freq,
                inputs=["trend_residual_table", "movie_list", "params:peak.height", "params:peak.distance", "params:peak.prominence"],
                outputs="peak_freq_series",
                name="get_peak_freq"
                ),
            Node(
                func=peak_freq_table_residual,
                inputs=["peak_freq_series","trend_residual_table", "movie_list"],
                outputs="peak_freq_table",
                name="peak_freq_table_residual"
            ),
            
            Node(
                func=integrate_features_table,
                inputs=["trend_mean_table","trend_var_table", "peak_freq_table"],
                outputs="integrated_feature",
                name="integrate_features_table"
            ),
            Node(
                func=integrate_metadata,
                inputs=["integrated_feature", "metadata_list"],
                outputs="features_table",
                name="integrate_metadata"
            )
        ]
    )
