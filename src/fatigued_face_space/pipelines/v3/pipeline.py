from kedro.pipeline import Node, Pipeline

from ..nodes import (
    get_cleansed_openface_timeseries,
    get_features_table,
    get_integrated_feature,
    get_movie_list,
)
from ..v2.nodes import (
    get_peak_freq_feature,
    get_trend_mean_feature,
    get_trend_residual_timeseries,
    get_trend_var_feature,
)
from .nodes import (
    get_peak_timeseries_madnormalize
)



def create_features_report_v3(**kwargs) -> Pipeline :
    return Pipeline(
        [
            Node(
                func=get_cleansed_openface_timeseries,
                inputs=["openface_timeseries"],
                outputs="cleansed_openface_timeseries",
                name="get_cleansed_openface_timeseries"
            ),
            Node(
                func=get_movie_list,
                inputs="movie_metadata",
                outputs="movie_list",
                name="get_movie_list"
            ),
            Node(
                func=get_trend_residual_timeseries,
                inputs=["cleansed_openface_timeseries", "movie_list", "params:trend.frac", "params:trend.it", "params:trend.delta", "params:trend.return_sorted"],
                outputs="trend_residual_timeseries",
                name="get_trend_residual_timeseries"
            ),
            Node(
                func=get_trend_mean_feature,
                inputs=["trend_residual_timeseries", "movie_list"],
                outputs="trend_mean_feature",
                name="get_trend_mean_feature"
                ),
            Node(
                func=get_trend_var_feature,
                inputs=["trend_residual_timeseries", "movie_list"],
                outputs="trend_var_feature",
                name="get_trend_var_feature"
                ),
            Node(
                func=get_peak_timeseries_madnormalize,
                inputs=["trend_residual_timeseries", "movie_list", "params:peak.mad_multiplier", "params:peak.min_prominence", "params:peak.max_width_seconds", "params:peak.min_distance_seconds"],
                outputs="peak_timeseries",
                name="get_peak_timeseries"
                ),
            Node(
                func=get_peak_freq_feature,
                inputs=["peak_timeseries","trend_residual_timeseries", "movie_list"],
                outputs="peak_freq_feature",
                name="get_peak_freq_feature"
            ),
            
            Node(
                func=get_integrated_feature,
                inputs=["trend_mean_feature","trend_var_feature", "peak_freq_feature"],
                outputs="integrated_feature",
                name="get_integrated_feature"
            ),
            Node(
                func=get_features_table,
                inputs=["integrated_feature", "movie_metadata"],
                outputs="features_table",
                name="get_features_table"
            )
        ]
    )
