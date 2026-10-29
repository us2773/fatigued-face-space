from kedro.pipeline import Node, Pipeline


from fatigued_face_space.nodes.features_integrate.v1.nodes import get_features_table, get_integrated_feature
from fatigued_face_space.nodes.metadata.v1.nodes import get_movie_list
from fatigued_face_space.nodes.peak.v4.nodes import get_peak_timeseries_madnormalize
from fatigued_face_space.nodes.peak.v2.nodes import get_peak_freq_feature
from fatigued_face_space.nodes.trend.v2.nodes import (
    get_trend_mean_feature,
    get_trend_residual_timeseries,
    get_trend_var_feature,
)


# 特徴量を抽出して統合するv4のPipelineを構築
def create_features_report_v4(**kwargs) -> Pipeline :
    """Build version 4 of the feature extraction pipeline."""
    return Pipeline(
        [
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
                inputs=["trend_residual_timeseries", "movie_list", "params:peak.mad_multiplier", "params:peak.min_prominence", "params:peak.max_width_seconds", "params:peak.min_distance_seconds", "params:peak.height_multiplier", "params:peak.rel_height"],
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
