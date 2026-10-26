from kedro.pipeline import Node, Pipeline

from fatigued_face_space.nodes.cleansing.v1.nodes import get_cleansed_openface_timeseries
from fatigued_face_space.nodes.features_integrate.v1.nodes import get_features_table, get_integrated_feature
from fatigued_face_space.nodes.metadata.v1.nodes import get_movie_list
from fatigued_face_space.nodes.peak.v1.nodes import get_peak_freq_feature
from fatigued_face_space.nodes.trend.v1.nodes import get_trend_mean_feature, get_trend_var_feature


def create_features_report(**kwargs) -> Pipeline :
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
                func=get_trend_mean_feature,
                inputs=["cleansed_openface_timeseries", "movie_list", "params:trend.frac", "params:trend.it", "params:trend.delta", "params:trend.return_sorted"],
                outputs="trend_mean_feature",
                name="get_trend_mean_feature"
                ),
            Node(
                func=get_trend_var_feature,
                inputs=["cleansed_openface_timeseries", "movie_list", "params:trend.frac", "params:trend.it", "params:trend.delta", "params:trend.return_sorted"],
                outputs="trend_var_feature",
                name="get_trend_var_feature"
                ),
            Node(
                func=get_peak_freq_feature,
                inputs=["cleansed_openface_timeseries", "movie_list", "params:peak.height", "params:peak.distance", "params:peak.prominence"],
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
