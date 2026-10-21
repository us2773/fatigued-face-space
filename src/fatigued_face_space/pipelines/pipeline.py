from kedro.pipeline import Node, Pipeline

from .nodes import (
    get_cleansed_openface_timeseries,
    get_features_table,
    get_incomplete_movies,
    get_integrated_feature,
    get_movie_list,
    get_openface_from_csv_timeseries,
    get_openface_from_video_timeseries,
    get_peak_freq_feature,
    get_trend_mean_feature,
    get_trend_var_feature,
)

def feature_extraction(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=get_incomplete_movies,
                inputs=["params:openface.input_dir","params:openface.result_dir", "params:openface.input_ext", "params:openface.result_ext"],
                outputs="incomplete_movies",
                name="get_incomplete_movies",
            ),
            Node(
                func=get_openface_from_video_timeseries,
                inputs=[
                    "params:openface.input_dir",
                    "params:openface.temporary_dir",
                    "incomplete_movies",
                    "params:docker.docker_id",
                    "params:docker.docker_workdir",
                    "params:docker.docker_processed_dir",
                    "params:docker.feature_extraction",
                    "params:au_model"
                    ],
                outputs="openface_timeseries",
                name="get_openface_from_video_timeseries",
            )
        ])

def import_external_csv(**kwargs) -> Pipeline :
    return Pipeline(
        [
            Node(
                func=get_incomplete_movies,
                inputs=["params:openface.temporary_dir","params:openface.result_dir", "params:openface.temporary_ext", "params:openface.result_ext"],
                outputs="incomplete_movies",
                name="get_incomplete_movies",
            ),
            Node(
                func=get_openface_from_csv_timeseries,
                inputs=["incomplete_movies", "params:openface.temporary_dir"],
                outputs="openface_timeseries",
                name="get_openface_from_csv_timeseries"
            )
        ]
    )
    
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
