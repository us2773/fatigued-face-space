from kedro.pipeline import Node, Pipeline

from .nodes import *

def feature_extraction(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=get_incomplete_movies,
                inputs=["params:openface.input_dir","params:openface.result_dir", "params:openface.import_dir", "params:openface.result_dir"],
                outputs="incomplete_movies",
                name="get_incomplete_movies",
            ),
            Node(
                func=run_openface,
                inputs=[
                    "params:openface.input_dir",
                    "params:openface.temporary_dir",
                    "incomplete_movies",
                    "params:docker.docker_id",
                    "params:docker.docker_workdir",
                    "params:docker.docker_processed_dir",
                    "params:docker.feature_extraction",
                    ],
                outputs="openface_result",
                name="run_openface",
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
                func=import_openface_files,
                inputs=["incomplete_movies", "params:openface.temporary_dir"],
                outputs="openface_result",
                name="import_openface_files"
            )
        ]
    )
    
def create_features_report(**kwargs) -> Pipeline :
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
                func=trend_mean_table,
                inputs=["cleansed_openface", "movie_list"],
                outputs="trend_mean_table",
                name="trend_mean_table"
                ),
            Node(
                func=trend_var_table,
                inputs=["cleansed_openface", "movie_list"],
                outputs="trend_var_table",
                name="trend_var_table"
                ),
            Node(
                func=peak_freq_table,
                inputs=["cleansed_openface", "movie_list"],
                outputs="peak_freq_table",
                name="peak_freq_table"
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