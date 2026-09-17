from kedro.pipeline import Node, Pipeline

from .nodes import *

def feature_extraction(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=get_incomplete_movies,
                inputs=["params:openface.input_dir","params:openface.csv_dir"],
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
    
def create_features_report(**kwargs) -> Pipeline :
    return Pipeline(
        [
            Node(
                func=integrate_features_report,
                inputs=["openface_result", "metadata_list"],
                outputs="feature_list",
                name="integrate_features_report"
            )
        ]
    )