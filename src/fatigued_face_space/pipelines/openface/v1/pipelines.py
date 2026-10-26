from kedro.pipeline import Node, Pipeline

from fatigued_face_space.nodes.openface.v1.nodes import (
    get_incomplete_movies,
    get_openface_from_video_timeseries,
)


def execute_openface(**kwargs) -> Pipeline:
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
