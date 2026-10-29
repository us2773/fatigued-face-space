from kedro.pipeline import Node, Pipeline

from fatigued_face_space.nodes.openface.v1.nodes import (
    get_incomplete_movies,
    get_openface_from_video_timeseries,
)

from fatigued_face_space.nodes.cleansing.v1.nodes import get_cleansed_openface_timeseries


# 動画からOpenFaceの結果を取得するPipelineを構築
def execute_openface(**kwargs) -> Pipeline:
    """Build a pipeline that runs OpenFace on videos."""
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
            ),
            Node(
                func=get_cleansed_openface_timeseries,
                inputs=["openface_timeseries"],
                outputs="cleansed_openface_timeseries",
                name="get_cleansed_openface_timeseries"
            ),
        ])
