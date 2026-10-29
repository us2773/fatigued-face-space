from kedro.pipeline import Node, Pipeline

from fatigued_face_space.nodes.openface.v1.nodes import (
    get_incomplete_movies,
    get_openface_from_csv_timeseries,
)

from fatigued_face_space.nodes.cleansing.v1.nodes import get_cleansed_openface_timeseries

# 外部CSVから時系列を取得するPipelineを構築
def import_external_csv(**kwargs) -> Pipeline :
    """Build a pipeline that imports OpenFace CSV files."""
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
            ),
            Node(
                func=get_cleansed_openface_timeseries,
                inputs=["openface_timeseries"],
                outputs="cleansed_openface_timeseries",
                name="get_cleansed_openface_timeseries"
            ),
        ]
    )
