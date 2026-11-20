from kedro.pipeline import Node, Pipeline, pipeline

from fatigued_face_space.nodes.feature_select.v2.nodes import filtered_features_table
from fatigued_face_space.pipelines.feature_select.v1.pipelines import (
    create_feature_select,
)


# 対象データを絞って特徴選択するPipelineを構築
def create_feature_select_v2(**kwargs) -> Pipeline:
    """Filter observations, then reuse the existing median comparison pipeline."""
    # 疲労度と分類から集計対象の行を選択
    filtering = Pipeline(
        [
            Node(
                func=filtered_features_table,
                inputs=[
                    "features_table",
                    "params:feature_select.selected_fatigue_levels",
                    "params:feature_select.selected_classes",
                ],
                outputs="filtered_features_table",
                name="filtered_features_table",
            ),
        ]
    )
    # 選択した表を既存の中央値比較処理へ接続
    selection = pipeline(
        create_feature_select(),
        inputs={"features_table": "filtered_features_table"},
    )
    return filtering + selection
