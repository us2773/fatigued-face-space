from kedro.pipeline import Node, Pipeline

from fatigued_face_space.nodes.feature_select.v1.nodes import (
    compare_fatigue_group_medians,
    get_fatigue_features_tables,
    summarize_fatigue_direction_matches,
)


# 被験者間での変化方向を集計するPipelineを構築
def create_feature_select(**kwargs) -> Pipeline:
    """Build a pipeline that compares medians and summarizes direction agreement."""
    return Pipeline(
        [
            # 被験者ごとの中央値を疲労群と基準表情に分けて集計
            Node(
                func=get_fatigue_features_tables,
                inputs=[
                    "features_table",
                    "params:feature_select.persons",
                    "params:feature_select.languid_classes",
                    "params:feature_select.tense_classes",
                ],
                outputs=[
                    "languid_features_table",
                    "tense_features_table",
                    "base_features_table",
                ],
                name="get_fatigue_features_tables",
            ),
            # 各疲労群を基準表情と比較
            *[
                Node(
                    func=compare_fatigue_group_medians,
                    inputs=[
                        "base_features_table",
                        f"{group}_features_table",
                        "params:feature_select.min_median_difference",
                    ],
                    outputs=f"{group}_compared_features_table",
                    name=f"compare_{group}_features",
                )
                for group in ("languid", "tense")
            ],
            # 被験者間の変化方向の一致を集計
            Node(
                func=summarize_fatigue_direction_matches,
                inputs=[
                    "languid_compared_features_table",
                    "tense_compared_features_table",
                ],
                outputs="fatigue_direction_matches",
                name="summarize_fatigue_direction_matches",
            ),
        ]
    )
