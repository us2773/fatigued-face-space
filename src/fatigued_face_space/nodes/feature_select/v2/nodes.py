import pandas as pd


# Node: 基準表情を残し疲労度と分類で対象行を選択
def filtered_features_table(
    features_table: pd.DataFrame,
    selected_fatigue_levels: list[int],
    selected_classes: list[int],
) -> pd.DataFrame:
    """Keep baseline rows and fatigue rows that match both selection lists.

    Other rows need both a selected fatigue level and a selected class.
    An empty selection list keeps baseline rows only.
    """
    # フィルタリングに必要な列と対象リストを確認
    required = {"fatigue_level", "class", "is_baseface"}
    missing = required - set(features_table.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")
    if not isinstance(selected_fatigue_levels, list):
        raise ValueError("selected_fatigue_levels must be a list")
    if not isinstance(selected_classes, list):
        raise ValueError("selected_classes must be a list")

    # 基準表情を残し、疲労側には疲労度と分類の両条件を適用
    baseline = features_table["is_baseface"].eq(1).fillna(False)
    selected = features_table["fatigue_level"].isin(
        selected_fatigue_levels
    ) & features_table["class"].isin(selected_classes)
    return features_table.loc[baseline | selected].copy()
