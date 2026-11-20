from math import isfinite
from numbers import Real

import pandas as pd

from fatigued_face_space.common.directions_map import dir_match, directions


# 中央値の差のしきい値が有効かを確認
def _validate_min_median_difference(min_median_difference: float) -> None:
    """Check that the threshold is finite and nonnegative."""
    if (
        isinstance(min_median_difference, bool)
        or not isinstance(min_median_difference, Real)
        or not isfinite(min_median_difference)
        or min_median_difference < 0
    ):
        raise ValueError("min_median_difference must be a finite nonnegative number")


# 二つの中央値の差から変化方向を判定
def judge_direction(base_median, group_median, min_median_difference: float = 0.0):
    """Compare two medians using the minimum difference threshold."""
    # しきい値と欠損を確認
    _validate_min_median_difference(min_median_difference)
    if pd.isna(base_median) or pd.isna(group_median):
        return directions.missing
    # 最小差に満たない値を変化なしとし、差の符号で方向を判定
    if abs(group_median - base_median) < min_median_difference:
        return directions.unchanged
    if group_median > base_median:
        return directions.up
    if group_median < base_median:
        return directions.down
    return directions.unchanged


# 指定した列と値の条件で表を絞り込み
def dffilter_by_dict(df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    """Filter rows using allowed values for each existing column."""
    # 全行を候補にして、列ごとの条件を積み重ねる
    is_match = pd.Series(True, index=df.index)

    for column, target_list in conditions.items():
        if column in df.columns:
            # 既存の条件に、新しいisinの条件をandで掛け合わせる
            is_match &= df[column].isin(target_list)

    return df[is_match]


# 数値列ごとの中央値を取得
def get_median_from_df(df: pd.DataFrame):
    """Return the median of each numeric column."""
    return df.describe().loc["50%", :]


# 指定した被験者の疲労群と基準表情を抽出
def get_fatigue_group_medians(df: pd.DataFrame, person: int) -> tuple:
    """Return fatigue and baseline rows for one subject."""
    # 被験者と疲労分類で各群を抽出
    languid = dffilter_by_dict(df, {"person": [person], "class": [1, 2]})

    tense = dffilter_by_dict(df, {"person": [person], "class": [3, 4]})

    # 同じ被験者の基準表情を抽出
    base = dffilter_by_dict(df, {"person": [person], "is_baseface": [1]})

    return languid, tense, base


# Node: 被験者ごとに疲労群と基準表情の中央値を算出
def get_fatigue_features_tables(
    features_table: pd.DataFrame,
    persons: list[int],
    languid_classes: list[int],
    tense_classes: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate medians, retaining configured subjects even if a group is absent."""
    # 被験者の指定と集計対象の特徴量を確認
    if not persons or len(persons) != len(set(persons)):
        raise ValueError("persons must be nonempty and unique")
    metadata_columns = {
        "Name",
        "person",
        "check_date",
        "class",
        "fatigue_level",
        "is_baseface",
    }
    features = [
        column
        for column in features_table.columns
        if column not in metadata_columns and not str(column).startswith("Unnamed:")
    ]
    if not features or len(features) != len(set(features)):
        raise ValueError("features must be nonempty and unique")
    # 必須列を確認し、対象被験者の行だけを残す
    required = ["person", "class", "is_baseface", *features]
    missing = set(required) - set(features_table.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")
    table = features_table.loc[features_table["person"].isin(persons)]

    # 対象行から全被験者の特徴量中央値を算出
    def aggregate(mask: pd.Series) -> pd.DataFrame:
        """Calculate group medians and keep every configured subject."""
        return (
            table.loc[mask]
            .groupby("person")[features]
            .median()
            .reindex(persons)
            .rename_axis("person")
            .reset_index()
        )

    # 疲労群と基準表情を同じ被験者順で集計
    return (
        aggregate(table["class"].isin(languid_classes)),
        aggregate(table["class"].isin(tense_classes)),
        aggregate(table["is_baseface"].eq(1)),
    )


# Node: 基準表情と疲労群の中央値から変化方向を判定
def compare_fatigue_group_medians(
    base_df: pd.DataFrame,
    fatigue_df: pd.DataFrame,
    min_median_difference: float = 0.0,
) -> pd.DataFrame:
    """Compare fatigue and baseline medians using an absolute threshold."""
    # しきい値と比較する表の構成を確認
    _validate_min_median_difference(min_median_difference)
    base = base_df.set_index("person")
    fatigue = fatigue_df.set_index("person")
    if not base.index.is_unique or not fatigue.index.is_unique:
        raise ValueError("Each person must have exactly one median row")

    if set(base.columns) != set(fatigue.columns):
        raise ValueError("比較する特徴量が一致していません")

    # 被験者と特徴量の順序を揃え、不足する値は欠損として残す
    persons = base.index.union(fatigue.index, sort=False)

    base = base.reindex(index=persons)
    fatigue = fatigue.reindex(index=persons, columns=base.columns)

    # 欠損を初期値にして、比較可能な値の差を計算
    compared = pd.DataFrame(
        directions.missing,
        index=base.index,
        columns=base.columns,
    )

    valid = base.notna() & fatigue.notna()
    difference = fatigue - base
    sufficient = difference.abs().ge(min_median_difference)

    # しきい値以上の差だけを上昇または下降として設定
    compared = compared.mask(valid, directions.unchanged)
    compared = compared.mask(valid & sufficient & difference.gt(0), directions.up)
    compared = compared.mask(valid & sufficient & difference.lt(0), directions.down)

    return compared.reset_index()


# 被験者間で特徴量の変化方向が一致するかを判定
def summarize_direction_matches(
    compared_df: pd.DataFrame,
) -> pd.Series:
    """Summarize direction agreement across subjects for each feature."""
    # 被験者列を除き、不一致を初期値にする
    values = compared_df.drop(columns="person")

    result = pd.Series(
        dir_match.nomatch,
        index=values.columns,
        dtype="object",
    )

    if values.empty:
        return result

    # 表記を統一し、欠損と想定外の方向を確認
    values = values.replace({"unchange": directions.unchanged})
    missing = (values.isna() | values.eq(directions.missing)).any()
    allowed = [directions.up, directions.down, directions.unchanged, directions.missing]
    invalid = ~(values.isna() | values.isin(allowed))
    if invalid.loc[:, ~missing].any().any():
        raise ValueError("Unexpected direction value")
    # 変化なしの混在と全員一致を判定し、欠損を最後に反映
    result = result.mask(
        values.eq(directions.unchanged).any(), dir_match.unchanged_mixed
    )
    result = result.mask(values.eq(directions.up).all(), dir_match.up)
    result = result.mask(values.eq(directions.down).all(), dir_match.down)
    result = result.mask(values.eq(directions.unchanged).all(), dir_match.unchanged)
    return result.mask(missing, dir_match.missing)


# Node: 疲労群ごとに特徴量の変化方向の一致を集計
def summarize_fatigue_direction_matches(
    languid_compared_features_table: pd.DataFrame,
    tense_compared_features_table: pd.DataFrame,
) -> pd.DataFrame:
    """Combine direction summaries for the two fatigue groups."""
    # 二つの疲労群で特徴量が揃っていることを確認
    if set(languid_compared_features_table.columns) != set(
        tense_compared_features_table.columns
    ):
        raise ValueError("Feature columns do not match")
    # 群ごとの判定を集計し、群を行に並べる
    return pd.DataFrame(
        {
            "languid": summarize_direction_matches(languid_compared_features_table),
            "tense": summarize_direction_matches(tense_compared_features_table),
        }
    ).T
