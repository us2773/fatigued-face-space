from math import isfinite
from numbers import Real

import pandas as pd

from fatigued_face_space.common.directions_map import dir_match, directions

def _validate_min_median_difference(min_median_difference: float) -> None:
    if (
        isinstance(min_median_difference, bool)
        or not isinstance(min_median_difference, Real)
        or not isfinite(min_median_difference)
        or min_median_difference < 0
    ):
        raise ValueError("min_median_difference must be a finite nonnegative number")


def judge_direction(base_median, group_median, min_median_difference: float = 0.0):
    _validate_min_median_difference(min_median_difference)
    if pd.isna(base_median) or pd.isna(group_median):
        return directions.missing
    if abs(group_median - base_median) < min_median_difference:
        return directions.unchanged
    if group_median > base_median:
        return directions.up
    if group_median < base_median:
        return directions.down
    return directions.unchanged


def dffilter_by_dict(df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    is_match = pd.Series(True, index=df.index)

    for column, target_list in conditions.items():
        if column in df.columns:
            # 既存の条件に、新しいisinの条件をandで掛け合わせる
            is_match &= df[column].isin(target_list)

    return df[is_match]


def get_median_from_df(df: pd.DataFrame):
    return df.describe().loc["50%", :]


def get_fatigue_group_medians(df: pd.DataFrame, person: int) -> tuple:
    languid = dffilter_by_dict(df, {"person": [person], "class": [1, 2]})

    tense = dffilter_by_dict(df, {"person": [person], "class": [3, 4]})

    base = dffilter_by_dict(df, {"person": [person], "is_baseface": [1]})

    return languid, tense, base


def get_fatigue_features_tables(
    features_table: pd.DataFrame,
    persons: list[int],
    languid_classes: list[int],
    tense_classes: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate medians, retaining configured subjects even if a group is absent."""
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
    required = ["person", "class", "is_baseface", *features]
    missing = set(required) - set(features_table.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")
    table = features_table.loc[features_table["person"].isin(persons)]

    def aggregate(mask: pd.Series) -> pd.DataFrame:
        return (
            table.loc[mask]
            .groupby("person")[features]
            .median()
            .reindex(persons)
            .rename_axis("person")
            .reset_index()
        )

    return (
        aggregate(table["class"].isin(languid_classes)),
        aggregate(table["class"].isin(tense_classes)),
        aggregate(table["is_baseface"].eq(1)),
    )


def compare_fatigue_group_medians(
    base_df: pd.DataFrame,
    fatigue_df: pd.DataFrame,
    min_median_difference: float = 0.0,
) -> pd.DataFrame:
    _validate_min_median_difference(min_median_difference)
    base = base_df.set_index("person")
    fatigue = fatigue_df.set_index("person")
    if not base.index.is_unique or not fatigue.index.is_unique:
        raise ValueError("Each person must have exactly one median row")

    if set(base.columns) != set(fatigue.columns):
        raise ValueError("比較する特徴量が一致していません")

    persons = base.index.union(fatigue.index, sort=False)

    base = base.reindex(index=persons)
    fatigue = fatigue.reindex(index=persons, columns=base.columns)

    compared = pd.DataFrame(
        directions.missing,
        index=base.index,
        columns=base.columns,
    )

    valid = base.notna() & fatigue.notna()
    difference = fatigue - base
    sufficient = difference.abs().ge(min_median_difference)

    compared = compared.mask(valid, directions.unchanged)
    compared = compared.mask(valid & sufficient & difference.gt(0), directions.up)
    compared = compared.mask(valid & sufficient & difference.lt(0), directions.down)

    return compared.reset_index()


def summarize_direction_matches(
    compared_df: pd.DataFrame,
) -> pd.Series:
    values = compared_df.drop(columns="person")

    result = pd.Series(
        dir_match.nomatch,
        index=values.columns,
        dtype="object",
    )

    if values.empty:
        return result

    # Match the Notebook's precedence: missing, uniform direction, unchanged mix.
    values = values.replace({"unchange": directions.unchanged})
    missing = (values.isna() | values.eq(directions.missing)).any()
    allowed = [directions.up, directions.down, directions.unchanged, directions.missing]
    invalid = ~(values.isna() | values.isin(allowed))
    if invalid.loc[:, ~missing].any().any():
        raise ValueError("Unexpected direction value")
    result = result.mask(
        values.eq(directions.unchanged).any(), dir_match.unchanged_mixed
    )
    result = result.mask(values.eq(directions.up).all(), dir_match.up)
    result = result.mask(values.eq(directions.down).all(), dir_match.down)
    result = result.mask(values.eq(directions.unchanged).all(), dir_match.unchanged)
    return result.mask(missing, dir_match.missing)


def summarize_fatigue_direction_matches(
    languid_compared_features_table: pd.DataFrame,
    tense_compared_features_table: pd.DataFrame,
) -> pd.DataFrame:
    if set(languid_compared_features_table.columns) != set(
        tense_compared_features_table.columns
    ):
        raise ValueError("Feature columns do not match")
    return pd.DataFrame(
        {
            "languid": summarize_direction_matches(languid_compared_features_table),
            "tense": summarize_direction_matches(tense_compared_features_table),
        }
    ).T
