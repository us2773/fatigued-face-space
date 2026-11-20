import pandas as pd


# Node: 全特徴量のDataFrameを結合
def get_integrated_feature(
    *feature_tables: pd.DataFrame
    ) -> pd.DataFrame :
    """Combine feature tables by their row index."""
    return pd.concat(feature_tables, axis=1)


# Node: 統合済み特徴量DataFrameにmetadataを結合
def get_features_table(
    integrated_feature: pd.DataFrame, 
    metadata: pd.DataFrame
    ) -> pd.DataFrame :
    """Join metadata and features by movie name."""
    metadata = metadata.set_index("Name")
    return pd.concat([metadata, integrated_feature], axis=1)
