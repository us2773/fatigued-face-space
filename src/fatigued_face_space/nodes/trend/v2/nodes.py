import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from fatigued_face_space.common.au_map import au_map_int


# Node: 複数のOpenFace時系列からTrendとResidualを出力
def get_trend_residual_timeseries(
    partitions: dict[str, pd.DataFrame],
    movie_list: list[str],
    frac: float,
    it: int,
    delta: float,
    return_sorted: bool
) -> dict[str, pd.DataFrame]:

    """Split AU signals into trends and residuals for each movie."""
    dfs: dict = {}
    
    for name in movie_list :
        loader = partitions[name]
        df = loader()
        timestamp =  df[" timestamp"]
        result_data: dict = {}
        result_data["timestamp"] = timestamp
        
        for au in au_map_int :
            AUR_row = df.loc[:, f" AU{au:02}_r"]

            # LOWESSによるトレンド抽出
            trend_est = lowess(AUR_row, timestamp, frac=frac, it=it, delta=delta, return_sorted=return_sorted)
            
            result_data[f"AU{au:02}_trend"] = trend_est
            residual = AUR_row - trend_est
            result_data[f"AU{au:02}_residual"] = residual
        
        dfs[name] = pd.DataFrame(result_data)
    
    return dfs


# Node: トレンド平均を算出
def get_trend_mean_feature(
    partitions: dict[str, pd.DataFrame], 
    movie_list: list[str], 
    ) -> pd.DataFrame :
    
    """Calculate the mean AU trend for each movie."""
    column_mean = [f"AU{x:02}_trend_mean" for x in au_map_int]
    result = pd.DataFrame(columns=column_mean)
    
    for name in movie_list :
        loader = partitions[name]
        df: pd.DataFrame = loader()
        trend_mean_list = []
        for au in au_map_int :
            trend_mean = df[f"AU{au:02}_trend"].mean()
            trend_mean_list.append(trend_mean)
            
        result = pd.concat([result, pd.DataFrame([trend_mean_list], columns=column_mean, index=[name])])
    return result


# Node: トレンド分散を算出
def get_trend_var_feature(
    partitions: dict[str, pd.DataFrame], 
    movie_list: list[str], 
    ) -> pd.DataFrame :
    
    """Calculate the AU trend variance for each movie."""
    column_var = [f"AU{x:02}_trend_var" for x in au_map_int]
    result = pd.DataFrame(columns=column_var)
    
    for name in movie_list :
        loader = partitions[name]
        df: pd.DataFrame = loader()
        trend_var_list = []
        for au in au_map_int :
            trend_var = df[f"AU{au:02}_trend"].var()
            trend_var_list.append(trend_var)
            
        result = pd.concat([result, pd.DataFrame([trend_var_list], columns=column_var, index=[name])])
    return result
