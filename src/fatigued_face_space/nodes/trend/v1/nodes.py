import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from fatigued_face_space.common.au_map import au_index, au_map_int


# 指定したAU時系列のトレンドと残差を分離
def separate_au_trend_residual(
    df: pd.DataFrame, 
    plot_num: int,
    frac: float,
    it: int,
    delta: float,
    return_sorted: bool
    ) -> tuple[np.ndarray, np.ndarray] :
    
    AUR_start = df.columns.get_loc(" AU01_r")
    AUR_row = df.iloc[:, AUR_start + plot_num]
        
    # LOWESSによるトレンド抽出
    trend_est = lowess(AUR_row, df[" timestamp"], frac=frac, it=it, delta=delta, return_sorted=return_sorted)
    # 残差計算
    residual = AUR_row - trend_est
    return (trend_est, residual)


# Node: トレンド平均を算出
def get_trend_mean_feature(
    partitions: dict[str, pd.DataFrame], 
    movie_list: list[str],
    frac: float,
    it: int,
    delta: float,
    return_sorted: bool
    ) -> pd.DataFrame :
    
    column_mean = [f"AU{x:02}_trend_mean" for x in au_map_int]
    result = pd.DataFrame(columns=column_mean)
    
    for name in movie_list :
        loader = partitions[name]
        df = loader()
        trend_mean_list = []
        for au in au_index :
            
            trend_est, _ = separate_au_trend_residual(df, au, frac, it, delta, return_sorted)
            trend_mean = trend_est.mean()
            trend_mean_list.append(trend_mean)
            
        result = pd.concat([result, pd.DataFrame([trend_mean_list], columns=column_mean, index=[name])])
    return result


# Node: トレンド分散を算出
def get_trend_var_feature(
    partitions: dict[str, pd.DataFrame], 
    movie_list: list[str],
    frac: float,
    it: int,
    delta: float,
    return_sorted: bool
    )  -> pd.DataFrame :
    
    column_var = [f"AU{x:02}_trend_var" for x in au_map_int]
    result = pd.DataFrame(columns=column_var)
    
    for name in movie_list :
        loader = partitions[name]
        df = loader()
        trend_var_list = []
        for au in au_index :
            
            trend_est, _ = separate_au_trend_residual(df, au, frac, it, delta, return_sorted)
            trend_var = trend_est.var()
            trend_var_list.append(trend_var)
        result = pd.concat([result, pd.DataFrame([trend_var_list], columns=column_var, index=[name])])
    return result
