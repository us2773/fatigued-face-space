import pandas as pd
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess

from ..au_map import au_map_int


# Node: 複数のOpenFace時系列からTrendとResidualを出力
def get_trend_residual_timeseries(
    partitions: dict[str, pd.DataFrame],
    movie_list: list[str],
    frac: float,
    it: int,
    delta: float,
    return_sorted: bool
) -> dict[str, pd.DataFrame]:

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


# Node: 残差のピーク点の時刻を取得
def get_peak_timeseries(
    partitions: dict[str, pd.DataFrame], 
    movie_list: list[str],
    height: float,
    distance: float, 
    prominence: float
    ) -> dict[str, pd.DataFrame] :
    
    dfs: dict = {}
    
    for name in movie_list :
        loader = partitions[name]
        df = loader()
        result_data: dict = {}
        
        for au in au_map_int :
            df_series = df[f"AU{au:02}_residual"]
            signal = df_series.values  
            times = df["timestamp"].values

            peaks, _ = find_peaks(signal, height=height, distance=distance, prominence=prominence)
            result_data[f"AU{au:02}_peakpoints"] = pd.Series(times[peaks], dtype="float64") # Seriesを結合する場合、データ長の不一致によるエラーは起きない
            
        dfs[name] = pd.DataFrame(result_data)
    return dfs



def get_peak_freq_feature(
    peak_partitions: dict[str, pd.DataFrame], 
    tr_partitions: dict[str, pd.DataFrame], 
    movie_list: list[str], 
    ) :
    
    column_peakfreq = [f"AU{x:02}_peakfreq" for x in au_map_int]
    
    rows = {}

    for name in movie_list:
        peak_df = peak_partitions[name]()
        trend_df = tr_partitions[name]()
        
        # peak_timeseriesはピーク点のある時刻しか持ってない
        # 全体の撮影時刻はtrend_residual_timeseriesから取得
        end_time = trend_df["timestamp"].iloc[-1]

        values = []
        for au in au_map_int:
            num = peak_df[f"AU{au:02}_peakpoints"].count()
            values.append(0 if num == 0 else float(num / end_time))

        rows[name] = values

    return pd.DataFrame.from_dict(rows, orient="index", columns=column_peakfreq)

