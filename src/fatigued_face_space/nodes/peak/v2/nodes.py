import pandas as pd
from scipy.signal import find_peaks

from fatigued_face_space.common.au_map import au_map_int


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
