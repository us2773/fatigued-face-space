import pandas as pd
from scipy.signal import find_peaks

from fatigued_face_space.common.au_map import au_index, au_map_int


# 指定したAU時系列のピーク点を取得
def find_au_peaks(
    df: pd.DataFrame, 
    plot_num: int,
    height: float,
    distance: float, 
    prominence: float
    ) -> tuple[list, list]:
    
    """Find peak positions and timestamps in an AU signal."""
    au_col = df.columns.get_loc(" AU01_r") + plot_num
    signal = df.iloc[:, au_col].values
    times = df[" timestamp"].values

    peaks, _ = find_peaks(signal, height=height, distance=distance, prominence=prominence)
    return peaks, times


# Node: ピーク点出現頻度を算出
def get_peak_freq_feature(
    partitions: dict[str, pd.DataFrame], 
    movie_list: list[str],
    height: float,
    distance: float, 
    prominence: float
    ) -> pd.DataFrame :
    """Calculate peak-based AU features for each movie."""
    column_peakfreq = [f"AU{x:02}_peakfreq" for x in au_map_int]
    result = pd.DataFrame(columns=column_peakfreq)
    
    for name in movie_list :
        loader = partitions[name]
        df = loader()
        peak_freq_list = []
        for au in au_index :
            
            peaks, times = find_au_peaks(df, au, height, distance, prominence)
            num = len(peaks)
            if num == 0 :
                f = 0
            else :
                f = float(times[-1] / num)
            peak_freq_list.append(f)
        result = pd.concat([result, pd.DataFrame([peak_freq_list], columns=column_peakfreq, index=[name])])
    return result
