import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from fatigued_face_space.common.au_map import au_map_int


# Node: 残差のピーク点の時刻を取得
def get_peak_timeseries_madnormalize(
    partitions: dict[str, pd.DataFrame], 
    movie_list: list[str],
    mad_multiplier: float = 3.0,
    min_prominence: float = 0.1,
    max_width_seconds: float | None = None,
    min_distance_seconds: float | None = None,
    
    ) -> dict[str, pd.DataFrame] :
    
    """Find residual peaks using thresholds based on signal variation."""
    dfs: dict = {}
    
    for name in movie_list :
        loader = partitions[name]
        df = loader()
        result_data: dict = {}
        
        for au in au_map_int :
            df_series = df[f"AU{au:02}_residual"]
            r = df_series.values  
            t = df["timestamp"].values

            # 1. サンプリング間隔
            dt = float(np.median(np.diff(t)))

            # 2. Residualの揺らぎの大きさ
            center = np.median(r)
            mad = np.median(np.abs(r - center))
            scale = 1.4826 * mad
            

            # 3. 検出に必要な突出量
            prominence_threshold = max(
                min_prominence,
                mad_multiplier * scale,
            )

            # 4. ピーク検出条件
            options = {
                "prominence": prominence_threshold,
            }

            if max_width_seconds is not None:
                options["width"] = (None, max_width_seconds / dt)

            if min_distance_seconds is not None:
                options["distance"] = max(
                    1,
                    int(np.ceil(min_distance_seconds / dt)),
                )

            # 5. 検出位置をタイムスタンプへ変換
            peak_positions, _ = find_peaks(r, **options)
            result_data[f"AU{au:02}_peakpoints"] = pd.Series(t[peak_positions], dtype="float64") # Seriesを結合する場合、データ長の不一致によるエラーは起きない

        dfs[name] = pd.DataFrame(result_data)
    return dfs
