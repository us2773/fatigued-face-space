import os

import pandas as pd


# CSVをクレンジングしDataFrameを取得
def get_cleansed_openface_timeseries(
    partitions: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:

    dfs = {}

    for name, loader in partitions.items():
        df = loader()
        df = df[(df[" success"] == 0) | (df[" success"] == 1)]

        movie_name = os.path.splitext(
            os.path.basename(name)
        )[0]

        df["movie_name"] = movie_name

        dfs[name] = df

        print(f"name:{name}")

    return dfs 
