import os

import pandas as pd


# CSVをクレンジングしDataFrameを取得
# Node: OpenFaceの時系列を整形して動画名を付与
def get_cleansed_openface_timeseries(
    partitions: dict[str, pd.DataFrame],
    movie_list: list[str] | None = None,
) -> dict[str, pd.DataFrame]:

    """Filter OpenFace rows and add the movie name."""
    if movie_list is not None:
        partitions = {name: partitions[name] for name in movie_list}

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
