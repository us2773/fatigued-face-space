import pandas as pd


# Node: metadataから動画名のみを抽出
def get_movie_list(
    metadata: pd.DataFrame
    ) -> list[str] :
    
    return metadata["Name"].tolist()
