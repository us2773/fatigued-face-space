import glob
import os
import subprocess
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from .au_map import *

# 任意のディレクトリから指定した拡張子のファイルの名前を一覧取得
def get_filenames_extention(
    inputdir: str, 
    extention: str
    ) -> list[str] :
    
    all_movie = glob.glob(inputdir + extention)
    results = []
    for path in all_movie:
        movie_name = os.path.splitext(os.path.basename(path))[0]
        results.append(movie_name)
    return results

# Node: OpenFaceコマンド未実行動画の一覧を取得
def get_incomplete_movies(
    inputdir: str, 
    output_dir: str, 
    input_ext: str, 
    result_expt: str
    ) -> list[str]:
    
    # 01_raw/movie の中身をリスト化
    all_movie = get_filenames_extention(inputdir, input_ext)
    for path in all_movie:        
        print(path)
        
    # さらに02_intermediate/openface_result の中身もリスト化
    all_openface_result = get_filenames_extention(output_dir, result_expt)
    for path in all_openface_result:        
        print(path)
    
    # 前者にあって後者にない動画名のリストをOutputとする
    results = []
    for path in all_movie :
        if path not in all_openface_result :
            results.append(path)

    print(results)
    return results

# 外部コマンドの実行
def run_command(
    command: list[str]
    ) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    # 標準出力の表示
    if result.stdout:
        print(result.stdout)

    # 標準エラー出力の表示
    if result.stderr:
        print(result.stderr)

    # 異常終了の検出
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed. "
            f"returncode={result.returncode}, command={command}"
        )

    return result

# Dockerコンテナのクリーンナップ
def clean_docker_files(
    docker_id: str, 
    docker_workdir: str, 
    docker_processed_dir: str
    ) -> None:
    
    # Docker内の入力動画を削除
    run_command(
        [
            "docker",
            "exec",
            docker_id,
            "bash",
            "-c",
            f"rm -f {docker_workdir}/*.mp4",
        ]
    )

    # Docker内の出力結果を削除
    run_command(
        [
            "docker",
            "exec",
            docker_id,
            "bash",
            "-c",
            f"rm -rf {docker_processed_dir}/*",
        ]
    )

# openface_temporaryのクリーンナップ
def clean_openface_temporary(
    temporary_dir: str
    ) -> None :
    
    ext = ".csv"
    rm_list = get_filenames_extention(temporary_dir, "/*"+ext)
    print(rm_list)
    
    for file in rm_list:  
        os.remove(temporary_dir+"/"+file+ext)

# 一時ディレクトリを読み出してpd.DataFrameを返すことでフレームワークが出力を認識
def import_openface_files(
    incomplete_movies: list[str],
    output_dir: str
    ) -> dict[str, pd.DataFrame] :
    # 出力CSVの存在確認
    missing_csv = []
    for movie_name in incomplete_movies:
        csv_path = output_dir + "/" + movie_name + ".csv"
        if not os.path.isfile(csv_path):
            missing_csv.append(os.path.basename(csv_path))

    if missing_csv:
        raise RuntimeError(
            "OpenFace CSV not found: " + ", ".join(missing_csv)
        )

    # CSVからDataFrameへの変換
    results = {}
    for movie_name in incomplete_movies:
        csv_path = output_dir + "/" + movie_name + ".csv"
        results[movie_name] = pd.read_csv(
            csv_path,
            low_memory=False,
        )

    # temporaryディレクトリのクリーンナップ
    clean_openface_temporary(output_dir)
    return results
            
# CSVをクレンジングしDataFrameを取得
def cleansing_dataframe(
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

# Node: OpenFaceコマンドの実行関数
# OpenFace実行結果は一時ディレクトリに保存
def run_openface(
    input_dir: str,
    output_dir:str,
    incomplete_movies: list[str],
    docker_id: str,
    docker_workdir: str,
    docker_processed_dir: str,
    feature_extraction: str,
) -> dict[str, pd.DataFrame]:
    # 未処理動画がない場合の終了
    if not incomplete_movies:
        print("No incomplete movies.")
        return {}

    # 一時保存先の存在確認
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Temporary directory not found: {output_dir}"
        )

    # Dockerの起動
    run_command(["docker", "start", docker_id])

    try:
        # 未処理動画をDockerへコピー
        for movie_name in incomplete_movies:
            movie_path = input_dir + "/" + movie_name + ".mp4"

            if not os.path.isfile(movie_path):
                raise FileNotFoundError(f"Movie not found: {movie_path}")

            print(f"Copying movie: {os.path.basename(movie_path)}")
            run_command(
                [
                    "docker",
                    "cp",
                    movie_path,
                    f"{docker_id}:{docker_workdir}",
                ]
            )

        # OpenFace FeatureExtractionの実行
        for movie_name in incomplete_movies:
            movie_filename = f"{movie_name}.mp4"
            docker_movie_path = f"{docker_workdir}/{movie_filename}"

            print(f"Processing movie: {movie_filename}")
            run_command(
                [
                    "docker",
                    "exec",
                    "--workdir",
                    docker_workdir,
                    docker_id,
                    feature_extraction,
                    "-f",
                    docker_movie_path,
                    "-au_static",
                ]
            )

        # 出力CSVを一時保存先へコピー
        for movie_name in incomplete_movies:
            csv_name = f"{movie_name}.csv"
            docker_csv_path = f"{docker_processed_dir}/{csv_name}"
            temporary_csv_path = output_dir + "/" + csv_name

            print(f"Copying result: {csv_name}")
            run_command(
                [
                        "docker",
                        "cp",
                        f"{docker_id}:{docker_csv_path}",
                        temporary_csv_path,
                ]
            )

        # 出力CSVの存在確認
        missing_csv = []
        for movie_name in incomplete_movies:
            csv_path = output_dir + "/" + movie_name + ".csv"
            if not os.path.isfile(csv_path):
                missing_csv.append(os.path.basename(csv_path))

        if missing_csv:
            raise RuntimeError(
                "OpenFace CSV not found: " + ", ".join(missing_csv)
            )

        return import_openface_files(incomplete_movies, output_dir)

    finally:
        # Docker内の作業ファイルを削除
        clean_docker_files(docker_id, docker_workdir, docker_processed_dir)
        
        # temporary内のファイルを削除
        clean_openface_temporary(output_dir)

# 指定したAU時系列のトレンドとノイズを分離
def separate_AU_trend_noise(
    df: pd.DataFrame, 
    plot_num: int
    ) -> tuple[np.ndarray, np.ndarray] :
    
    AUR_start = df.columns.get_loc(" AU01_r")
    AUR_row = df.iloc[:, AUR_start + plot_num]
        
    # LOWESSによるトレンド抽出
    trend_est = lowess(AUR_row, df[" timestamp"], frac=0.1, return_sorted=False)
    # 残差計算
    residual = AUR_row - trend_est
    return (trend_est, residual)

# 指定したAU時系列のピーク点出現頻度の算出
def find_AU_peaks(
    df: pd.DataFrame, 
    plot_num: int
    ) -> tuple[list, list]:
    
    au_col = df.columns.get_loc(" AU01_r") + plot_num
    signal = df.iloc[:, au_col].values
    times = df[" timestamp"].values

    peaks, _ = find_peaks(signal, height=0.1, distance=5, prominence=0.1)
    return peaks, times

# Node: metadataから動画名のみを抽出
def get_movie_list(
    metadata: pd.DataFrame
    ) -> list[str] :
    
    return metadata["Name"].tolist()

# Node: トレンド平均を算出
def trend_mean_table(
    partitions: dict[str, pd.DataFrame], 
    movie_list: list[str]
    ) -> pd.DataFrame :
    
    column_mean = [f"AU{x:02}_trend_mean" for x in au_map_int]
    result = pd.DataFrame(columns=column_mean)
    
    for name in movie_list :
        loader = partitions[name]
        df = loader()
        trend_mean_list = []
        for au in au_index :
            
            trend_est, _ = separate_AU_trend_noise(df, au)
            trend_mean = trend_est.mean()
            trend_mean_list.append(trend_mean)
            
        result = pd.concat([result, pd.DataFrame([trend_mean_list], columns=column_mean, index=[name])])
    return result

# Node: トレンド分散を算出
def trend_var_table(
    partitions: dict[str, pd.DataFrame], 
    movie_list: list[str]
    )  -> pd.DataFrame :
    
    column_var = [f"AU{x:02}_trend_var" for x in au_map_int]
    result = pd.DataFrame(columns=column_var)
    
    for name in movie_list :
        loader = partitions[name]
        df = loader()
        trend_var_list = []
        for au in au_index :
            
            trend_est, _ = separate_AU_trend_noise(df, au)
            trend_var = trend_est.var()
            trend_var_list.append(trend_var)
        result = pd.concat([result, pd.DataFrame([trend_var_list], columns=column_var, index=[name])])
    return result

# Node: ピーク点出現頻度を算出
def peak_freq_table(
    partitions: dict[str, pd.DataFrame], 
    movie_list: list[str]
    ) -> pd.DataFrame :
    column_peakfreq = [f"AU{x:02}_peakfreq" for x in au_map_int]
    result = pd.DataFrame(columns=column_peakfreq)
    
    for name in movie_list :
        loader = partitions[name]
        df = loader()
        peak_freq_list = []
        for au in au_index :
            
            peaks, times = find_AU_peaks(df, au)
            num = len(peaks)
            if num == 0 :
                f = 0
            else :
                f = float(times[-1] / num)
            peak_freq_list.append(f)
        result = pd.concat([result, pd.DataFrame([peak_freq_list], columns=column_peakfreq, index=[name])])
    return result

# Node: 全特徴量のDataFrameを結合
def integrate_features_table(
    *feature_tables: pd.DataFrame
    ) -> pd.DataFrame :
    return pd.concat(feature_tables, axis=1)

# Node: 統合済み特徴量DataFrameにmetadataを結合
def integrate_metadata(
    integrated_feature: pd.DataFrame, 
    metadata: pd.DataFrame
    ) -> pd.DataFrame :
    metadata = metadata.set_index("Name")
    return pd.concat([metadata, integrated_feature], axis=1)