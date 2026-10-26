import os

import pandas as pd

from ._helpers import (
    clean_docker_files,
    clean_openface_temporary,
    get_filenames_by_extension,
    run_command,
)


# Node: OpenFaceコマンド未実行動画の一覧を取得
def get_incomplete_movies(
    inputdir: str, 
    output_dir: str, 
    input_ext: str, 
    result_ext: str
    ) -> list[str]:
    
    # 01_raw/movie の中身をリスト化
    all_movie = get_filenames_by_extension(inputdir, input_ext)
    print("01_raw/movie")
    for path in all_movie:        
        print(path)
        
    # さらに02_intermediate/openface_result の中身もリスト化
    all_openface_result = get_filenames_by_extension(output_dir, result_ext)
    print("02_intermediate/openface_result")
    for path in all_openface_result:        
        print(path)
    
    # 前者にあって後者にない動画名のリストをOutputとする
    results = []
    for path in all_movie :
        if path not in all_openface_result :
            results.append(path)

    print("result")
    print(results)
    return results


# 一時ディレクトリを読み出してpd.DataFrameを返すことでフレームワークが出力を認識
def get_openface_from_csv_timeseries(
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


# Node: OpenFaceコマンドの実行関数
# OpenFace実行結果は一時ディレクトリに保存
def get_openface_from_video_timeseries(
    input_dir: str,
    output_dir:str,
    incomplete_movies: list[str],
    docker_id: str,
    docker_workdir: str,
    docker_processed_dir: str,
    feature_extraction: str,
    au_model: str,
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
                    f"-au_{au_model}",
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

        return get_openface_from_csv_timeseries(incomplete_movies, output_dir)

    finally:
        # Docker内の作業ファイルを削除
        clean_docker_files(docker_id, docker_workdir, docker_processed_dir)
        
        # temporary内のファイルを削除
        clean_openface_temporary(output_dir)
