import glob
import os
import subprocess


# 任意のディレクトリから指定した拡張子のファイルの名前を一覧取得
def get_filenames_by_extension(
    inputdir: str, 
    extension: str
    ) -> list[str] :
    
    all_movie = glob.glob(inputdir + extension)
    results = []
    for path in all_movie:
        movie_name = os.path.splitext(os.path.basename(path))[0]
        results.append(movie_name)
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
    rm_list = get_filenames_by_extension(temporary_dir, "/*"+ext)
    print(rm_list)
    
    for file in rm_list:  
        os.remove(temporary_dir+"/"+file+ext)
