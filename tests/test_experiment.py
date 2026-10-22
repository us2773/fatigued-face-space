"""Experiment checks using Kedro's real configuration loader."""

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project

from fatigued_face_space.cli import cli
from fatigued_face_space.experiment import requirements_for_pipeline

PROJECT_ROOT = Path(__file__).parents[1]


def _write_experiment(tmp_path: Path, *, metadata: bool, prominence: bool) -> None:
    base = tmp_path / "conf" / "base"
    environment = tmp_path / "conf" / "demo"
    base.mkdir(parents=True)
    environment.mkdir(parents=True)
    (base / "parameters.yml").write_text(
        """openface:
  temporary_dir: incoming
  result_dir: processed
  temporary_ext: '*.csv'
  result_ext: '*.parquet'
trend:
  frac: 0.1
  it: 3
  delta: 0
  return_sorted: false
""",
        encoding="utf-8",
    )
    peak_parameters = "peak:\n  height: 0.1\n  distance: 5\n"
    if prominence:
        peak_parameters += "  prominence: 0.1\n"
    (environment / "parameters.yml").write_text(peak_parameters, encoding="utf-8")
    (environment / "globals.yml").write_text("expt_name: demo\n", encoding="utf-8")
    metadata_path = tmp_path / "metadata_${globals:expt_name}.csv"
    (base / "catalog.yml").write_text(
        f"movie_metadata:\n  type: pandas.CSVDataset\n  filepath: '{metadata_path.as_posix()}'\n",
        encoding="utf-8",
    )
    if metadata:
        # The preflight checks existence, not the CSV's contents or columns.
        (tmp_path / "metadata_demo.csv").write_text(
            "invalid CSV contents", encoding="utf-8"
        )


def test_requirements_follow_selected_pipeline():
    bootstrap_project(PROJECT_ROOT)

    v2 = requirements_for_pipeline(pipelines["v2"])
    external_csv = requirements_for_pipeline(pipelines["external_csv"])

    assert all(
        requirements_for_pipeline(pipelines[name]).metadata
        for name in ("__default__", "v2", "au_plots")
    )
    assert v2.parameters == (
        "peak.distance",
        "peak.height",
        "peak.prominence",
        "trend.delta",
        "trend.frac",
        "trend.it",
        "trend.return_sorted",
    )
    assert not any(
        requirements_for_pipeline(pipelines[name]).metadata
        for name in ("feature_extraction", "external_csv")
    )
    assert external_csv.parameters == (
        "openface.result_dir",
        "openface.result_ext",
        "openface.temporary_dir",
        "openface.temporary_ext",
    )


def test_check_accepts_merged_configuration_and_does_not_parse_metadata(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=True, prominence=True)

    result = CliRunner().invoke(
        cli,
        ["expt", "check", "--env=demo", "--pipeline=v2"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert "Parameters: 7 required keys found" in result.output
    assert "Metadata: " in result.output
    assert "PASS: 0 errors" in result.output


def test_check_reports_all_missing_requirements(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=False, prominence=False)

    result = CliRunner().invoke(
        cli,
        ["expt", "check", "--env=demo", "--pipeline=v2"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 1, result.output
    assert "Parameter missing: params:peak.prominence" in result.output
    assert "Metadata missing:" in result.output
    assert "FAIL: 2 errors" in result.output


def test_external_csv_does_not_require_metadata(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=False, prominence=True)

    result = CliRunner().invoke(
        cli,
        ["expt", "check", "--env=demo", "--pipeline=external_csv"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert "Metadata: pipeline does not use movie_metadata" in result.output
    assert "PASS: 0 errors" in result.output


def test_experiment_name_must_match_environment(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=True, prominence=True)
    (tmp_path / "conf" / "demo" / "globals.yml").write_text(
        "expt_name: another_experiment\n", encoding="utf-8"
    )

    result = CliRunner().invoke(
        cli,
        ["expt", "check", "--env=demo", "--pipeline=v2"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 1, result.output
    assert "Experiment name: expt_name must equal demo" in result.output
    assert "Metadata path: expected metadata_demo.csv" in result.output


def test_experiment_globals_are_required_even_without_metadata(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=False, prominence=True)
    (tmp_path / "conf" / "demo" / "globals.yml").unlink()

    result = CliRunner().invoke(
        cli,
        ["expt", "check", "--env=demo", "--pipeline=external_csv"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 1, result.output
    assert (
        f"Experiment settings missing: {Path('conf') / 'demo' / 'globals.yml'}"
        in result.output
    )
    assert "Metadata: pipeline does not use movie_metadata" in result.output
