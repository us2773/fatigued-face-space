"""Experiment checks using Kedro's real configuration loader."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project

from fatigued_face_space.cli import cli
from fatigued_face_space.experiment import requirements_for_pipeline

PROJECT_ROOT = Path(__file__).parents[1]


def _write_experiment(
    tmp_path: Path, *, metadata: bool, prominence: bool, metadata_file: str = "shared"
) -> None:
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
    (environment / "globals.yml").write_text(
        f"metadata_file: {metadata_file}\n", encoding="utf-8"
    )
    metadata_path = tmp_path / "metadata_${globals:metadata_file}.csv"
    (base / "catalog.yml").write_text(
        f"movie_metadata:\n  type: pandas.CSVDataset\n  filepath: '{metadata_path.as_posix()}'\n",
        encoding="utf-8",
    )
    if metadata:
        # The preflight checks existence, not the CSV's contents or columns.
        (tmp_path / f"metadata_{metadata_file}.csv").write_text(
            "invalid CSV contents", encoding="utf-8"
        )


def _write_v3_parameters(tmp_path: Path, min_distance_entry: str | None) -> None:
    lines = [
        "peak:",
        "  mad_multiplier: 3.0",
        "  min_prominence: 0.1",
        "  max_width_seconds: null",
    ]
    if min_distance_entry is not None:
        lines.append(f"  {min_distance_entry}")
    (tmp_path / "conf" / "demo" / "parameters.yml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
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
    assert "Metadata selection: metadata_file=shared" in result.output
    assert "metadata_shared.csv exists" in result.output
    assert "PASS: 0 errors" in result.output


def test_two_environments_can_share_metadata_file(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=True, prominence=True)
    variation = tmp_path / "conf" / "variation"
    variation.mkdir()
    (variation / "globals.yml").write_text("metadata_file: shared\n", encoding="utf-8")
    (variation / "parameters.yml").write_text(
        "peak:\n  height: 0.2\n  distance: 7\n  prominence: 0.3\n",
        encoding="utf-8",
    )

    for env in ("demo", "variation"):
        result = CliRunner().invoke(
            cli,
            ["expt", "check", f"--env={env}", "--pipeline=v2"],
            obj=SimpleNamespace(project_path=tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "metadata_shared.csv exists" in result.output


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


def test_v3_accepts_explicit_null_parameters(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=True, prominence=True)
    _write_v3_parameters(tmp_path, "min_distance_seconds: null")

    result = CliRunner().invoke(
        cli,
        ["expt", "check", "--env=demo", "--pipeline=v3"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert "PASS: 0 errors" in result.output


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        (None, "Parameter missing: params:peak.min_distance_seconds"),
        (
            'min_distance_seconds: ""',
            "Parameter blank: params:peak.min_distance_seconds",
        ),
        ("min_distance_seconds:", "Parameter blank: params:peak.min_distance_seconds"),
    ],
)
def test_v3_rejects_missing_or_blank_parameter(tmp_path, entry, error):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=True, prominence=True)
    _write_v3_parameters(tmp_path, entry)

    result = CliRunner().invoke(
        cli,
        ["expt", "check", "--env=demo", "--pipeline=v3"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 1, result.output
    assert error in result.output
    if entry is not None:
        assert f"{Path('conf') / 'demo' / 'parameters.yml'}:5" in result.output


def test_v3_uses_effective_environment_definition(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=True, prominence=True)
    base = tmp_path / "conf" / "base" / "parameters.yml"
    base.write_text(
        base.read_text(encoding="utf-8") + "peak:\n  min_distance_seconds:\n",
        encoding="utf-8",
    )
    _write_v3_parameters(tmp_path, "min_distance_seconds: null")

    result = CliRunner().invoke(
        cli,
        ["expt", "check", "--env=demo", "--pipeline=v3"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 0, result.output


def test_v3_reports_blank_value_in_additional_parameters_file(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=True, prominence=True)
    _write_v3_parameters(tmp_path, None)
    extra = tmp_path / "conf" / "demo" / "parameters_v3.yml"
    extra.write_text("peak:\n  min_distance_seconds:\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        ["expt", "check", "--env=demo", "--pipeline=v3"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 1, result.output
    assert "Parameter blank: params:peak.min_distance_seconds" in result.output
    assert f"{Path('conf') / 'demo' / 'parameters_v3.yml'}:2" in result.output


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


def test_metadata_file_is_required(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=True, prominence=True)
    (tmp_path / "conf" / "demo" / "globals.yml").write_text(
        "expt_name: demo\n", encoding="utf-8"
    )

    result = CliRunner().invoke(
        cli,
        ["expt", "check", "--env=demo", "--pipeline=v2"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 1, result.output
    assert "Metadata selection: metadata_file is unset" in result.output


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


def test_check_command_runs_existing_preflight(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=True, prominence=True)

    result = CliRunner().invoke(
        cli,
        ["check", "--env=demo", "--pipeline=v2"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert "Parameters: 7 required keys found" in result.output
    assert "PASS: 0 errors" in result.output


def test_check_params_lists_pipeline_values_and_types(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=False, prominence=True)
    _write_v3_parameters(tmp_path, "min_distance_seconds: null")

    result = CliRunner().invoke(
        cli,
        ["check", "params", "--env=demo", "--pipeline=v3"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert "peak.min_distance_seconds" in result.output
    assert "NoneType  float | None  OK" in result.output
    assert "trend.it" in result.output
    assert "8 parameters, 0 errors" in result.output


def test_check_params_group_and_single_selection(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=False, prominence=True)
    _write_v3_parameters(tmp_path, "min_distance_seconds: null")
    runner = CliRunner()
    obj = SimpleNamespace(project_path=tmp_path)

    group = runner.invoke(
        cli,
        ["check", "params", "peak", "--env=demo", "--pipeline=v3"],
        obj=obj,
    )
    single = runner.invoke(
        cli,
        [
            "check",
            "params",
            "peak.min_distance_seconds",
            "--env=demo",
            "--pipeline=v3",
        ],
        obj=obj,
    )

    assert group.exit_code == 0, group.output
    assert "4 parameters, 0 errors" in group.output
    assert "trend.frac" not in group.output
    assert single.exit_code == 0, single.output
    assert "value: None" in single.output
    assert "actual type: NoneType" in single.output
    assert "used by: get_peak_timeseries(min_distance_seconds)" in single.output


def test_check_params_shows_actual_int_compatible_with_float(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=False, prominence=True)

    result = CliRunner().invoke(
        cli,
        ["check", "params", "peak.distance", "--env=demo", "--pipeline=v2"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert "value: 5" in result.output
    assert "actual type: int" in result.output
    assert "expected type: float" in result.output
    assert "result: OK" in result.output


@pytest.mark.parametrize("setting", ['"3"', "null"])
def test_check_params_reports_type_mismatch(tmp_path, setting):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=False, prominence=True)
    base = tmp_path / "conf" / "base" / "parameters.yml"
    base.write_text(
        base.read_text(encoding="utf-8").replace("it: 3", f"it: {setting}"),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        ["check", "params", "trend.it", "--env=demo", "--pipeline=v2"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 1, result.output
    assert "expected type: int" in result.output
    assert "result: ERROR" in result.output


def test_check_params_reports_missing_value_and_unknown_selection(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=False, prominence=True)
    _write_v3_parameters(tmp_path, None)
    runner = CliRunner()
    obj = SimpleNamespace(project_path=tmp_path)

    missing = runner.invoke(
        cli,
        ["check", "params", "peak.min_distance_seconds", "--env=demo", "--pipeline=v3"],
        obj=obj,
    )
    unknown = runner.invoke(
        cli,
        ["check", "params", "peak_other", "--env=demo", "--pipeline=v3"],
        obj=obj,
    )

    assert missing.exit_code == 1, missing.output
    assert "value: <missing>" in missing.output
    assert "result: MISSING" in missing.output
    assert unknown.exit_code == 1, unknown.output
    assert "Pipeline parameter not found: peak_other" in unknown.output


def test_check_params_handles_pipeline_without_parameters(tmp_path):
    bootstrap_project(PROJECT_ROOT)
    _write_experiment(tmp_path, metadata=False, prominence=True)

    result = CliRunner().invoke(
        cli,
        ["check", "params", "--env=demo", "--pipeline=au_plots"],
        obj=SimpleNamespace(project_path=tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert "0 parameters, 0 errors" in result.output
