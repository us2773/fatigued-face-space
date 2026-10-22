"""Project-specific Kedro commands."""

from __future__ import annotations

import click

from fatigued_face_space.experiment import check_experiment
from fatigued_face_space.parameter_inspection import (
    ParameterInspectionError,
    ParameterRow,
    inspect_parameters,
)


@click.group(name="kedro")
def cli() -> None:
    """Commands for this Kedro project."""


def _run_experiment_check(metadata, env: str, pipeline: str) -> None:
    click.echo(f"Experiment check: env={env}  pipeline={pipeline}")
    results = check_experiment(metadata.project_path, env, pipeline)
    for result in results:
        click.echo(f"[{result.status}] {result.message}")

    errors = sum(result.status == "ERROR" for result in results)
    click.echo(
        f"\n{'FAIL' if errors else 'PASS'}: {errors} {'error' if errors == 1 else 'errors'}"
    )
    if errors:
        raise SystemExit(1)


def _show_parameter_table(rows: list[ParameterRow]) -> None:
    headers = ("PARAMETER", "VALUE", "ACTUAL", "EXPECTED", "RESULT")
    values = [
        (row.name, row.value, row.actual_type, row.expected_type, row.result)
        for row in rows
    ]
    widths = [
        max([len(header), *(len(value[index]) for value in values)])
        for index, header in enumerate(headers)
    ]
    for line in (headers, *values):
        click.echo(
            "  ".join(
                value.ljust(width) for value, width in zip(line, widths, strict=True)
            ).rstrip()
        )


@cli.group(name="check", invoke_without_command=True)
@click.option("--env", help="Experiment configuration directory name.")
@click.option("--pipeline", help="Registered pipeline name.")
@click.pass_context
def check(ctx: click.Context, env: str | None, pipeline: str | None) -> None:
    """Check an experiment or inspect its pipeline parameters."""
    if ctx.invoked_subcommand is not None:
        return
    if not env or not pipeline:
        raise click.UsageError("--env and --pipeline are required")
    _run_experiment_check(ctx.obj, env, pipeline)


@check.command(name="params")
@click.argument("parameter", required=False)
@click.option("--env", required=True, help="Experiment configuration directory name.")
@click.option("--pipeline", required=True, help="Registered pipeline name.")
@click.pass_obj
def check_params(metadata, parameter: str | None, env: str, pipeline: str) -> None:
    """Show effective values and check node argument types."""
    try:
        rows, single = inspect_parameters(
            metadata.project_path, env, pipeline, parameter
        )
    except ParameterInspectionError as error:
        click.echo(f"[ERROR] {error}")
        raise SystemExit(1) from error

    if single:
        row = rows[0]
        click.echo(f"params:{row.name}")
        click.echo(f"  value: {row.value}")
        click.echo(f"  actual type: {row.actual_type}")
        click.echo(f"  expected type: {row.expected_type}")
        click.echo(f"  used by: {', '.join(row.used_by)}")
        click.echo(f"  result: {row.result}")
    else:
        _show_parameter_table(rows)
        errors = sum(row.result != "OK" for row in rows)
        click.echo(f"\n{len(rows)} parameters, {errors} errors")

    if any(row.result != "OK" for row in rows):
        raise SystemExit(1)


@cli.group(hidden=True)
def expt() -> None:
    """Compatibility commands for existing experiment scripts."""


@expt.command(name="check")
@click.option("--env", required=True, help="Experiment configuration directory name.")
@click.option("--pipeline", required=True, help="Registered pipeline name.")
@click.pass_obj
def expt_check(metadata, env: str, pipeline: str) -> None:
    """Check experiment configuration and required metadata."""
    _run_experiment_check(metadata, env, pipeline)
