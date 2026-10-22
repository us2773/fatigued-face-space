"""Project-specific Kedro commands."""

from __future__ import annotations

import click

from fatigued_face_space.experiment import check_experiment


@click.group(name="kedro")
def cli() -> None:
    """Commands for this Kedro project."""


@cli.group()
def expt() -> None:
    """Inspect an experiment before running a pipeline."""


@expt.command()
@click.option("--env", required=True, help="Experiment configuration directory name.")
@click.option("--pipeline", required=True, help="Registered pipeline name.")
@click.pass_obj
def check(metadata, env: str, pipeline: str) -> None:
    """Check experiment configuration and required metadata."""
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
