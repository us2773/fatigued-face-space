"""Read-only checks for an experiment before running a pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from kedro.framework.project import pipelines, settings
from kedro.framework.session import KedroSession
from kedro.pipeline import Pipeline


@dataclass(frozen=True)
class ExperimentRequirements:
    """Inputs that an experiment must provide for a selected pipeline."""

    parameters: tuple[str, ...]
    metadata: bool


@dataclass(frozen=True)
class CheckResult:
    status: Literal["OK", "ERROR", "SKIP"]
    message: str


def requirements_for_pipeline(pipeline: Pipeline) -> ExperimentRequirements:
    """Derive requirements from the pipeline instead of maintaining a second list."""
    inputs = pipeline.inputs()
    return ExperimentRequirements(
        parameters=tuple(
            sorted({item[7:] for item in inputs if item.startswith("params:")})
        ),
        metadata="movie_metadata" in inputs,
    )


def _parameter_state(parameters: Mapping, name: str) -> str:
    value = parameters
    for part in name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return "missing"
        value = value[part]
    if value is None or (isinstance(value, str) and not value.strip()):
        return "unset"
    return "present"


def _safe_error_name(error: Exception) -> str:
    """Avoid printing configuration values or credentials from exception text."""
    return type(error).__name__


def _load_config(
    name: str, load: Callable[[], Any], results: list[CheckResult]
) -> Any | None:
    try:
        return load()
    except Exception as error:
        results.append(
            CheckResult(
                "ERROR",
                f"Configuration: {name} could not be loaded ({_safe_error_name(error)})",
            )
        )
        return None


def _check_experiment_name(
    globals_config: Mapping | None,
    globals_file: Path,
    env: str,
    results: list[CheckResult],
) -> None:
    if not globals_file.is_file():
        display_file = Path(settings.CONF_SOURCE) / env / "globals.yml"
        results.append(
            CheckResult("ERROR", f"Experiment settings missing: {display_file}")
        )
    if globals_config is None:
        results.append(CheckResult("SKIP", "Experiment name: globals unavailable"))
    elif not globals_file.is_file():
        results.append(CheckResult("SKIP", "Experiment name: globals.yml missing"))
    elif globals_config.get("expt_name") != env:
        results.append(
            CheckResult("ERROR", f"Experiment name: expt_name must equal {env}")
        )
    else:
        results.append(CheckResult("OK", f"Experiment name: expt_name matches {env}"))


def _check_parameters(
    parameters: Mapping | None,
    requirements: ExperimentRequirements,
    results: list[CheckResult],
) -> None:
    if parameters is None:
        results.append(CheckResult("SKIP", "Parameters: configuration unavailable"))
        return

    missing = [
        (name, _parameter_state(parameters, name)) for name in requirements.parameters
    ]
    missing = [(name, state) for name, state in missing if state != "present"]
    for name, state in missing:
        results.append(CheckResult("ERROR", f"Parameter {state}: params:{name}"))
    if not missing:
        results.append(
            CheckResult(
                "OK", f"Parameters: {len(requirements.parameters)} required keys found"
            )
        )


def _check_metadata(
    catalog_config: Mapping | None,
    catalog: Any | None,
    requirements: ExperimentRequirements,
    env: str,
    results: list[CheckResult],
) -> None:
    if not requirements.metadata:
        results.append(
            CheckResult("SKIP", "Metadata: pipeline does not use movie_metadata")
        )
        return
    if catalog_config is None or catalog is None:
        results.append(CheckResult("SKIP", "Metadata: catalog unavailable"))
        return

    metadata_config = catalog_config.get("movie_metadata")
    filepath = (
        metadata_config.get("filepath")
        if isinstance(metadata_config, Mapping)
        else None
    )
    if not isinstance(filepath, str) or not filepath:
        results.append(
            CheckResult("ERROR", "Metadata: movie_metadata filepath missing")
        )
    elif Path(filepath).name != f"metadata_{env}.csv":
        results.append(
            CheckResult(
                "ERROR", f"Metadata path: expected metadata_{env}.csv, got {filepath}"
            )
        )
    else:
        try:
            exists = catalog.exists("movie_metadata")
        except Exception as error:
            results.append(
                CheckResult(
                    "ERROR",
                    f"Metadata: existence check failed ({_safe_error_name(error)})",
                )
            )
            return
        if exists:
            results.append(CheckResult("OK", f"Metadata: {filepath} exists"))
        else:
            results.append(CheckResult("ERROR", f"Metadata missing: {filepath}"))


def _check_context(
    context: Any,
    env: str,
    requirements: ExperimentRequirements,
    globals_file: Path,
    results: list[CheckResult],
) -> None:
    globals_config = _load_config(
        "globals", lambda: context.config_loader["globals"], results
    )
    parameters = _load_config("parameters", lambda: context.params, results)
    catalog_config = _load_config(
        "catalog", lambda: context.config_loader["catalog"], results
    )
    catalog = (
        _load_config("catalog", lambda: context.catalog, results)
        if catalog_config is not None
        else None
    )
    if all(
        config is not None
        for config in (globals_config, parameters, catalog_config, catalog)
    ):
        results.append(CheckResult("OK", f"Configuration: base + {env} loaded"))

    _check_experiment_name(globals_config, globals_file, env, results)
    _check_parameters(parameters, requirements, results)
    _check_metadata(catalog_config, catalog, requirements, env, results)


def check_experiment(
    project_path: Path, env: str, pipeline_name: str
) -> list[CheckResult]:
    """Check the effective Kedro configuration without executing pipeline nodes."""
    results: list[CheckResult] = []
    project_path = Path(project_path)
    if not env or env in {".", ".."} or "/" in env or "\\" in env:
        return [CheckResult("ERROR", "Environment: expected one directory name")]

    conf_source = project_path / settings.CONF_SOURCE
    env_dir = conf_source / env
    display_dir = Path(settings.CONF_SOURCE) / env
    if env_dir.is_dir():
        results.append(CheckResult("OK", f"Environment: {display_dir}"))
    else:
        results.append(CheckResult("ERROR", f"Environment missing: {display_dir}"))

    try:
        pipeline = pipelines[pipeline_name]
    except KeyError:
        results.append(
            CheckResult("ERROR", f"Pipeline not registered: {pipeline_name}")
        )
        return results
    if not env_dir.is_dir():
        return results

    try:
        with KedroSession.create(
            project_path=project_path, env=env, save_on_close=False
        ) as session:
            _check_context(
                session.load_context(),
                env,
                requirements_for_pipeline(pipeline),
                env_dir / "globals.yml",
                results,
            )
    except Exception as error:
        results.append(
            CheckResult(
                "ERROR",
                f"Configuration: Kedro session failed ({_safe_error_name(error)})",
            )
        )
    return results
