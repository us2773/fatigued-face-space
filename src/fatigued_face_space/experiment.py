"""Read-only checks for an experiment before running a pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from kedro.framework.project import pipelines, settings
from kedro.framework.session import KedroSession
from kedro.pipeline import Pipeline
from yaml.nodes import MappingNode, Node, ScalarNode


@dataclass(frozen=True)
class ExperimentRequirements:
    """Inputs that an experiment must provide for a selected pipeline."""

    parameters: tuple[str, ...]
    metadata: bool


@dataclass(frozen=True)
class CheckResult:
    status: Literal["OK", "ERROR", "SKIP"]
    message: str


@dataclass(frozen=True)
class ParameterDefinition:
    path: Path
    node: Node

    def location(self, project_path: Path) -> str:
        return f"{self.path.relative_to(project_path)}:{self.node.start_mark.line + 1}"


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
    if value is None:
        return "null"
    if isinstance(value, str) and not value.strip():
        return "blank"
    return "present"


def _parameter_files(config_loader: Any, directory: Path) -> list[Path]:
    patterns = config_loader.config_patterns["parameters"]
    ignore_hidden = config_loader.ignore_hidden
    files = {
        path
        for pattern in patterns
        for path in directory.glob(pattern)
        if path.is_file()
        and path.suffix in {".yml", ".yaml", ".json"}
        and (
            not ignore_hidden
            or not any(
                part.startswith(".") for part in path.relative_to(directory).parts
            )
        )
    }
    return sorted(files)


def _parameter_documents(
    config_loader: Any, directory: Path
) -> list[tuple[Path, Node]]:
    documents = []
    for path in _parameter_files(config_loader, directory):
        node = yaml.compose(path.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
        if node is not None:
            documents.append((path, node))
    return documents


def _node_at_path(node: Node, parts: tuple[str, ...]) -> Node | None:
    for part in parts:
        if not isinstance(node, MappingNode):
            return None
        values = [
            value
            for key, value in node.value
            if isinstance(key, ScalarNode) and key.value == part
        ]
        if not values:
            return None
        node = values[-1]
    return node


def _parameter_definition(
    name: str,
    base_documents: list[tuple[Path, Node]],
    env_documents: list[tuple[Path, Node]],
    merge_strategy: str,
) -> ParameterDefinition | None:
    """Locate the winning definition under Kedro's base/environment merge."""
    parts = tuple(name.split("."))
    if merge_strategy == "destructive" and any(
        _node_at_path(node, parts[:1]) is not None for _, node in env_documents
    ):
        documents = env_documents
    else:
        documents = [*env_documents, *base_documents]
    for path, node in documents:
        parameter_node = _node_at_path(node, parts)
        if parameter_node is not None:
            return ParameterDefinition(path, parameter_node)
    return None


def _is_bare_empty(node: Node) -> bool:
    """Separate ``key:`` from explicit ``key: null`` before YAML loses that detail."""
    return (
        isinstance(node, ScalarNode)
        and node.tag == "tag:yaml.org,2002:null"
        and node.value == ""
        and node.style is None
    )


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


def _check_metadata_selection(
    globals_config: Mapping | None,
    globals_file: Path,
    env: str,
    results: list[CheckResult],
) -> str | None:
    if not globals_file.is_file():
        display_file = Path(settings.CONF_SOURCE) / env / "globals.yml"
        results.append(
            CheckResult("ERROR", f"Experiment settings missing: {display_file}")
        )
    if globals_config is None:
        results.append(CheckResult("SKIP", "Metadata selection: globals unavailable"))
    elif not globals_file.is_file():
        results.append(CheckResult("SKIP", "Metadata selection: globals.yml missing"))
    else:
        metadata_file = globals_config.get("metadata_file")
        if not isinstance(metadata_file, str) or not metadata_file.strip():
            results.append(
                CheckResult("ERROR", "Metadata selection: metadata_file is unset")
            )
        else:
            results.append(
                CheckResult("OK", f"Metadata selection: metadata_file={metadata_file}")
            )
            return metadata_file
    return None


def _check_parameters(
    parameters: Mapping | None,
    requirements: ExperimentRequirements,
    config_loader: Any,
    env: str,
    results: list[CheckResult],
) -> None:
    if parameters is None:
        results.append(CheckResult("SKIP", "Parameters: configuration unavailable"))
        return

    conf_source = Path(config_loader.conf_source)
    base_env = config_loader.base_env
    base_documents = _parameter_documents(config_loader, conf_source / base_env)
    env_documents = (
        _parameter_documents(config_loader, conf_source / env)
        if env != base_env
        else []
    )
    merge_strategy = config_loader.merge_strategy.get(
        "parameters", "destructive"
    ).lower()

    errors = 0
    for name in requirements.parameters:
        state = _parameter_state(parameters, name)
        if state == "missing":
            results.append(CheckResult("ERROR", f"Parameter missing: params:{name}"))
            errors += 1
            continue
        if state in {"null", "blank"}:
            definition = _parameter_definition(
                name, base_documents, env_documents, merge_strategy
            )
            if state == "blank" or (
                definition is not None and _is_bare_empty(definition.node)
            ):
                location = (
                    f" ({definition.location(conf_source.parent)})"
                    if definition
                    else ""
                )
                results.append(
                    CheckResult("ERROR", f"Parameter blank: params:{name}{location}")
                )
                errors += 1
    if not errors:
        results.append(
            CheckResult(
                "OK", f"Parameters: {len(requirements.parameters)} required keys found"
            )
        )


def _check_metadata(
    catalog_config: Mapping | None,
    catalog: Any | None,
    requirements: ExperimentRequirements,
    metadata_file: str | None,
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
    if metadata_file is None:
        results.append(CheckResult("SKIP", "Metadata: metadata_file unavailable"))
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
    elif Path(filepath).name != f"metadata_{metadata_file}.csv":
        results.append(
            CheckResult(
                "ERROR",
                f"Metadata path: expected metadata_{metadata_file}.csv, got {filepath}",
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

    metadata_file = _check_metadata_selection(
        globals_config, globals_file, env, results
    )
    _check_parameters(parameters, requirements, context.config_loader, env, results)
    _check_metadata(catalog_config, catalog, requirements, metadata_file, results)


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
