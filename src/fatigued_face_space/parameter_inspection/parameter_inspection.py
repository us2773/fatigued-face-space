"""Inspect the effective parameter values passed to pipeline nodes."""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from kedro.framework.project import pipelines, settings
from kedro.framework.session import KedroSession
from kedro.pipeline import Pipeline

_MISSING = object()


class ParameterInspectionError(Exception):
    """A requested pipeline, environment, or parameter cannot be inspected."""


@dataclass(frozen=True)
class ParameterUse:
    node_name: str
    argument_name: str
    annotation: Any


@dataclass(frozen=True)
class ParameterRow:
    name: str
    value: str
    actual_type: str
    expected_type: str
    result: Literal["OK", "ERROR", "MISSING", "UNVERIFIED"]
    used_by: tuple[str, ...]


# Nodeが使用するパラメータと引数を取得
def _parameter_uses(pipeline: Pipeline) -> dict[str, list[ParameterUse]]:
    """Find the node arguments that use each parameter."""
    uses: dict[str, list[ParameterUse]] = defaultdict(list)
    for node in pipeline.nodes:
        signature = inspect.signature(node.func, follow_wrapped=False)
        try:
            annotations = get_type_hints(node.func)
        except Exception:
            annotations = {}

        # Kedro exposes ordered inputs, but only the original dict retains keyword names.
        raw_inputs = node._inputs
        bindings = (
            raw_inputs.items()
            if isinstance(raw_inputs, dict)
            else zip(signature.parameters, node.inputs, strict=False)
        )
        for argument_name, dataset_name in bindings:
            if dataset_name.startswith("params:"):
                parameter = signature.parameters[argument_name]
                annotation = annotations.get(argument_name, parameter.annotation)
                uses[dataset_name[7:]].append(
                    ParameterUse(node.name, argument_name, annotation)
                )
    return uses


# 指定した階層のパラメータ値を取得
def _value_at_path(parameters: Mapping, name: str) -> Any:
    """Read a parameter value from its path."""
    value: Any = parameters
    for part in name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


# 型注釈を表示用の文字列に変換
def _type_name(annotation: Any) -> str:
    """Return a readable name for a type annotation."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return "unknown"
    if annotation is type(None):
        return "None"
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return " | ".join(_type_name(arg) for arg in get_args(annotation))
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).removeprefix("typing.")


# 値が指定した型に合うかを確認
def _matches_type(value: Any, annotation: Any) -> bool | None:
    """Check whether a value matches a type annotation."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        result = None
    elif annotation is type(None):
        result = value is None
    elif annotation is float:
        result = type(value) in (float, int)
    elif annotation in (int, bool, str):
        result = type(value) is annotation
    elif (origin := get_origin(annotation)) in (Union, UnionType):
        matches = [_matches_type(value, arg) for arg in get_args(annotation)]
        if True in matches:
            result = True
        else:
            result = None if None in matches else False
    else:
        result = _matches_collection(value, origin, get_args(annotation))
    return result


# リストや辞書の要素の型を確認
def _matches_collection(value: Any, origin: Any, args: tuple[Any, ...]) -> bool | None:
    """Check the types of items in a collection."""
    if origin is list:
        if not isinstance(value, list):
            return False
        return _all_match(value, args[0]) if args else True
    if origin is dict:
        if not isinstance(value, dict):
            return False
        if not args:
            return True
        return _combine_checks(
            [
                _all_match(value.keys(), args[0]),
                _all_match(value.values(), args[1]),
            ]
        )
    return None


# 複数の型確認結果を集約
def _combine_checks(checks: list[bool | None]) -> bool | None:
    """Combine type checks while keeping unknown results."""
    if False in checks:
        return False
    return None if None in checks else True


# 全要素が指定した型に合うかを確認
def _all_match(values: Any, annotation: Any) -> bool | None:
    """Check whether all items match the given type."""
    return _combine_checks([_matches_type(value, annotation) for value in values])


# パラメータの確認結果を一行に整理
def _row_for_parameter(
    name: str, parameters: Mapping, uses: list[ParameterUse]
) -> ParameterRow:
    """Build a table row for one parameter and its uses."""
    value = _value_at_path(parameters, name)
    expected = tuple(dict.fromkeys(_type_name(use.annotation) for use in uses))
    expected_type = " / ".join(expected)
    if value is _MISSING:
        result = "MISSING"
        value_display = "<missing>"
        actual_type = "-"
    else:
        checks = [_matches_type(value, use.annotation) for use in uses]
        result = (
            "ERROR" if False in checks else "UNVERIFIED" if None in checks else "OK"
        )
        value_display = repr(value)
        actual_type = type(value).__name__
    return ParameterRow(
        name=name,
        value=value_display,
        actual_type=actual_type,
        expected_type=expected_type,
        result=result,
        used_by=tuple(f"{use.node_name}({use.argument_name})" for use in uses),
    )


# 選択したPipelineのパラメータを確認
def inspect_parameters(
    project_path: Path, env: str, pipeline_name: str, selector: str | None = None
) -> tuple[list[ParameterRow], bool]:
    """Inspect only the parameters used by the selected registered pipeline."""
    if not env or env in {".", ".."} or "/" in env or "\\" in env:
        raise ParameterInspectionError("Environment: expected one directory name")

    project_path = Path(project_path)
    env_dir = project_path / settings.CONF_SOURCE / env
    if not env_dir.is_dir():
        raise ParameterInspectionError(
            f"Environment missing: {Path(settings.CONF_SOURCE) / env}"
        )
    try:
        pipeline = pipelines[pipeline_name]
    except KeyError as error:
        raise ParameterInspectionError(
            f"Pipeline not registered: {pipeline_name}"
        ) from error

    uses = _parameter_uses(pipeline)
    names = sorted(uses)
    if selector is not None:
        names = [
            name
            for name in names
            if name == selector or name.startswith(f"{selector}.")
        ]
        if not names:
            raise ParameterInspectionError(f"Pipeline parameter not found: {selector}")
    if not names:
        return [], False

    try:
        with KedroSession.create(
            project_path=project_path, env=env, save_on_close=False
        ) as session:
            parameters = session.load_context().params
    except Exception as error:
        raise ParameterInspectionError(
            f"Parameters could not be loaded ({type(error).__name__})"
        ) from error

    rows = [_row_for_parameter(name, parameters, uses[name]) for name in names]
    single = selector is not None and names == [selector]
    return rows, single
