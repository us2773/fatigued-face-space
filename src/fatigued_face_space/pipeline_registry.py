"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from fatigued_face_space.pipelines.pipeline import (
    create_features_report,
    feature_extraction,
    import_external_csv,
)
from fatigued_face_space.pipelines.plots.pipeline import create_au_plots
from fatigued_face_space.pipelines.v2.pipeline import create_features_report_v2


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    features_report_pipeline = create_features_report()

    return {
        "__default__": features_report_pipeline,
        "feature_extraction": feature_extraction(),
        "external_csv": import_external_csv(),
        "au_plots": create_au_plots(),
        "v2" : create_features_report_v2()
    }
