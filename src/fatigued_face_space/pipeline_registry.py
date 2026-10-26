"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from fatigued_face_space.pipelines.external_csv.v1.pipelines import import_external_csv
from fatigued_face_space.pipelines.feature_report.v1.pipelines import create_features_report
from fatigued_face_space.pipelines.feature_report.v2.pipelines import create_features_report_v2
from fatigued_face_space.pipelines.feature_report.v3.pipelines import create_features_report_v3
from fatigued_face_space.pipelines.feature_report.v4.pipelines import create_features_report_v4
from fatigued_face_space.pipelines.openface.v1.pipelines import execute_openface
from fatigued_face_space.pipelines.plots.v1.pipelines import create_au_plots


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    return {
        "__default__": create_features_report(),
        "feature_extraction": execute_openface(),
        "external_csv": import_external_csv(),
        "au_plots": create_au_plots(),
        "v2" : create_features_report_v2(),
        "v3": create_features_report_v3(),
        "v4": create_features_report_v4()
    }
