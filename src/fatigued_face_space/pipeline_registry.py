"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from fatigued_face_space.pipelines.external_csv.v1.pipelines import import_external_csv
from fatigued_face_space.pipelines.feature_report.v1.pipelines import create_features_report
from fatigued_face_space.pipelines.feature_report.v2.pipelines import create_features_report_v2
from fatigued_face_space.pipelines.feature_report.v3.pipelines import create_features_report_v3
from fatigued_face_space.pipelines.feature_report.v4.pipelines import create_features_report_v4
from fatigued_face_space.pipelines.openface.v1.pipelines import execute_openface
from fatigued_face_space.pipelines.feature_select.v1.pipelines import create_feature_select
from fatigued_face_space.pipelines.feature_select.v2.pipelines import create_feature_select_v2
from fatigued_face_space.pipelines.plots.v1.pipelines import create_au_plots


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    return {
        "__default__": create_features_report(),
        "openface": execute_openface(),
        "external_csv": import_external_csv(),
        "au_plots": create_au_plots(),
        "ext_v2" : create_features_report_v2(),
        "ext_v3": create_features_report_v3(),
        "ext_v4": create_features_report_v4(),
        "feature_select": create_feature_select(),
        "ext_select": create_features_report_v4() + create_feature_select(),
        "feature_select_v2": create_feature_select_v2(),
        "ext_select_v2": create_features_report_v4() + create_feature_select_v2(),
    }
