"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
import fatigued_face_space.pipelines.pipeline as pl


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    feature_extraction = pl.feature_extraction()
    create_features_report = pl.create_features_report()
    import_external_csv = pl.import_external_csv()
    return {
        "__default__": create_features_report,
        "feature_extraction": feature_extraction,
        "external_csv": import_external_csv
            }
    
