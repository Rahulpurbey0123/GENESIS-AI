"""
Global Feature Importance Module for GENESIS-AI Week 5 Explainability Engine.

Processes, validates, ranks, and formats global feature importances across all supported explanation strategies.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from backend.explainability.schemas import FeatureImportanceRecord


def format_global_importance(
    importance_records: List[FeatureImportanceRecord]
) -> List[FeatureImportanceRecord]:
    """
    Format, sort, and assign 1-indexed ranks to global feature importance records.

    Args:
        importance_records: Raw feature importance records.

    Returns:
        Sorted and properly ranked list of FeatureImportanceRecord.
    """
    if not importance_records:
        return []

    # Sort records by importance descending
    sorted_recs = sorted(importance_records, key=lambda r: r.importance, reverse=True)

    formatted: List[FeatureImportanceRecord] = []
    for rank_idx, rec in enumerate(sorted_recs, start=1):
        formatted.append(
            FeatureImportanceRecord(
                feature=rec.feature,
                importance=round(float(rec.importance), 4),
                rank=rank_idx,
                direction=rec.direction,
                mean_importance=round(float(rec.mean_importance), 4) if rec.mean_importance is not None else None,
                std_importance=round(float(rec.std_importance), 4) if rec.std_importance is not None else None
            )
        )

    return formatted
