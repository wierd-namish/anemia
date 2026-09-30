"""
Evaluation package for clinical metrics and leakage auditing.
"""

from anemia_ai.evaluation.leakage import audit_partitions, audit_patient_leakage
from anemia_ai.evaluation.metrics import (
    aggregate_patient_predictions,
    compute_diagnostic_metrics,
)

__all__ = [
    "compute_diagnostic_metrics",
    "aggregate_patient_predictions",
    "audit_partitions",
    "audit_patient_leakage",
]
