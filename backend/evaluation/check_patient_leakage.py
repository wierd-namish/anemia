"""
Backward-compatibility facade for patient leakage audit.
Re-exports audit_partitions from anemia_ai.evaluation.
"""

from anemia_ai.evaluation.leakage import audit_partitions, audit_patient_leakage

__all__ = ["audit_partitions", "audit_patient_leakage"]
