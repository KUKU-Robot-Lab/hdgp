from __future__ import annotations

RIGHT_EEF_NAME = "right"
LEFT_EEF_NAME = "left"

GRASP_DONE = "grasp_done"
LIFT_DONE = "lift_done"
ALIGN_DONE = "align_done"
POUR_DONE = "pour_done"

SUBTASK_TERM_SIGNALS = (GRASP_DONE, LIFT_DONE, ALIGN_DONE, POUR_DONE)


def validate_subtask_signals(signals: dict) -> None:
    """Fail fast when automatic annotation cannot see all pour phase signals."""
    missing = [signal for signal in SUBTASK_TERM_SIGNALS if signal not in signals]
    if missing:
        raise KeyError(f"missing pour mimic subtask term signals: {missing}")
