"""Open-loop latent rollout and prediction metrics.

Importing this package loads PyTorch. The environment, schema, and corpus
commands do not import it. Install the optional ``ml`` extra first.
"""

from worldforge.eval.metrics import (
    PREDICT_REPORT_FORMAT,
    PredictReport,
    format_predict_report,
    open_loop_metrics,
    read_predict_report,
    write_predict_report,
)
from worldforge.eval.rollout import (
    LatentRollout,
    TrajectoryWindow,
    rollout_latent,
    rollout_trajectory,
    trajectory_window,
)

__all__ = [
    "PREDICT_REPORT_FORMAT",
    "LatentRollout",
    "PredictReport",
    "TrajectoryWindow",
    "format_predict_report",
    "open_loop_metrics",
    "read_predict_report",
    "rollout_latent",
    "rollout_trajectory",
    "trajectory_window",
    "write_predict_report",
]
