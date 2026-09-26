"""One-step training for the Day 3 world model.

Importing this package loads PyTorch. The environment, schema, and corpus
commands do not import it. Install the optional ``ml`` extra first.
"""

from worldforge.train.loop import (
    TRAIN_RUN_FORMAT,
    EpochLog,
    TrainConfig,
    TrainResult,
    format_train_report,
    one_step_loss,
    read_metrics,
    read_run,
    replay_batch_seed,
    train_world_model,
)

__all__ = [
    "TRAIN_RUN_FORMAT",
    "EpochLog",
    "TrainConfig",
    "TrainResult",
    "format_train_report",
    "one_step_loss",
    "read_metrics",
    "read_run",
    "replay_batch_seed",
    "train_world_model",
]
