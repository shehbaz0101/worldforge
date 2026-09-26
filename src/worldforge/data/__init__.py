"""Offline trajectory corpora.

``collect_corpus`` rolls the default environment and writes a directory.
``load_corpus`` reads that directory, a corpus JSONL file, or one trajectory
file. ``split_trajectories`` partitions episodes. ``sample_transition_batch``
draws a deterministic replay batch. Nothing here uses the network.
"""

from worldforge.data.dataset import (
    CorpusManifest,
    DatasetInfo,
    EpisodeSpec,
    TrajectorySplit,
    TransitionBatch,
    collect_corpus,
    format_dataset_info,
    inspect_dataset,
    load_corpus,
    load_manifest,
    sample_transition_batch,
    split_trajectories,
)

__all__ = [
    "CorpusManifest",
    "DatasetInfo",
    "EpisodeSpec",
    "TrajectorySplit",
    "TransitionBatch",
    "collect_corpus",
    "format_dataset_info",
    "inspect_dataset",
    "load_corpus",
    "load_manifest",
    "sample_transition_batch",
    "split_trajectories",
]
