"""Stage 5: Dataset caching — build once, reload from disk on every subsequent run.

Graph construction across 100k+ beats is the most expensive pipeline step, so
this module is the single place that should ever call ingestion/preprocessing/
visibility_graph in bulk. Training code should only ever touch the cached
.pt files produced here.
"""

import argparse
import logging

import torch
from torch_geometric.data import InMemoryDataset

from config import ALL_RECORDS, DS1_RECORDS, DS2_RECORDS, EXCLUDED_RECORDS, PROCESSED_DATA_DIR
from ingestion import load_record
from preprocessing import segment_beats
from visibility_graph import beat_to_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ECGVisibilityGraphDataset(InMemoryDataset):
    """PyG InMemoryDataset over cached visibility-graph beats for a given
    split ('ds1', 'ds2', or 'all')."""

    def __init__(self, root, split="all", record_ids=None, transform=None, pre_transform=None):
        self.split = split
        self.record_ids = record_ids or self._records_for_split(split)
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @staticmethod
    def _records_for_split(split):
        if split == "ds1":
            return DS1_RECORDS
        if split == "ds2":
            return DS2_RECORDS
        if split == "all":
            return ALL_RECORDS
        raise ValueError(f"Unknown split: {split}")

    @property
    def processed_file_names(self):
        return [f"ecg_visibility_graphs_{self.split}.pt"]

    def process(self):
        data_list = []
        for record_id in self.record_ids:
            if record_id in EXCLUDED_RECORDS:
                continue
            try:
                record = load_record(record_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("Skipping record %s: %s", record_id, exc)
                continue
            beats = segment_beats(record)
            for beat in beats:
                data_list.append(beat_to_graph(beat))
            logger.info("Record %s: %d graphs built (running total %d)",
                        record_id, len(beats), len(data_list))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])
        logger.info("Saved %d graphs to %s", len(data_list), self.processed_paths[0])


def build_split(split: str, root=PROCESSED_DATA_DIR):
    """Force-build (or load-from-cache) a given split and report class balance."""
    ds = ECGVisibilityGraphDataset(root=str(root), split=split)
    labels = [int(d.y.item()) for d in ds]
    from collections import Counter
    logger.info("Split '%s': %d graphs, class counts=%s", split, len(ds), Counter(labels))
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build cached visibility-graph dataset splits.")
    parser.add_argument("--split", choices=["ds1", "ds2", "all"], default="all")
    args = parser.parse_args()
    build_split(args.split)
