import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np


def make_run_id(prefix: str = "run") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


class JsonlWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def compact_indices(indices: List[int], max_items: int = 200) -> Union[List[int], Dict[str, Any]]:
    if len(indices) <= max_items:
        return indices
    head = indices[: max_items // 2]
    tail = indices[-max_items // 2 :]
    return {"count": len(indices), "head": head, "tail": tail}


def position_stats(indices: List[int]) -> Dict[str, Optional[float]]:
    if not indices:
        return {"min": None, "mean": None, "max": None}
    arr = np.array(indices, dtype=np.float32)
    return {"min": float(arr.min()), "mean": float(arr.mean()), "max": float(arr.max())}


def token_type_summary(token_ids: List[int], tokenizer) -> Dict[str, int]:
    import string

    summary = {"punctuation": 0, "numeric": 0, "whitespace": 0, "other": 0}
    for tid in token_ids:
        s = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
        if not s:
            summary["other"] += 1
            continue
        if s.isspace():
            summary["whitespace"] += 1
        elif s.isdigit():
            summary["numeric"] += 1
        elif all((c in string.punctuation) for c in s.strip()):
            summary["punctuation"] += 1
        else:
            summary["other"] += 1
    return summary


class RunLogger:
    def __init__(
        self,
        root_dir: str = "logs",
        run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.run_id = run_id or make_run_id()
        self.root_dir = root_dir
        self.run_dir = os.path.join(root_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.event_writer = JsonlWriter(os.path.join(self.run_dir, "events.jsonl"))
        self.token_writer = JsonlWriter(os.path.join(self.run_dir, "tokens.jsonl"))

        if config is not None:
            with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

    def log_event(self, record: Dict[str, Any]) -> None:
        self.event_writer.write(record)

    def log_token(self, record: Dict[str, Any]) -> None:
        self.token_writer.write(record)

    def close(self) -> None:
        self.event_writer.close()
        self.token_writer.close()
