"""
P2 Progress
===========
Reads and writes dataset/progress.json after EVERY decision.
On relaunch, already-decided records are skipped automatically.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger(__name__)

PROGRESS_FILE = "dataset/progress.json"


class Progress:

    def __init__(self, progress_path: str = PROGRESS_FILE):
        self.path    = Path(progress_path)
        self.decided: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.decided = data.get("decided", {})
                log.info(f"Resumed: {len(self.decided)} prior decisions loaded")
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f"Could not load progress file: {e} — starting fresh")
                self.decided = {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"decided": self.decided}, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def is_decided(self, record_id: str) -> bool:
        return record_id in self.decided

    def accept(self, record_id: str):
        self.decided[record_id] = "accept"
        self._save()

    def edit(self, record_id: str, new_label: str, original_label: str):
        self.decided[record_id] = {
            "action":         "edit",
            "new_label":      new_label,
            "original_label": original_label,
        }
        self._save()

    def discard(self, record_id: str):
        self.decided[record_id] = "discard"
        self._save()

    def skip(self, record_id: str):
        # Skipped not written to decided — will reappear on next launch
        pass

    def summary(self) -> Dict[str, int]:
        accepted  = sum(1 for v in self.decided.values() if v == "accept")
        discarded = sum(1 for v in self.decided.values() if v == "discard")
        edited    = sum(
            1 for v in self.decided.values()
            if isinstance(v, dict) and v.get("action") == "edit"
        )
        return {
            "accepted":  accepted,
            "edited":    edited,
            "discarded": discarded,
            "total":     len(self.decided),
        }
