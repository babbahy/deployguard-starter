import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("modal")
spec = importlib.util.spec_from_file_location("train_modal", Path(__file__).parents[1] / "scripts/train_modal.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_rejects_split_changes_before_upload(tmp_path):
    manifest = {"audit": {"passed": True}, "splits": {
        name: {"sha256": "incorrect"} for name in ("train", "validation", "test")
    }}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "train.jsonl").write_text('{"label": 1}\n')
    with pytest.raises(ValueError, match="changed since the audit"):
        launcher.validate_data(tmp_path)


def test_rejects_missing_audit(tmp_path):
    (tmp_path / "manifest.json").write_text('{}')
    with pytest.raises(ValueError, match="leakage audit"):
        launcher.validate_data(tmp_path)
