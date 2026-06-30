"""
Model registry & versioning.

A lightweight, file-based MLflow-style registry. Each registered model version
gets a directory under models_registry/artifacts/<name>/<version>/ containing the
serialized artifact plus a metadata.json (metrics, params, validation status,
data hash, timestamp). A per-model index.json tracks all versions and which one
is promoted to "production".
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from config import ARTIFACTS_DIR, REGISTRY_DIR

INDEX_FILE = REGISTRY_DIR / "registry_index.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_index() -> dict:
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text())
    return {"models": {}}


def _save_index(idx: dict) -> None:
    INDEX_FILE.write_text(json.dumps(idx, indent=2, default=str))


def data_fingerprint(*objs) -> str:
    """Stable hash of training inputs for reproducibility / drift tracking."""
    h = hashlib.sha256()
    for o in objs:
        try:
            h.update(str(o).encode("utf-8"))
        except Exception:
            h.update(repr(o).encode("utf-8"))
    return h.hexdigest()[:16]


def register_model(
    name: str,
    artifact,
    metrics: dict,
    params: dict | None = None,
    validation: dict | None = None,
    data_hash: str | None = None,
    promote: bool = False,
) -> dict:
    """Persist a new model version and (optionally) promote it to production."""
    idx = _load_index()
    model_entry = idx["models"].setdefault(name, {"versions": [], "production": None})
    version = len(model_entry["versions"]) + 1
    vdir = ARTIFACTS_DIR / name / f"v{version}"
    vdir.mkdir(parents=True, exist_ok=True)

    artifact_path = vdir / "model.joblib"
    try:
        joblib.dump(artifact, artifact_path)
        artifact_saved = True
    except Exception:
        artifact_saved = False

    meta = {
        "name": name,
        "version": version,
        "created_at": _now(),
        "metrics": metrics,
        "params": params or {},
        "validation": validation or {},
        "data_hash": data_hash,
        "artifact_path": str(artifact_path) if artifact_saved else None,
        "stage": "production" if promote else "staging",
    }
    (vdir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    model_entry["versions"].append(meta)
    if promote:
        for v in model_entry["versions"]:
            if v["version"] != version and v.get("stage") == "production":
                v["stage"] = "archived"
        model_entry["production"] = version
    _save_index(idx)
    return meta


def list_models() -> dict:
    return _load_index()["models"]


def get_versions(name: str) -> list[dict]:
    return _load_index()["models"].get(name, {}).get("versions", [])


def get_production_meta(name: str) -> dict | None:
    entry = _load_index()["models"].get(name)
    if not entry or entry.get("production") is None:
        return None
    for v in entry["versions"]:
        if v["version"] == entry["production"]:
            return v
    return None


def load_production_artifact(name: str):
    meta = get_production_meta(name)
    if meta and meta.get("artifact_path") and Path(meta["artifact_path"]).exists():
        return joblib.load(meta["artifact_path"])
    return None


def promote_version(name: str, version: int) -> bool:
    idx = _load_index()
    entry = idx["models"].get(name)
    if not entry:
        return False
    found = False
    for v in entry["versions"]:
        if v["version"] == version:
            v["stage"] = "production"
            found = True
        elif v.get("stage") == "production":
            v["stage"] = "archived"
    if found:
        entry["production"] = version
        _save_index(idx)
    return found
