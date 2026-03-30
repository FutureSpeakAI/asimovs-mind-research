"""
provenance.py — Attribution tracking for imported code.

Every piece of code imported from GitHub gets a three-stage provenance record:
  1. candidate — metadata captured when the code is first fetched
  2. experiment — recorded when the code is committed to train.py
  3. outcome — recorded after training with the imported code

The provenance log (provenance_log.jsonl) is APPEND-ONLY. Records are never
modified or deleted — this is enforced by the Third Law.
"""

import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

PROVENANCE_LOG = Path(__file__).parent / "provenance_log.jsonl"

# Licenses categorized by compatibility
ALLOWED_LICENSES = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
    "ISC", "CC-BY-4.0", "Unlicense", "0BSD",
}
COPYLEFT_LICENSES = {
    "GPL-2.0-only", "GPL-3.0-only", "GPL-2.0-or-later", "GPL-3.0-or-later",
    "LGPL-2.1-only", "LGPL-3.0-only", "LGPL-2.1-or-later", "LGPL-3.0-or-later",
    "AGPL-3.0-only", "AGPL-3.0-or-later",
}
BLOCKED_LICENSES = {"no-license", "proprietary", "all-rights-reserved"}

# Regex patterns for detecting license type from LICENSE file content
_LICENSE_PATTERNS = [
    (r"MIT License", "MIT"),
    (r"Apache License.*Version 2\.0", "Apache-2.0"),
    (r"BSD 2-Clause", "BSD-2-Clause"),
    (r"BSD 3-Clause", "BSD-3-Clause"),
    (r"GNU General Public License.*version 3", "GPL-3.0-only"),
    (r"GNU General Public License.*version 2", "GPL-2.0-only"),
    (r"GNU Lesser General Public License.*version 3", "LGPL-3.0-only"),
    (r"GNU Lesser General Public License.*version 2\.1", "LGPL-2.1-only"),
    (r"ISC License", "ISC"),
    (r"The Unlicense", "Unlicense"),
    (r"Permission is hereby granted, free of charge", "MIT"),  # MIT body
    (r"Redistribution and use in source and binary forms", "BSD-3-Clause"),  # BSD body
]


@dataclass
class ProvenanceRecord:
    """A single provenance record. One per stage per import."""
    record_id: str
    timestamp: str
    stage: str  # "candidate", "experiment", "outcome"
    repo_full_name: str
    data: dict = field(default_factory=dict)


def generate_record_id() -> str:
    """Generate a unique record ID."""
    return str(uuid.uuid4())[:12]


def content_hash(content: str) -> str:
    """SHA256 hash of content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def write_record(record: ProvenanceRecord) -> None:
    """Append a provenance record to the log. NEVER modifies existing records."""
    PROVENANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), ensure_ascii=False)
    with open(PROVENANCE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_candidate_record(
    record_id: str,
    repo_full_name: str,
    repo_url: str,
    license_spdx: Optional[str],
    commit_sha: str,
    file_path: str,
    source_content: str,
    extracted_component: str,
    scout_relevance: float,
    scout_trust: float,
    scanner_verdict: str,
    scanner_trust: float,
    target_zone: str,
    renames: Optional[dict] = None,
) -> None:
    """Write a candidate-stage provenance record."""
    record = ProvenanceRecord(
        record_id=record_id,
        timestamp=datetime.now().isoformat(),
        stage="candidate",
        repo_full_name=repo_full_name,
        data={
            "repo_url": repo_url,
            "license_spdx": license_spdx or "unknown",
            "commit_sha": commit_sha,
            "file_path": file_path,
            "content_hash": content_hash(source_content),
            "extracted_component": extracted_component,
            "scout_relevance_score": round(scout_relevance, 4),
            "scout_trust_score": round(scout_trust, 4),
            "scanner_verdict": scanner_verdict,
            "scanner_trust_score": round(scanner_trust, 4),
            "target_zone": target_zone,
            "renames_applied": renames or {},
        },
    )
    write_record(record)


def write_experiment_record(
    record_id: str,
    repo_full_name: str,
    git_commit_hash: str,
    insertion_start_line: int,
    insertion_end_line: int,
) -> None:
    """Write an experiment-stage provenance record."""
    record = ProvenanceRecord(
        record_id=record_id,
        timestamp=datetime.now().isoformat(),
        stage="experiment",
        repo_full_name=repo_full_name,
        data={
            "git_commit_hash": git_commit_hash,
            "train_py_insertion_start": insertion_start_line,
            "train_py_insertion_end": insertion_end_line,
        },
    )
    write_record(record)


def write_outcome_record(
    record_id: str,
    repo_full_name: str,
    val_bpb: float,
    baseline_val_bpb: float,
    peak_vram_mb: float,
    decision: str,
    reverted_commit: Optional[str] = None,
) -> None:
    """Write an outcome-stage provenance record."""
    record = ProvenanceRecord(
        record_id=record_id,
        timestamp=datetime.now().isoformat(),
        stage="outcome",
        repo_full_name=repo_full_name,
        data={
            "val_bpb": val_bpb,
            "baseline_val_bpb": baseline_val_bpb,
            "delta_bpb": round(baseline_val_bpb - val_bpb, 6),
            "peak_vram_mb": peak_vram_mb,
            "decision": decision,
            "reverted_commit": reverted_commit,
        },
    )
    write_record(record)


def load_provenance_log() -> list[dict]:
    """Read all provenance records."""
    if not PROVENANCE_LOG.exists():
        return []
    records = []
    for line in PROVENANCE_LOG.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def get_experiment_history() -> list[dict]:
    """Get all outcome records joined with their candidate data."""
    records = load_provenance_log()
    candidates = {r["record_id"]: r for r in records if r["stage"] == "candidate"}
    outcomes = [r for r in records if r["stage"] == "outcome"]
    history = []
    for outcome in outcomes:
        rid = outcome["record_id"]
        candidate = candidates.get(rid, {})
        history.append({
            "record_id": rid,
            "repo": outcome.get("repo_full_name", ""),
            "component": candidate.get("data", {}).get("extracted_component", ""),
            "val_bpb": outcome.get("data", {}).get("val_bpb", 0),
            "delta": outcome.get("data", {}).get("delta_bpb", 0),
            "decision": outcome.get("data", {}).get("decision", ""),
            "timestamp": outcome.get("timestamp", ""),
        })
    return history


def check_license_compliance(license_spdx: Optional[str]) -> tuple[bool, str]:
    """Check if a license is compliant. Returns (ok, reason)."""
    if not license_spdx or license_spdx.lower() in ("", "none", "null"):
        return False, "No license — cannot legally use this code"

    normalized = license_spdx.strip()

    if normalized in ALLOWED_LICENSES:
        return True, f"License {normalized} is fully compatible"

    if normalized in COPYLEFT_LICENSES:
        return False, f"License {normalized} is copyleft — requires acknowledgment and may impose restrictions"

    if normalized.lower() in {l.lower() for l in BLOCKED_LICENSES}:
        return False, f"License {normalized} is blocked"

    return False, f"Unknown license '{normalized}' — treating as blocked for safety"


def detect_license_from_text(text: str) -> Optional[str]:
    """Attempt to identify a license SPDX from LICENSE file content."""
    for pattern, spdx in _LICENSE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return spdx
    return None


def format_attribution_comment(
    record_id: str,
    repo_url: str,
    commit_sha: str,
    file_path: str,
    license_spdx: str,
    component_name: str,
    scout_trust: float,
    scanner_trust: float,
) -> str:
    """Generate the attribution comment block for insertion into train.py."""
    timestamp = datetime.now().isoformat()
    return (
        f"# ---------------------------------------------------------------------------\n"
        f"# IMPORTED COMPONENT — Provenance Record {record_id}\n"
        f"# Source: {repo_url} @ {commit_sha[:12]}\n"
        f"# File:   {file_path}\n"
        f"# License: {license_spdx}\n"
        f"# Fetched: {timestamp}\n"
        f"# Component: {component_name}\n"
        f"# Trust scores: scout={scout_trust:.2f} scanner={scanner_trust:.2f}\n"
        f"# ---------------------------------------------------------------------------\n"
    )
