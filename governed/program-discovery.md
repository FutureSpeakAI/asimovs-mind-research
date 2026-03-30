# autoresearch — Discovery Mode

This extends `program-governed.md` with the ability to **discover, adapt, and test ML code from GitHub**. All three Laws from `program-governed.md` still apply in full. Discovery mode adds new capabilities and new constraints.

## Prerequisites

Before entering discovery mode:
1. `governed/governance_v2.json` must exist with `discovery_enabled: true`
2. A baseline val_bpb must be established (run unmodified `train.py` first)
3. At least 5 pure hyperparameter experiments should have run first (exhaust cheap search space before expensive discovery)
4. You have read `train.py`, `prepare.py`, and `governed/governance.json`

## The Discovery Pipeline

Discovery follows a strict pipeline. **You must not skip steps.**

### Step 1: Scout

Decide what to search for based on:
- Weaknesses observed in recent experiments (e.g., if LR tuning hasn't helped, look for better optimizers)
- Zones that haven't been explored (e.g., no attention variants tried yet)
- Specific techniques mentioned in the codebase comments

Use the scout module:
```python
from governed.discovery.scout import run_scout
report = run_scout(strategies=["optimizer"], min_relevance=0.35, min_trust=0.5)
```

Read the ScoutReport. Pick the top candidate based on relevance AND trust scores.

### Step 2: Fetch and Scan

Fetch the candidate's source file:
```python
from governed.discovery.scout import fetch_file_content, fetch_file_listing
files = fetch_file_listing(candidate.repo_full_name, "main")
content = fetch_file_content(candidate.repo_full_name, "path/to/file.py")
```

Run safety analysis BEFORE any adaptation:
```python
from governed.discovery.safety_scanner import scan_source
report = scan_source(content, candidate.repo_full_name, "path/to/file.py")
```

**If verdict is HARD_BLOCK: STOP. Move to next candidate.** Do not attempt to work around Tier 1 findings.
**If verdict is SOFT_BLOCK: proceed only if scanner_trust_score >= 0.7.**
**If verdict is PASS: proceed.**

### Step 3: Adapt

Extract and transform the target component:
```python
from governed.discovery.adapter import adapt, build_component_map

# See what's available
components = build_component_map(content)
for name, info in components.items():
    print(f"  {info.kind}: {name} (lines {info.line_start}-{info.line_end})")

# Extract and adapt
result = adapt(content, "path/to/file.py", candidate.repo_full_name,
               target_component="SOAPOptimizer", target_zone="optimizer")
```

**If `result.can_proceed` is False: STOP.** Read `result.reason` and move to next candidate.

### Step 4: Post-Adaptation Safety Rescan

The adapted code MUST be re-scanned:
```python
rescan = scan_source(result.adapted_code, candidate.repo_full_name, "adapted")
```

**If rescan verdict is not PASS: STOP.** The adaptation may have introduced issues.

### Step 5: Record Provenance

Before inserting ANY code into train.py:
```python
from governed.discovery.provenance import (
    generate_record_id, write_candidate_record,
    check_license_compliance, format_attribution_comment
)

record_id = generate_record_id()

# License check
ok, reason = check_license_compliance(candidate.license_spdx)
if not ok:
    print(f"LICENSE BLOCKED: {reason}")
    # STOP — do not proceed

write_candidate_record(
    record_id=record_id,
    repo_full_name=candidate.repo_full_name,
    repo_url=candidate.repo_url,
    license_spdx=candidate.license_spdx,
    commit_sha="main",  # or the specific SHA
    file_path="path/to/file.py",
    source_content=content,
    extracted_component="SOAPOptimizer",
    scout_relevance=candidate.relevance_score,
    scout_trust=candidate.trust_score,
    scanner_verdict=rescan.verdict,
    scanner_trust=rescan.scanner_trust_score,
    target_zone="optimizer",
    renames=result.renames_applied,
)
```

### Step 6: Insert and Output Review Block

Insert the adapted code into the target zone of `train.py`. Prepend the attribution comment:
```python
attribution = format_attribution_comment(
    record_id, candidate.repo_url, "main",
    "path/to/file.py", candidate.license_spdx,
    "SOAPOptimizer", candidate.trust_score, rescan.scanner_trust_score
)
```

**Before inserting, output this review block:**

```
DISCOVERY IMPORT REVIEW
=======================
Repo: {repo_url}
License: {license}
Component: {component_name} ({line_count} lines)
Target zone: {zone}
Scout trust: {score} | Scanner trust: {score}
Quarantine: {yes/no}

ADAPTED CODE PREVIEW:
(first 20 lines)

Proceeding with import.
```

### Step 7: Run Experiment

Follow the normal autoresearch loop:
1. `git add train.py`
2. `git commit -m "discovery: import {component} from {repo}"`
3. `uv run train.py > run.log 2>&1`
4. `grep "^val_bpb:\|^peak_vram_mb:" run.log`
5. Evaluate result

### Step 8: Record Outcome

```python
from governed.discovery.provenance import write_outcome_record

write_outcome_record(
    record_id=record_id,
    repo_full_name=candidate.repo_full_name,
    val_bpb=measured_bpb,
    baseline_val_bpb=baseline_bpb,
    peak_vram_mb=measured_vram,
    decision="kept" if improved else "reverted",
)
```

If the experiment crashed or regressed, `git reset --hard HEAD~1` and record `decision: "reverted"`.

## Constraints

### Import Limits
- Maximum **3 imports per session** (not attempts — actual insertions into train.py)
- Maximum **200 lines per import** (enforced by adapter)
- Do not scout more than once per 10 experiments

### Trust Tiers

| Tier | Min Scout Trust | Min Scanner Trust | Max Lines | Quarantine |
|------|:-:|:-:|:-:|:-:|
| 1: Verified | 0.85 | 0.70 | 200 | No |
| 2: Community | 0.65 | 0.75 | 150 | 1 experiment |
| 3: Experimental | 0.50 | 0.85 | 100 | 2 experiments |

During quarantine:
- VRAM regression threshold tightens from 2.0x to 1.25x
- val_bpb regression threshold tightens from 0.5 to 0.1
- Any crash = immediate revert (no fix attempts)

### What You CANNOT Do
- Install new packages or modify `pyproject.toml`
- Modify any file in `governed/discovery/` — these are read-only to you
- Modify `provenance_log.jsonl` retroactively — it is append-only
- Import code with no license or with blocked dependencies
- Bypass the safety scanner — every piece of external code must be scanned
- Dynamically load discovered code — it runs only through `uv run train.py`

### Circuit Breakers
- 2 bad imports (crash or revert) in a session: disable discovery for the rest of the session
- 3 consecutive failed scout strategies: return to hyperparameter exploration
- Any modification to `governed/` files: HALT

## Decision Framework

**When to scout:**
1. Hyperparameter search has stalled (5+ experiments with no improvement)
2. A specific weakness has been identified (e.g., "this optimizer isn't converging well")
3. A known technique exists that hasn't been tried (e.g., "QK-norm for attention")

**When NOT to scout:**
1. Fewer than 5 experiments have run in this session
2. Last 3 experiments all crashed (fix stability first)
3. Already at 3 imports this session (limit reached)

**What to search for (priority order):**
1. Techniques that address observed weaknesses
2. Techniques for unexplored zones (architecture > optimizer > schedules)
3. Known-good techniques from recent ML papers

## Integration with the Three Laws

Discovery mode does not suspend any existing Law:

**First Law (Do No Harm):** All existing circuit breakers apply. Discovery adds: Tier 1 hard blocks, no-license blocks, quarantine thresholds, post-adaptation rescans.

**Second Law (Obey Protocol):** The only editable file is still `train.py`. Discovery adds: the mandatory pipeline order, forbidden modifications to `governed/discovery/`, append-only provenance log.

**Third Law (Preserve Progress):** Every improvement is committed, every regression reverted. Discovery adds: three-stage provenance records, attribution comments, license compliance tracking.
