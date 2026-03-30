"""
experiment_runner.py — Robust experiment runner for Asimov's Mind research.

Runs ML training experiments in two phases:
  Phase 1 (Isolated):    Each experiment tests ONE parameter change vs baseline.
  Phase 2 (Cumulative):  Best changes are applied together, per condition.

Fixes over the original run_single.py:
  - No git branch switching during experiments (caches baseline train.py in memory)
  - Process-tree killing via PID, not blanket taskkill
  - Proper error handling and atomic result logging
  - Per-experiment log files

Usage:
  python governed/experiment_runner.py baseline
  python governed/experiment_runner.py isolated
  python governed/experiment_runner.py cumulative
  python governed/experiment_runner.py all
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
LOGS_DIR = RESULTS_DIR / "logs"
GOVERNANCE_PATH = Path(__file__).parent / "governance.json"
TRAIN_PY = REPO_ROOT / "train.py"
BASELINE_COMMIT = "649e8c6"

# Find uv executable
UV_EXE = None
_uv_env = os.environ.get("UV_EXE", "")
_uv_candidates = []
if _uv_env:
    _uv_candidates.append(Path(_uv_env))
_uv_candidates.extend([
    Path.home() / "AppData" / "Roaming" / "Python" / "Python314" / "Scripts" / "uv.exe",
    Path.home() / "AppData" / "Roaming" / "Python" / "Python313" / "Scripts" / "uv.exe",
    Path.home() / ".local" / "bin" / "uv",
    Path.home() / ".cargo" / "bin" / "uv",
])
for candidate in _uv_candidates:
    if candidate.is_file():
        UV_EXE = str(candidate)
        break
if UV_EXE is None:
    # Fall back to PATH
    UV_EXE = "uv"

# ---------------------------------------------------------------------------
# Baseline hyperparameters (from commit 649e8c6)
# ---------------------------------------------------------------------------

BASELINE_PARAMS = {
    "ASPECT_RATIO": "64",
    "HEAD_DIM": "128",
    "WINDOW_PATTERN": '"SSSL"',
    "TOTAL_BATCH_SIZE": "2**19",
    "EMBEDDING_LR": "0.6",
    "UNEMBEDDING_LR": "0.004",
    "MATRIX_LR": "0.04",
    "SCALAR_LR": "0.5",
    "WEIGHT_DECAY": "0.2",
    "ADAM_BETAS": "(0.8, 0.95)",
    "WARMUP_RATIO": "0.0",
    "WARMDOWN_RATIO": "0.5",
    "FINAL_LR_FRAC": "0.0",
    "DEPTH": "6",
    "DEVICE_BATCH_SIZE": "16",
}

# ---------------------------------------------------------------------------
# Experiment definitions (10 isolated experiments)
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {"id": 1,  "desc": "Matrix LR 0.06",      "params": {"MATRIX_LR": "0.06"},     "zone": "hypertuner"},
    {"id": 2,  "desc": "Matrix LR 0.02",      "params": {"MATRIX_LR": "0.02"},     "zone": "hypertuner"},
    {"id": 3,  "desc": "Weight decay 0.1",     "params": {"WEIGHT_DECAY": "0.1"},   "zone": "regularizer"},
    {"id": 4,  "desc": "No weight decay",      "params": {"WEIGHT_DECAY": "0.0"},   "zone": "regularizer"},
    {"id": 5,  "desc": "Warmup 10pct",         "params": {"WARMUP_RATIO": "0.1"},   "zone": "regularizer"},
    {"id": 6,  "desc": "Warmdown 70pct",       "params": {"WARMDOWN_RATIO": "0.7"}, "zone": "regularizer"},
    {"id": 7,  "desc": "Embedding LR 1.0",     "params": {"EMBEDDING_LR": "1.0"},   "zone": "hypertuner"},
    {"id": 8,  "desc": "Embedding LR 0.3",     "params": {"EMBEDDING_LR": "0.3"},   "zone": "hypertuner"},
    {"id": 9,  "desc": "Scalar LR 1.0",        "params": {"SCALAR_LR": "1.0"},      "zone": "regularizer"},
    {"id": 10, "desc": "Narrow aspect ratio 48","params": {"ASPECT_RATIO": "48"},    "zone": "architect"},
]

# Cumulative application order per condition (reconstructed from git history)
# Each entry: list of param dicts to apply ON TOP of baseline, in order
CUMULATIVE_ORDERS = {
    "A": [
        # Condition A kept: warmup, warmdown, no weight decay, scalar LR
        {"WARMUP_RATIO": "0.1"},
        {"WARMDOWN_RATIO": "0.7"},
        {"WEIGHT_DECAY": "0.0"},
        {"SCALAR_LR": "1.0"},
    ],
    "B": [
        # Condition B kept: matrix LR, weight decay, warmup, embed LR up, embed LR down, no decay, scalar LR
        {"MATRIX_LR": "0.02"},
        {"WEIGHT_DECAY": "0.1"},
        {"WARMUP_RATIO": "0.1"},
        {"EMBEDDING_LR": "1.0"},
        {"EMBEDDING_LR": "0.3"},
        {"WEIGHT_DECAY": "0.0"},
        {"SCALAR_LR": "1.0"},
    ],
    "C": [
        # Condition C kept: narrow AR, matrix LR up, weight decay, embed LR, matrix LR down, warmdown
        {"ASPECT_RATIO": "48"},
        {"MATRIX_LR": "0.06"},
        {"WEIGHT_DECAY": "0.1"},
        {"EMBEDDING_LR": "1.0"},
        {"MATRIX_LR": "0.02"},
        {"WARMDOWN_RATIO": "0.7"},
    ],
}

# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

def load_governance():
    """Load governance bounds from governance.json."""
    if not GOVERNANCE_PATH.exists():
        return None
    with open(GOVERNANCE_PATH) as f:
        gov = json.load(f)
    return gov["laws"]["second"]["enforcement"]["editable_zones"]["hyperparameters"]["parameters"]


def check_governance(param, value, bounds):
    """Check if a parameter value is within governance bounds. Returns (ok, reason)."""
    if bounds is None:
        return True, ""
    if param not in bounds:
        return True, ""  # no bound defined = allowed
    spec = bounds[param]
    try:
        numeric = float(value)
    except ValueError:
        return True, ""  # non-numeric params (e.g., WINDOW_PATTERN) aren't bounded
    if numeric < spec["min"]:
        return False, f"{param}={value} below min {spec['min']}"
    if numeric > spec["max"]:
        return False, f"{param}={value} above max {spec['max']}"
    return True, ""


def check_zone(param, zone, governance_json_path=None):
    """Check if a parameter belongs to the specified specialist zone (for condition C)."""
    # Zone assignments based on governance.json multi_agent_zones
    zone_params = {
        "architect":   {"ASPECT_RATIO", "HEAD_DIM", "DEPTH", "WINDOW_PATTERN"},
        "hypertuner":  {"MATRIX_LR", "EMBEDDING_LR", "UNEMBEDDING_LR", "SCALAR_LR",
                        "TOTAL_BATCH_SIZE", "DEVICE_BATCH_SIZE"},
        "regularizer": {"WEIGHT_DECAY", "WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC",
                        "ADAM_BETAS"},
    }
    allowed = zone_params.get(zone, set())
    if param in allowed:
        return True, ""
    return False, f"{param} not in {zone} zone (allowed: {allowed})"


# ---------------------------------------------------------------------------
# train.py manipulation
# ---------------------------------------------------------------------------

def get_baseline_train_py():
    """Get the baseline train.py content from git commit."""
    result = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:train.py"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=10,
    )
    if result.returncode != 0:
        print(f"ERROR: Could not read baseline train.py from {BASELINE_COMMIT}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout


def apply_params(content, params):
    """Apply parameter changes to train.py content string. Returns modified content."""
    lines = content.splitlines(keepends=True)
    new_lines = []
    for line in lines:
        replaced = False
        for param, value in params.items():
            # Match lines like: PARAM = value  # optional comment
            pattern = rf'^(\s*){re.escape(param)}\s*=\s*'
            if re.match(pattern, line):
                indent = line[:len(line) - len(line.lstrip())]
                # Preserve comment if present
                comment = ""
                # Find comment that's not inside a string
                stripped = line.rstrip()
                hash_pos = stripped.find("#")
                if hash_pos > 0:
                    before_hash = stripped[:hash_pos]
                    # Simple heuristic: if quotes are balanced before #, it's a real comment
                    if before_hash.count('"') % 2 == 0 and before_hash.count("'") % 2 == 0:
                        comment = "  " + stripped[hash_pos:]
                new_lines.append(f"{indent}{param} = {value}{comment}\n")
                replaced = True
                break
        if not replaced:
            new_lines.append(line)
    return "".join(new_lines)


# ---------------------------------------------------------------------------
# Training execution
# ---------------------------------------------------------------------------

def run_training(log_path, timeout=900):
    """Run train.py and return (exit_code, val_bpb, peak_vram_mb, elapsed)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                [UV_EXE, "run", "train.py"],
                cwd=str(REPO_ROOT),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                # Windows: create a new process group so we can kill the tree
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            exit_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout}s — killing process tree (PID {proc.pid})")
        _kill_process_tree(proc.pid)
        try:
            proc.wait(timeout=15)
        except Exception:
            pass
        return 1, 0.0, 0.0, timeout
    except Exception as e:
        print(f"    ERROR launching training: {e}")
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
        return 1, 0.0, 0.0, 0.0

    elapsed = 0.0
    val_bpb = 0.0
    peak_vram = 0.0

    try:
        stdout = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        stdout = ""

    for line in stdout.split("\n"):
        line = line.strip()
        if line.startswith("val_bpb:"):
            try:
                val_bpb = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("peak_vram_mb:"):
            try:
                peak_vram = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("total_seconds:"):
            try:
                elapsed = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("training_seconds:"):
            try:
                training_secs = float(line.split(":")[1].strip())
            except ValueError:
                pass

    return exit_code, val_bpb, peak_vram, elapsed


def _kill_process_tree(pid):
    """Kill a process and all its children. Windows-safe."""
    import platform
    if platform.system() == "Windows":
        # /T = kill tree, /F = force, /PID = specific process
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True, timeout=15,
        )
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------

TSV_HEADER = "phase\texperiment\tval_bpb\tpeak_vram_mb\tstatus\tcondition\tdescription\ttimestamp\n"


def log_result(tsv_path, phase, exp_id, val_bpb, peak_vram, status, condition, description):
    """Append a result row to the TSV file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not tsv_path.exists():
        tsv_path.write_text(TSV_HEADER, encoding="utf-8")
    with open(tsv_path, "a", encoding="utf-8") as f:
        f.write(f"{phase}\t{exp_id}\t{val_bpb:.6f}\t{peak_vram:.1f}\t{status}\t"
                f"{condition}\t{description}\t{datetime.now().isoformat()}\n")


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_baseline(baseline_content):
    """Run baseline training and return val_bpb."""
    print("=" * 60)
    print("PHASE 0: BASELINE")
    print("=" * 60)

    # Write baseline train.py
    TRAIN_PY.write_text(baseline_content, encoding="utf-8")

    log_path = LOGS_DIR / "baseline.log"
    print(f"  Running baseline training (~5 min)...")
    exit_code, val_bpb, peak_vram, elapsed = run_training(log_path)

    tsv_path = RESULTS_DIR / "all_results.tsv"

    if exit_code != 0 or val_bpb == 0:
        print(f"  BASELINE CRASHED (exit_code={exit_code})")
        print(f"  Check log: {log_path}")
        # Show last 20 lines of log
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").strip().split("\n")
            print("  --- Last 20 lines of log ---")
            for line in lines[-20:]:
                print(f"    {line}")
        except Exception:
            pass
        log_result(tsv_path, "baseline", 0, 0.0, 0.0, "crash", "all", "baseline")
        return 0.0

    print(f"  BASELINE: val_bpb={val_bpb:.6f}  vram={peak_vram:.0f}MB  ({elapsed:.0f}s)")
    log_result(tsv_path, "baseline", 0, val_bpb, peak_vram, "keep", "all", "baseline")
    return val_bpb


def run_isolated_experiments(baseline_content, baseline_bpb):
    """Run all isolated experiments (each tests one change vs baseline)."""
    print()
    print("=" * 60)
    print("PHASE 1: ISOLATED EXPERIMENTS")
    print("=" * 60)

    governance_bounds = load_governance()
    tsv_path = RESULTS_DIR / "all_results.tsv"
    results = []

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        desc = exp["desc"]
        params = exp["params"]
        zone = exp["zone"]

        print(f"\n  --- Experiment {exp_id}/10: {desc} ---")

        # Governance check (for reporting — all conditions run the same experiments)
        gov_ok = True
        zone_ok = True
        for param, value in params.items():
            ok, reason = check_governance(param, value, governance_bounds)
            if not ok:
                gov_ok = False
                print(f"    [GOV] BLOCKED: {reason}")
            ok, reason = check_zone(param, zone)
            if not ok:
                zone_ok = False
                print(f"    [ZONE] Outside zone: {reason}")

        # Apply params to baseline and write
        modified = apply_params(baseline_content, params)
        TRAIN_PY.write_text(modified, encoding="utf-8")

        # Run training
        log_path = LOGS_DIR / f"isolated_{exp_id:02d}.log"
        print(f"    Training... (~5 min)")
        exit_code, val_bpb, peak_vram, elapsed = run_training(log_path)

        if exit_code != 0 or val_bpb == 0:
            print(f"    CRASH (exit_code={exit_code}, {elapsed:.0f}s)")
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").strip().split("\n")
                for line in lines[-5:]:
                    print(f"      {line}")
            except Exception:
                pass
            status = "crash"
            delta = 0.0
        else:
            delta = baseline_bpb - val_bpb
            if val_bpb < baseline_bpb:
                status = "improved"
                print(f"    IMPROVED: val_bpb={val_bpb:.6f} (delta={delta:+.6f})  vram={peak_vram:.0f}MB  ({elapsed:.0f}s)")
            else:
                status = "no_improvement"
                print(f"    NO IMPROVEMENT: val_bpb={val_bpb:.6f} (delta={delta:+.6f})  vram={peak_vram:.0f}MB  ({elapsed:.0f}s)")

        # Log for all three conditions (same experiment, different governance annotations)
        for cond in ["A", "B", "C"]:
            cond_status = status
            if cond == "B" and not gov_ok:
                cond_status = "governance_blocked"
            elif cond == "C" and (not gov_ok or not zone_ok):
                cond_status = "governance_blocked"
            log_result(tsv_path, "isolated", exp_id, val_bpb, peak_vram, cond_status, cond, desc)

        results.append({
            "id": exp_id, "desc": desc, "params": params, "zone": zone,
            "val_bpb": val_bpb, "peak_vram": peak_vram, "status": status,
            "delta": delta, "gov_ok": gov_ok, "zone_ok": zone_ok,
        })

        # Restore baseline before next experiment
        TRAIN_PY.write_text(baseline_content, encoding="utf-8")

    # Summary
    print("\n  --- Isolated Results Summary ---")
    improved = [r for r in results if r["status"] == "improved"]
    improved.sort(key=lambda r: r["delta"], reverse=True)
    print(f"  {len(improved)}/{len(results)} experiments improved over baseline ({baseline_bpb:.6f})")
    for r in improved:
        print(f"    #{r['id']:2d}  delta={r['delta']:+.6f}  bpb={r['val_bpb']:.6f}  {r['desc']}")

    return results


def run_cumulative_experiments(baseline_content, baseline_bpb, isolated_results):
    """Run cumulative experiments for each condition."""
    print()
    print("=" * 60)
    print("PHASE 2: CUMULATIVE EXPERIMENTS")
    print("=" * 60)

    tsv_path = RESULTS_DIR / "all_results.tsv"
    governance_bounds = load_governance()

    for condition, changes in CUMULATIVE_ORDERS.items():
        print(f"\n  === Condition {condition} ===")

        # Build cumulative param dict (each change overrides previous)
        cumulative_params = {}
        for i, change in enumerate(changes, 1):
            cumulative_params.update(change)
            desc_parts = [f"{k}={v}" for k, v in cumulative_params.items()
                          if cumulative_params[k] != BASELINE_PARAMS.get(k)]
            desc = f"Cumulative step {i}: {', '.join(desc_parts)}"

            print(f"\n    Step {i}/{len(changes)}: +{change}")

            # Governance check for condition B and C
            skip = False
            if condition in ("B", "C"):
                for param, value in change.items():
                    ok, reason = check_governance(param, value, governance_bounds)
                    if not ok:
                        print(f"      [GOV] BLOCKED: {reason} — skipping")
                        log_result(tsv_path, "cumulative", i, 0.0, 0.0,
                                   "governance_blocked", condition, desc)
                        skip = True

            if skip:
                continue

            # Apply all cumulative params
            modified = apply_params(baseline_content, cumulative_params)
            TRAIN_PY.write_text(modified, encoding="utf-8")

            log_path = LOGS_DIR / f"cumulative_{condition}_{i:02d}.log"
            print(f"      Training... (~5 min)")
            exit_code, val_bpb, peak_vram, elapsed = run_training(log_path)

            if exit_code != 0 or val_bpb == 0:
                print(f"      CRASH ({elapsed:.0f}s)")
                log_result(tsv_path, "cumulative", i, 0.0, 0.0, "crash", condition, desc)
                # Revert last change since it crashed
                for k in change:
                    if k in cumulative_params:
                        del cumulative_params[k]
            else:
                delta = baseline_bpb - val_bpb
                status = "improved" if val_bpb < baseline_bpb else "no_improvement"
                print(f"      val_bpb={val_bpb:.6f} (vs baseline {delta:+.6f})  vram={peak_vram:.0f}MB  ({elapsed:.0f}s)")
                log_result(tsv_path, "cumulative", i, val_bpb, peak_vram, status, condition, desc)

        # Restore baseline
        TRAIN_PY.write_text(baseline_content, encoding="utf-8")


def run_optimal_cumulative(baseline_content, baseline_bpb, isolated_results):
    """Phase 3: Apply isolated improvements in optimal order (best delta first)."""
    print()
    print("=" * 60)
    print("PHASE 3: OPTIMAL CUMULATIVE (best improvements first)")
    print("=" * 60)

    tsv_path = RESULTS_DIR / "all_results.tsv"

    # Filter to experiments that actually improved, sort by delta descending
    improved = [r for r in isolated_results if r["status"] == "improved"]
    improved.sort(key=lambda r: r["delta"], reverse=True)

    if not improved:
        print("  No isolated experiments improved over baseline. Skipping.")
        return

    print(f"  Stacking {len(improved)} improvements in order of effectiveness:")
    for r in improved:
        print(f"    delta={r['delta']:+.6f}  {r['desc']}")

    cumulative_params = {}
    for i, result in enumerate(improved, 1):
        cumulative_params.update(result["params"])
        desc_parts = [f"{k}={v}" for k, v in cumulative_params.items()
                      if cumulative_params[k] != BASELINE_PARAMS.get(k)]
        desc = f"Optimal step {i}: +{result['desc']} ({', '.join(desc_parts)})"

        print(f"\n    Step {i}/{len(improved)}: +{result['desc']}")

        modified = apply_params(baseline_content, cumulative_params)
        TRAIN_PY.write_text(modified, encoding="utf-8")

        log_path = LOGS_DIR / f"optimal_{i:02d}.log"
        print(f"      Training... (~5 min)")
        exit_code, val_bpb, peak_vram, elapsed = run_training(log_path)

        if exit_code != 0 or val_bpb == 0:
            print(f"      CRASH ({elapsed:.0f}s)")
            log_result(tsv_path, "optimal", i, 0.0, 0.0, "crash", "optimal", desc)
            # Revert last change
            for k in result["params"]:
                if k in cumulative_params:
                    del cumulative_params[k]
        else:
            delta = baseline_bpb - val_bpb
            status = "improved" if val_bpb < baseline_bpb else "no_improvement"
            print(f"      val_bpb={val_bpb:.6f} (vs baseline {delta:+.6f})  vram={peak_vram:.0f}MB  ({elapsed:.0f}s)")
            log_result(tsv_path, "optimal", i, val_bpb, peak_vram, status, "optimal", desc)

    # Restore baseline
    TRAIN_PY.write_text(baseline_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode not in ("baseline", "isolated", "cumulative", "optimal", "all"):
        print(f"Usage: python {sys.argv[0]} [baseline|isolated|cumulative|optimal|all]")
        sys.exit(1)

    print("Asimov's Mind — Experiment Runner v2")
    print(f"Mode: {mode}")
    print(f"Repo: {REPO_ROOT}")
    print(f"UV:   {UV_EXE}")
    print()

    # Ensure directories exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Cache baseline train.py
    baseline_content = get_baseline_train_py()
    print(f"Loaded baseline train.py from commit {BASELINE_COMMIT} ({len(baseline_content)} chars)")

    # Save current train.py so we can restore it when done
    original_content = TRAIN_PY.read_text(encoding="utf-8") if TRAIN_PY.exists() else None

    try:
        if mode in ("baseline", "all"):
            baseline_bpb = run_baseline(baseline_content)
            if baseline_bpb == 0:
                print("\nBaseline failed. Fix the training setup before proceeding.")
                sys.exit(1)

        if mode in ("isolated", "all"):
            # Read baseline_bpb from results if not just computed
            if mode == "isolated":
                baseline_bpb = _read_baseline_bpb()
                if baseline_bpb == 0:
                    print("No baseline result found. Run 'baseline' first.")
                    sys.exit(1)
            isolated_results = run_isolated_experiments(baseline_content, baseline_bpb)

        if mode in ("cumulative", "all"):
            if mode == "cumulative":
                baseline_bpb = _read_baseline_bpb()
                if baseline_bpb == 0:
                    print("No baseline result found. Run 'baseline' first.")
                    sys.exit(1)
                isolated_results = []
            run_cumulative_experiments(baseline_content, baseline_bpb, isolated_results)

        if mode in ("optimal", "all"):
            if mode == "optimal":
                baseline_bpb = _read_baseline_bpb()
                if baseline_bpb == 0:
                    print("No baseline result found. Run 'baseline' first.")
                    sys.exit(1)
                isolated_results = _read_isolated_results()
            run_optimal_cumulative(baseline_content, baseline_bpb, isolated_results)

        print("\n" + "=" * 60)
        print("ALL PHASES COMPLETE")
        print(f"Results: {RESULTS_DIR / 'all_results.tsv'}")
        print(f"Logs:    {LOGS_DIR}")
        print("=" * 60)

    finally:
        # Restore original train.py
        if original_content is not None:
            TRAIN_PY.write_text(original_content, encoding="utf-8")
            print("\nRestored original train.py")


def _read_isolated_results():
    """Reconstruct isolated results from the TSV (for running optimal phase standalone)."""
    tsv_path = RESULTS_DIR / "all_results.tsv"
    if not tsv_path.exists():
        return []
    # Build a lookup from experiment definitions
    exp_lookup = {e["id"]: e for e in EXPERIMENTS}
    results = []
    baseline_bpb = _read_baseline_bpb()
    seen = set()
    for line in tsv_path.read_text(encoding="utf-8").strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 6 and parts[0] == "isolated" and parts[5] == "A":
            # Only read condition A rows (same data for all conditions)
            exp_id = int(parts[1])
            if exp_id in seen:
                continue
            seen.add(exp_id)
            val_bpb = float(parts[2])
            peak_vram = float(parts[3])
            status = parts[4]
            exp_def = exp_lookup.get(exp_id, {})
            delta = baseline_bpb - val_bpb if val_bpb > 0 else 0.0
            results.append({
                "id": exp_id,
                "desc": exp_def.get("desc", parts[6] if len(parts) > 6 else ""),
                "params": exp_def.get("params", {}),
                "zone": exp_def.get("zone", ""),
                "val_bpb": val_bpb, "peak_vram": peak_vram,
                "status": "improved" if delta > 0 else status,
                "delta": delta,
                "gov_ok": True, "zone_ok": True,
            })
    return results


def _read_baseline_bpb():
    """Read baseline val_bpb from existing results."""
    tsv_path = RESULTS_DIR / "all_results.tsv"
    if not tsv_path.exists():
        return 0.0
    for line in tsv_path.read_text(encoding="utf-8").strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 5 and parts[0] == "baseline" and parts[4] == "keep":
            try:
                return float(parts[2])
            except ValueError:
                pass
    return 0.0


if __name__ == "__main__":
    main()
