"""
run_experiment.py — Asimov's Mind ML Research Experiment Runner

Orchestrates the three experimental conditions and generates comparison data:
  A) Ungoverned (original autoresearch — single agent, no constraints)
  B) Governed (single agent + Asimov's Laws)
  C) Swarm (3 specialist agents + Asimov's Laws)

Each condition runs on the same hardware, same data, same time budget.
Results are logged to separate TSV files and compared.

Usage:
  python governed/run_experiment.py --condition A  # Run ungoverned baseline
  python governed/run_experiment.py --condition B  # Run governed single-agent
  python governed/run_experiment.py --condition C  # Run governed swarm
  python governed/run_experiment.py --compare      # Compare all conditions
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
GOVERNED_DIR = Path(__file__).parent
RESULTS_DIR = GOVERNED_DIR / "results"
GOVERNANCE_PATH = GOVERNED_DIR / "governance.json"


def load_governance():
    """Load the governance constraints."""
    with open(GOVERNANCE_PATH) as f:
        return json.load(f)


def parse_run_log(log_path: str) -> dict:
    """Extract metrics from a training run log file."""
    metrics = {}
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("val_bpb:"):
                    metrics["val_bpb"] = float(line.split(":")[1].strip())
                elif line.startswith("peak_vram_mb:"):
                    metrics["peak_vram_mb"] = float(line.split(":")[1].strip())
                elif line.startswith("training_seconds:"):
                    metrics["training_seconds"] = float(line.split(":")[1].strip())
                elif line.startswith("mfu_percent:"):
                    metrics["mfu_percent"] = float(line.split(":")[1].strip())
                elif line.startswith("num_steps:"):
                    metrics["num_steps"] = int(line.split(":")[1].strip())
                elif line.startswith("num_params_M:"):
                    metrics["num_params_M"] = float(line.split(":")[1].strip())
                elif line.startswith("total_tokens_M:"):
                    metrics["total_tokens_M"] = float(line.split(":")[1].strip())
    except (FileNotFoundError, ValueError):
        pass
    return metrics


def check_governance(metrics: dict, baseline: dict, governance: dict) -> tuple[bool, str]:
    """
    Check if experiment results comply with Asimov's Laws.
    Returns (compliant, reason).
    """
    first_law = governance["laws"]["first"]["enforcement"]

    # NaN check
    if "val_bpb" not in metrics:
        return False, "No val_bpb in results (possible crash)"

    val_bpb = metrics["val_bpb"]

    # Catastrophic loss
    if val_bpb > first_law["catastrophic_loss_threshold"]:
        return False, f"val_bpb {val_bpb:.6f} exceeds catastrophic threshold {first_law['catastrophic_loss_threshold']}"

    # Regression check (if we have a baseline)
    if baseline and "val_bpb" in baseline:
        max_regression = first_law["max_val_bpb_regression"]
        if val_bpb > baseline["val_bpb"] + max_regression:
            return False, f"val_bpb regression {val_bpb - baseline['val_bpb']:.6f} exceeds max allowed {max_regression}"

    # VRAM check
    if baseline and "peak_vram_mb" in baseline and "peak_vram_mb" in metrics:
        max_ratio = first_law["max_vram_increase_ratio"]
        if metrics["peak_vram_mb"] > baseline["peak_vram_mb"] * max_ratio:
            return False, f"VRAM {metrics['peak_vram_mb']:.0f}MB exceeds {max_ratio}x baseline {baseline['peak_vram_mb']:.0f}MB"

    return True, "Compliant"


def run_training(cwd: str = None) -> tuple[dict, int]:
    """Run train.py and return (metrics, exit_code)."""
    log_path = os.path.join(cwd or str(REPO_ROOT), "run.log")

    print(f"  Running train.py (5-minute budget)...")
    t0 = time.time()

    result = subprocess.run(
        ["python3", "-m", "uv", "run", "train.py"],
        cwd=cwd or str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=700,  # 10min + margin
    )

    # Write log
    with open(log_path, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s (exit code {result.returncode})")

    metrics = parse_run_log(log_path)
    if metrics.get("val_bpb"):
        print(f"  val_bpb: {metrics['val_bpb']:.6f}")
    if metrics.get("peak_vram_mb"):
        print(f"  VRAM: {metrics['peak_vram_mb']:.0f} MB")

    return metrics, result.returncode


def log_result(condition: str, experiment: int, metrics: dict, status: str, description: str):
    """Append a result to the condition's TSV file."""
    RESULTS_DIR.mkdir(exist_ok=True)
    tsv_path = RESULTS_DIR / f"condition_{condition}.tsv"

    # Create header if new file
    if not tsv_path.exists():
        with open(tsv_path, "w") as f:
            f.write("experiment\tval_bpb\tpeak_vram_mb\tmfu_percent\tstatus\tdescription\ttimestamp\n")

    val_bpb = metrics.get("val_bpb", 0.0)
    vram = metrics.get("peak_vram_mb", 0.0)
    mfu = metrics.get("mfu_percent", 0.0)
    ts = datetime.now().isoformat()

    with open(tsv_path, "a") as f:
        f.write(f"{experiment}\t{val_bpb:.6f}\t{vram:.1f}\t{mfu:.2f}\t{status}\t{description}\t{ts}\n")


def compare_conditions():
    """Compare results across all three conditions."""
    print("\n" + "=" * 70)
    print("ASIMOV'S MIND — EXPERIMENTAL COMPARISON")
    print("=" * 70)

    for condition in ["A", "B", "C"]:
        tsv_path = RESULTS_DIR / f"condition_{condition}.tsv"
        if not tsv_path.exists():
            print(f"\nCondition {condition}: No data yet")
            continue

        with open(tsv_path) as f:
            lines = f.readlines()[1:]  # skip header

        if not lines:
            print(f"\nCondition {condition}: No experiments recorded")
            continue

        experiments = []
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                experiments.append({
                    "experiment": int(parts[0]),
                    "val_bpb": float(parts[1]),
                    "vram": float(parts[2]),
                    "mfu": float(parts[3]),
                    "status": parts[4],
                    "description": parts[5] if len(parts) > 5 else "",
                })

        kept = [e for e in experiments if e["status"] == "keep"]
        discarded = [e for e in experiments if e["status"] == "discard"]
        crashed = [e for e in experiments if e["status"] == "crash"]
        governed_reverts = [e for e in experiments if e["status"] == "governance_revert"]

        best = min(kept, key=lambda e: e["val_bpb"]) if kept else None
        baseline_bpb = experiments[0]["val_bpb"] if experiments else 0

        label = {"A": "Ungoverned", "B": "Governed", "C": "Swarm"}[condition]
        print(f"\n{'─' * 50}")
        print(f"Condition {condition}: {label}")
        print(f"{'─' * 50}")
        print(f"  Total experiments:    {len(experiments)}")
        print(f"  Kept:                 {len(kept)}")
        print(f"  Discarded:            {len(discarded)}")
        print(f"  Crashed:              {len(crashed)}")
        if governed_reverts:
            print(f"  Governance reverts:   {len(governed_reverts)}")
        print(f"  Baseline val_bpb:     {baseline_bpb:.6f}")
        if best:
            print(f"  Best val_bpb:         {best['val_bpb']:.6f}")
            print(f"  Total improvement:    {baseline_bpb - best['val_bpb']:.6f}")
            print(f"  Improvement rate:     {len(kept) / len(experiments) * 100:.1f}%")
        print(f"  Best VRAM:            {best['vram']:.0f} MB" if best else "")

    print(f"\n{'=' * 70}")
    print("KEY QUESTIONS:")
    print("  1. Does Condition B (governed) match Condition A (ungoverned) on val_bpb?")
    print("     → If yes: governance is 'free' safety (no performance cost)")
    print("  2. Does Condition C (swarm) outperform Condition A?")
    print("     → If yes: collaborative specialization beats solo generalism")
    print("  3. How many governance reverts occurred in B and C?")
    print("     → High count: governance is actively preventing harmful exploration")
    print("     → Low count: the agent naturally avoids harmful actions anyway")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Asimov's Mind ML Research Runner")
    parser.add_argument("--condition", choices=["A", "B", "C"], help="Run a specific condition")
    parser.add_argument("--compare", action="store_true", help="Compare all conditions")
    parser.add_argument("--baseline-only", action="store_true", help="Just run the baseline for a condition")
    args = parser.parse_args()

    if args.compare:
        compare_conditions()
        return

    if not args.condition:
        parser.print_help()
        return

    print(f"\n{'=' * 50}")
    label = {"A": "Ungoverned (baseline autoresearch)", "B": "Governed (single agent + laws)", "C": "Swarm (3 agents + laws)"}
    print(f"CONDITION {args.condition}: {label[args.condition]}")
    print(f"{'=' * 50}")

    if args.baseline_only:
        print("\nRunning baseline (no modifications)...")
        metrics, code = run_training(str(REPO_ROOT))
        if code == 0 and metrics:
            log_result(args.condition, 0, metrics, "keep", "baseline")
            print(f"\nBaseline recorded: val_bpb={metrics.get('val_bpb', 'N/A')}")
        else:
            print("\nBaseline run failed!")
        return

    # For conditions B and C, load governance
    governance = load_governance() if args.condition in ("B", "C") else None

    print(f"\nReady to run Condition {args.condition}.")
    print("Point your Claude Code agent at the appropriate program.md:")
    if args.condition == "A":
        print("  → program.md (original autoresearch)")
    elif args.condition == "B":
        print("  → governed/program-governed.md (governed single-agent)")
    elif args.condition == "C":
        print("  → governed/program-swarm.md (governed multi-agent)")
    print("\nThe agent will run autonomously. Results logged to:")
    print(f"  → governed/results/condition_{args.condition}.tsv")


if __name__ == "__main__":
    main()
