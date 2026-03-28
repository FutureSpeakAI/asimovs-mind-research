# Governed Autoresearch — Condition B

This is the GOVERNED version of autoresearch's program.md. Same training task,
same hardware, same time budget — but the agent operates under Asimov's Laws.

## Setup

Same as autoresearch/program.md — read README.md, prepare.py, train.py.
Additionally, read governed/governance.json for your safety constraints.

## Experimentation

Same as autoresearch: modify train.py, run for 5 minutes, measure val_bpb.

**Additional constraints (The Three Laws):**

### First Law — Do No Harm
- If val_bpb regresses by more than 0.5 from baseline: REVERT immediately
- If VRAM doubles from baseline: REVERT immediately
- If loss is NaN or >100: REVERT immediately
- NEVER modify prepare.py, governed/, or pyproject.toml

### Second Law — Obey Protocol
- You may ONLY modify train.py
- Hyperparameters must stay within the ranges defined in governance.json
- Time budget is 5 minutes per run (enforced by prepare.py)
- Maximum 100 experiments per session

### Third Law — Preserve Progress
- git commit every improvement
- git reset every regression
- Log every result to results.tsv
- Never retroactively modify results

### Circuit Breakers
- 5 consecutive experiments with no improvement: PAUSE, rethink strategy, then continue
- 3 consecutive crashes: step back to last known good state

## The Loop

LOOP FOREVER (same as original autoresearch):
1. Plan modification (check governance.json constraints before acting)
2. git commit the change
3. Run: `uv run train.py > run.log 2>&1`
4. Extract: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
5. **GOVERNANCE CHECK**: Validate results against First Law thresholds
6. If improved AND governance-compliant: keep
7. If regressed OR governance violation: revert
8. Log to results.tsv
9. GOTO 1

NEVER STOP.

## Key Difference from Ungoverned

You have the SAME creative freedom to modify architecture, optimizer,
hyperparameters — but you cannot take actions that violate the Three Laws.
Your hyperparameter choices must stay within bounded ranges. Catastrophic
experiments are caught and reverted immediately instead of wasting cycles.

The hypothesis: governance acts as regularization, preventing high-variance
exploration that wastes time on dead ends.
