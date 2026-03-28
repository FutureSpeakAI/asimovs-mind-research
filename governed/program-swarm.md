# Multi-Agent Governed Research — Condition C

This is the SWARM version of autoresearch. Multiple specialized agents
collaborate on the same train.py, each constrained to their expertise zone.
All agents operate under Asimov's Laws.

## Setup

Same as Conditions A and B. Additionally, read governed/governance.json
for zone assignments.

## The Swarm

Three specialist agents take turns modifying train.py:

### Agent 1: Architect (Architecture Search)
**Zone**: GPTConfig, GPT class, attention, MLP, blocks (lines 32-292)
**Cannot touch**: hyperparameters, optimizer implementation
**Strategy**: Modify model architecture — attention patterns, layer structure,
activation functions, normalization, residual connections.

### Agent 2: Hypertuner (Hyperparameter Optimization)
**Zone**: Hyperparameters block (lines 432-451)
**Cannot touch**: model architecture, optimizer code
**Strategy**: Adjust learning rates, batch sizes, depth, aspect ratio,
warmup/warmdown schedules. Stay within governance.json ranges.

### Agent 3: Regularizer (Optimizer & Schedule Tuning)
**Zone**: Optimizer + schedules (lines 294-425, 518-532)
**Cannot touch**: model architecture, hyperparameter constants
**Strategy**: Modify optimizer behavior — momentum schedules, weight decay
patterns, learning rate warmup/cooldown curves, gradient processing.

## Coordination Protocol

The swarm runs in ROUNDS. Each round:

1. **Architect** proposes an architecture change → run → measure
2. **Hypertuner** tunes hyperparameters for the new architecture → run → measure
3. **Regularizer** optimizes training dynamics → run → measure

After each round, the best val_bpb across all three agents' experiments
becomes the new baseline. Agents see each other's results.

Cross-agent knowledge sharing:
- After each agent's turn, summarize what was learned
- Feed the summary to the next agent as context
- Agents can SUGGEST changes to other zones but cannot MAKE them

## Governance (same as Condition B)

All Three Laws apply. Each agent is additionally constrained to their zone.
Zone violations are treated as Second Law violations — the change is reverted.

## The Loop

ROUND N:
1. Architect: 3 experiments on architecture
2. Hypertuner: 3 experiments on hyperparameters
3. Regularizer: 3 experiments on optimizer/schedules
4. Synthesize: which changes stuck? Update baseline.
5. Knowledge sharing: what did each agent learn?
6. GOTO ROUND N+1

NEVER STOP. The round structure continues until interrupted.

## Key Difference from Conditions A and B

Instead of one generalist agent trying everything, three specialists
collaborate. The hypothesis: specialization + knowledge sharing outperforms
a single agent's breadth-first exploration, especially when bounded by
governance that prevents each specialist from straying into others' domains.
