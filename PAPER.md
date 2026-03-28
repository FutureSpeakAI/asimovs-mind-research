# Governed Multi-Agent Optimization: Safety Constraints as Regularization in Autonomous ML Research

## Authors
FutureSpeak.AI

## Abstract

We investigate whether safety constraints improve or degrade the performance of autonomous AI agents conducting machine learning research. Building on Karpathy's autoresearch framework — where LLM agents autonomously modify training code, run experiments, and iteratively improve model performance — we introduce Asimov's Mind, a governance framework that constrains agent behavior through three laws: harm prevention (circuit breakers on catastrophic regression), directive compliance (bounded parameter ranges), and improvement preservation (git commit discipline). We evaluate three experimental conditions on identical hardware and training tasks: (A) ungoverned single-agent (the autoresearch baseline), (B) governed single-agent (same agent + Three Laws), and (C) governed multi-agent swarm (3 specialist agents + Three Laws). Our central hypothesis is that governance acts as a form of regularization — preventing high-variance exploration that wastes cycles on dead ends — and that multi-agent specialization enables more efficient search through the optimization space.

## 1. Introduction

Autonomous AI research agents represent a paradigm shift in ML experimentation. Karpathy's autoresearch (2025) demonstrated that LLM agents can conduct meaningful ML research overnight on a single GPU, iterating on model architecture and hyperparameters through a modify-execute-measure loop. This raises a fundamental question: **should autonomous research agents be constrained?**

Unconstrained agents can explore the full parameter space, including catastrophic regions (OOM, NaN loss, architectural dead ends). Constrained agents are bounded to "safe" regions but may miss important discoveries in unexplored territory. This mirrors the exploration-exploitation tradeoff in reinforcement learning, but with a safety dimension: catastrophic exploration wastes compute and time.

We propose that safety constraints function as **regularization for optimization processes** — analogous to how dropout, weight decay, and early stopping prevent overfitting in neural network training. Our governance framework (Asimov's Mind) constrains the agent's search space without modifying the training code itself.

## 2. Related Work

- **Autoresearch** (Karpathy, 2025): Single-agent autonomous ML research. The methodological foundation for this work.
- **Constitutional AI** (Anthropic, 2022): LLM behavior constrained by a constitution. We apply a similar concept to research agents.
- **Safe Reinforcement Learning** (Garcia & Fernandez, 2015): Constrained MDPs with safety boundaries. Our governance mirrors this in the research agent domain.
- **Multi-Agent Systems** (Wooldridge, 2009): Collaborative agent architectures. We apply specialization to ML research.
- **AutoML** (Hutter et al., 2019): Automated machine learning. Our work differs by using LLM reasoning rather than Bayesian optimization.

## 3. Methodology

### 3.1 The Asimov's Mind Governance Framework

Three laws constrain agent behavior:

**First Law (Do No Harm)**: Circuit breakers halt experiments that produce NaN loss, >100 loss, >2x VRAM increase, or >0.5 val_bpb regression from baseline. These prevent catastrophic exploration without limiting productive exploration.

**Second Law (Obey Protocol)**: Agents may only modify `train.py`. Hyperparameters must stay within bounded ranges (e.g., DEPTH ∈ [2, 24], learning rates within 1-2 orders of magnitude of defaults). This prevents the agent from wasting cycles on clearly unproductive regions.

**Third Law (Preserve Progress)**: Every improvement is committed to git. Every regression is reverted. Results are logged to a structured TSV. This ensures monotonic progress on the best-known configuration.

### 3.2 Multi-Agent Specialization

The swarm (Condition C) divides the optimization space into three domains:
- **Architect**: Model architecture (layers, attention, MLP structure)
- **Hypertuner**: Training hyperparameters (LR, batch size, schedules)
- **Regularizer**: Optimizer behavior (momentum, weight decay, warmup)

Agents take turns in rounds. After each round, improvements are shared. No agent can modify another's zone.

### 3.3 Experimental Setup

| Parameter | Value |
|-----------|-------|
| Hardware | NVIDIA RTX 4060 (8GB VRAM) |
| Training data | FineWeb-Edu (via autoresearch/prepare.py) |
| Time budget | 5 minutes per experiment (wall clock) |
| Metric | val_bpb (bits per byte, lower is better) |
| Base model | GPT (50M parameters, 8 layers) |
| Experiments per condition | 30-100 (time permitting) |

### 3.4 Conditions

| Condition | Agent(s) | Governance | Program File |
|-----------|----------|------------|--------------|
| A | Single generalist | None | program.md |
| B | Single generalist | Three Laws | program-governed.md |
| C | 3 specialists | Three Laws + zones | program-swarm.md |

## 4. Experimental Protocol

For each condition:
1. Start from identical `train.py` (the autoresearch baseline)
2. Run baseline training (no modifications) to establish starting val_bpb
3. Allow the agent(s) to iterate autonomously
4. Log every experiment: val_bpb, VRAM, status (keep/discard/crash/governance_revert)
5. Stop after N experiments or human interruption

## 5. Metrics

Primary: **Best val_bpb achieved** (lower is better — same as autoresearch)

Secondary:
- **Improvement rate**: fraction of experiments that improved val_bpb
- **Waste rate**: fraction of experiments that crashed or were reverted
- **Governance intervention rate**: (Conditions B, C only) how often laws triggered
- **Time to best**: number of experiments before reaching the best val_bpb
- **Stability**: variance of val_bpb across experiments

## 6. Expected Results

**Hypothesis 1**: Condition B ≈ Condition A on final val_bpb, with lower waste rate.
*Rationale*: Governance prevents catastrophic exploration but doesn't limit productive exploration. The agent reaches similar performance with fewer wasted cycles.

**Hypothesis 2**: Condition C > Condition A on final val_bpb.
*Rationale*: Specialized agents search their domains more deeply than a generalist. Architecture improvements compound with hyperparameter tuning, which compounds with optimizer tuning.

**Hypothesis 3**: Governance intervention rate is moderate (10-30% of experiments).
*Rationale*: Too low means constraints are irrelevant. Too high means constraints are too tight. A moderate rate indicates the agent is exploring productively but occasionally hitting boundaries that prevent waste.

## 7. Analysis Plan

### 7.1 Statistical Tests
- Mann-Whitney U test for val_bpb distributions between conditions
- Bootstrap confidence intervals for improvement rates
- Permutation test for time-to-best differences

### 7.2 Ablation Studies
- Remove individual laws (First only, Second only, Third only) to measure each law's contribution
- Vary governance tightness (wider vs narrower parameter ranges)
- Vary swarm size (2 agents, 3 agents, 4 agents)

## 8. Contributions

1. **First empirical study of governed autonomous ML research agents**
2. **Asimov's Mind governance framework** — portable, open-source, with adapters for LangChain/CrewAI/AutoGen
3. **Multi-agent collaborative ML research** — demonstrating specialist > generalist for autonomous experimentation
4. **Safety-as-regularization hypothesis** — reframing AI safety constraints as beneficial optimization constraints

## 9. Limitations

- Single GPU, single model family (GPT), single dataset — results may not generalize
- LLM agents (Claude) may have training-data bias toward certain architectures
- 5-minute training budget limits the complexity of architectures that can be evaluated
- The governance framework is not cryptographically enforced in this study (it relies on the agent reading and following the constraints)

## 10. Reproducibility

All code, data, results, and agent logs will be published at:
https://github.com/FutureSpeakAI/asimovs-mind

The experiment can be reproduced with:
```bash
git clone https://github.com/FutureSpeakAI/asimovs-mind
cd asimovs-mind-research
uv sync
uv run prepare.py
python governed/run_experiment.py --condition A --baseline-only
```

## References

[1] Karpathy, A. (2025). autoresearch. https://github.com/karpathy/autoresearch
[2] Bai, Y. et al. (2022). Constitutional AI: Harmlessness from AI Feedback. Anthropic.
[3] Garcia, J. & Fernandez, F. (2015). A Comprehensive Survey on Safe Reinforcement Learning. JMLR.
[4] Wooldridge, M. (2009). An Introduction to MultiAgent Systems. Wiley.
[5] Hutter, F. et al. (2019). Automated Machine Learning. Springer.
