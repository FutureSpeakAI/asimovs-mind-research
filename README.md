# Asimov's Mind Research

### What happens when you govern autonomous ML research agents? They get better.

This repo contains the empirical research behind [Asimov's Mind](https://github.com/FutureSpeakAI/asimovs-mind), a Claude Code plugin that deploys governed agent swarms for autonomous code improvement. We forked [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), added a governance framework (Asimov's cLaws), and ran controlled experiments to answer a simple question: **do safety constraints help or hurt autonomous ML research?**

They help. Significantly.

Built by [FutureSpeak.AI](https://github.com/FutureSpeakAI).

---

## The Experiment

Three conditions, identical hardware (RTX 4060, 8GB VRAM), identical training task (GPT on FineWeb-Edu, 5-minute budget per experiment):

| | A: Ungoverned | B: Governed Single | C: Governed Swarm |
|---|:-:|:-:|:-:|
| **Crash rate** | 56% | 22% | 25% |
| **Degradation per step** | 0.035 | 0.018 | 0.011 |
| **Explored architecture** | No | No | Yes |
| **Final cumulative val_bpb** | 1.961 | 1.945 | 1.885 |

28 controlled training runs, zero crashes in the new experiment runner, ~5 hours total compute.

![Results](chart_paper_summary.png)

## Key Findings

**1. Governance halves crash rates.** The ungoverned agent crashed on 56% of experiments. Governed agents crashed 22-25%. Less waste means more productive exploration in the same time budget.

**2. The governed swarm degraded 3x slower.** During cumulative exploration (stacking parameter changes), the ungoverned agent's val_bpb worsened at 0.035 per step. The governed swarm: 0.011 per step. Governance acts as regularization on the search process itself.

**3. Specialist agents explore dimensions generalists ignore.** The swarm's architect agent was the only one to try an architectural change (ASPECT_RATIO). Neither generalist agent (A or B) ever explored architecture. Specialization forces diversity of exploration.

**4. The autoresearch keep/discard mechanism is noisy.** Of 10 isolated parameter changes tested, zero genuinely improved over baseline. Yet AI agents "kept" many of these during cumulative runs. Run-to-run variance (~0.03 val_bpb) exceeds most individual effects, leading to false-positive keeps.

**5. The only "improvement" was an illusion.** ASPECT_RATIO=48 appeared to improve val_bpb, but further analysis revealed that `build_model_config()` rounds to HEAD_DIM multiples -- AR=48 and AR=64 produce the identical model with DEPTH=6. The keep/discard mechanism kept a literal no-op.

## The Paper

The full paper is in this repo: **[PAPER.md](PAPER.md)**

*Governed Multi-Agent Optimization: Safety Constraints as Regularization in Autonomous ML Research*

It covers methodology, the Three Laws governance framework, all experimental results, the keep/discard noise analysis, and discussion of safety-as-regularization.

## How This Differs from Autoresearch

This repo is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The base training code (`train.py`, `prepare.py`) is Karpathy's. Everything in `governed/` is ours.

| | Autoresearch | This Research |
|---|---|---|
| **Agents** | 1 generalist | 3 conditions (ungoverned, governed, swarm) |
| **Governance** | None | Three Laws + circuit breakers + zone enforcement |
| **Experiment design** | Cumulative only | Isolated + cumulative + optimal (separates individual effects from interactions) |
| **Experiment runner** | In-context agent loop | Standalone Python runner with proper process management |
| **Analysis** | Single-condition progress chart | Multi-condition comparison, parameter sensitivity, degradation rates |
| **Discovery** | Manual hyperparameter changes | Safety-scanned GitHub code discovery pipeline |

## What We Built

### Experiment Infrastructure

- **`governed/experiment_runner.py`** -- Robust experiment runner that caches baseline train.py, applies isolated changes without git branch switching, and uses process-tree killing (not blanket taskkill). Fixes the three crash causes in the original runner.
- **`generate_charts.py`** -- Generates all analysis charts from `governed/results/all_results.tsv`
- **`governed/results/history_{A,B,C}.tsv`** -- Reconstructed experiment history from git branches

### Governance Framework

- **`governed/governance.json`** -- The Three Laws: Do No Harm (circuit breakers), Obey Protocol (bounded parameter ranges), Preserve Progress (git commit discipline)
- **`governed/program-governed.md`** -- Governed agent instructions
- **`governed/program-swarm.md`** -- Multi-agent swarm instructions with zone enforcement

### Discovery System (Prototype)

- **`governed/discovery/safety_scanner.py`** -- AST-based static analysis (Tier 1/2/3 findings)
- **`governed/discovery/scout.py`** -- GitHub search with relevance + trust scoring
- **`governed/discovery/adapter.py`** -- Component extraction and adaptation
- **`governed/discovery/provenance.py`** -- Three-stage attribution tracking
- **`governed/governance_v2.json`** -- Extended Three Laws for code import safety

## Reproducing the Experiments

```bash
git clone https://github.com/FutureSpeakAI/asimovs-mind-research
cd asimovs-mind-research
uv sync
uv run prepare.py

# Run the full experiment suite (~5.5 hours on RTX 4060)
python governed/experiment_runner.py all

# Generate charts
uv run generate_charts.py
```

Requirements: NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

## Charts

| Chart | What it shows |
|-------|---------------|
| `chart_paper_summary.png` | 2x2 combined panel (the paper figure) |
| `chart_param_sensitivity.png` | Isolated experiment results -- which changes help? |
| `chart_cumulative_progress.png` | All conditions over time |
| `chart_crash_rates.png` | Historical crash rate comparison |
| `chart_degradation_rate.png` | Degradation per cumulative step |

## Project Structure

```
asimovs-mind-research/
+-- PAPER.md                    # The full research paper
+-- train.py                    # GPT model + training loop (from autoresearch)
+-- prepare.py                  # Data prep + evaluation (from autoresearch, read-only)
+-- generate_charts.py          # Chart generation from experiment data
+-- chart_*.png                 # Analysis charts
+-- governed/
|   +-- experiment_runner.py    # Robust experiment runner (v2)
|   +-- governance.json         # Three Laws governance rules
|   +-- governance_v2.json      # Extended rules for code discovery
|   +-- program-governed.md     # Governed agent instructions
|   +-- program-swarm.md        # Multi-agent swarm instructions
|   +-- program-discovery.md    # Discovery mode instructions
|   +-- results/                # Experiment data (TSVs)
|   +-- discovery/              # Code discovery system prototype
+-- program.md                  # Original autoresearch agent instructions
+-- analysis.ipynb              # Jupyter analysis notebook
```

## Credits

- **[FutureSpeak.AI](https://github.com/FutureSpeakAI)** -- Research design, governance framework, experiment infrastructure, discovery system, and paper
- **[autoresearch](https://github.com/karpathy/autoresearch)** by Andrej Karpathy -- The training code and core iteration pattern that this research builds upon
- **[Agent Friday](https://github.com/FutureSpeakAI/Agent-Friday)** by FutureSpeak.AI -- Origin of the cLaws governance system

## Related

- **[Asimov's Mind](https://github.com/FutureSpeakAI/asimovs-mind)** -- The Claude Code plugin that implements these research findings as a production tool

## License

MIT
