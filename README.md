# Atom Donor Game

Companion code and paper for **Atom Hybrid: Emergent Cooperation via Hebbian Learning and Lazy LLM Inference**.

A multi-agent simulation of the iterated Donor Game comparing three architectures:

- **A: Single LLM**: stateless control, all agents share one LLM
- **B: LLM Society**: per-agent LLM with private reputation memory
- **C: Atom Hybrid**: Hebbian S1 (fast) + lazy LLM S2 (slow), gossip-based reputation

## Result

Across N=3 replications, Atom Hybrid converges to **100% cooperation** by tick ~120 while using **97.2% fewer LLM calls** than the per-agent LLM baseline.

| Condition | Avg Coop (last 20t) | LLM Calls (Run 7, 300t) |
|-----------|---------------------|--------------------------|
| A: Single LLM | 98.9% | 8994 |
| B: LLM Society | 97.0% | 9141 |
| **C: Atom Hybrid** | **100.0%** | **63** |

## Quick start

Requires Rust toolchain and a local [Ollama](https://ollama.com) server with three models:

```sh
ollama pull qwen3:0.6b
ollama pull qwen3:8b
ollama pull llama3.2:3b

cargo build --release
./target/release/donor_world 200 20  # 200 ticks, 20 agents
```

The binary runs all three conditions sequentially and prints a summary table.

## Repository layout

```
src/
  donor_agent.rs     S1 Hebbian weights, reputation, reinforce, breed
  donor_bench.rs     World loop, three conditions, gossip protocol, main
paper/
  atom_hybrid.md     Source paper (Markdown)
  atom_hybrid.tex    Generated LaTeX (via build_latex.py + pandoc + postprocess_tex.py)
  atom_hybrid.pdf    Compiled PDF
  figures/           Cooperation trajectory chart
  plot_trajectory.py Reproduces Figure 1 from raw run data
data/
  run{6,7,8,9}_*.txt Raw stdout from the four reported runs
```

## Reproducing the paper figure

```sh
cd paper
python3 plot_trajectory.py
```

Reads hardcoded data points (also present in `data/run*_*.txt`) and writes `figures/cooperation_trajectory.{png,pdf}`.

## Hardware reference

Runs in the paper were performed on Apple M3 Pro, 18GB RAM, with Ollama serving the three models locally. Wall-clock is dominated by LLM inference latency (~30 minutes per condition at 200 ticks; ~90 minutes per condition at 300 ticks).

## License

MIT.
