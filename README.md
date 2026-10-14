# GS-MADE

Reference implementation accompanying:

> Kumar, K. and Singh, M., "GS-MADE: A Graph-Seeded Memetic Metaheuristic
> for Generalizable Multi-Objective Workflow Scheduling in Fog-Cloud
> Systems", *International Journal of Intelligent Engineering and
> Systems*, 2026.

GS-MADE schedules multi-objective (makespan, energy, cost) scientific
workflows across a simulated IoT-edge / fog / cloud environment. A
graph-attention network seeds part of an adaptive differential
evolution (L-SHADE) population, which is periodically refined by a
graph-aware variable neighbourhood search. This repository contains
GS-MADE itself, 11 reimplemented baselines, the three-tier simulator,
the instance generators and seed blocks, and the scripts that run the
evaluation / ablation / transfer-learning protocols and regenerate
tables and figures.

This package is what the paper's "Data and code availability"
paragraph refers to: *"The instance manifests, generator invocations
and WfCommons identifiers, the three-tier configuration file, the 30
seeds, the scheduler and baseline implementations, the simulator
harness and the per-run objective vectors behind Tables 4–9 and Figures
2–3 ... with the scripts that regenerate every table and figure in this
paper."* Read **"What this is / is not"** below before citing a number
out of a `--full` run as if it were from the paper.

## What this is 

**This is a real, runnable, independently-verified implementation** of
every algorithmic component the paper describes: the LRV encoding, the
GAT seeding policy (REINFORCE-trained), multi-objective L-SHADE, the
four VNS neighbourhoods, all 11 baselines, the three-tier discrete-event
simulator, and the evaluation/ablation/transfer-learning protocols.
Every non-trivial component has a unit test that checks it against an
independent ground truth (a finite-difference gradient, a hand-computed
or Monte-Carlo-cross-checked hypervolume, a precedence-feasibility
invariant, a round-trip encoding guarantee, ...) rather than just
"it ran without crashing" — see `tests/` and **Running the tests**
below.




## Install

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
pip install -e .        # optional: installs the `gsmade` package itself
```

Requires Python ≥ 3.10. No GPU, no PyTorch, and no network access are
needed for anything in this repository by default — `gsmade/nn.py` is a
small, dependency-free NumPy autodiff engine (see **Swapping in
PyTorch** below if you want one anyway).

## Quickstart (a few minutes, smoke scale)

```bash
# 1. Generate synthetic instances (5 Pegasus-style families x 3 scales x 6 seeds)
python scripts/generate_instances.py

# 2. Generate the evaluation/training seed blocks (reproduces the paper's
#    documented {11, 23, 37, ..., 331} shape from one fixed meta-seed)
python scripts/generate_seeds.py

# 3. Run the pipeline at smoke scale (small pop size / eval budget, a
#    couple of instances and seeds) -- every script below defaults to
#    this; pass --full for the paper's protocol scale (see next section)
python scripts/run_evaluation.py
python scripts/run_ablation.py
python scripts/run_transfer.py

# 4. Render Markdown tables and PNG figures from whatever the above produced
python scripts/make_tables.py
python scripts/make_figures.py
```

Results land in `results/` (gitignored — see `results/README.md`):
`raw_results.csv`, `fronts.json`, `ablation_results.csv`,
`transfer_results.csv`, `tables.md`, and a few `fig_*.png` files.

## Running at the paper's protocol scale

Every `run_*.py` script accepts `--full`, which switches to
`pop_size=70`, `max_evals=7340`, all 30 evaluation seeds, and (for
`run_evaluation.py`) all instances and all 11 baselines — the same
order of magnitude of compute the paper describes. This is *slow*
(hours, not minutes; `run_evaluation.py --full` is roughly
instances × seeds × 12 methods independent optimization runs) and is
meant for a real machine, not a laptop smoke test:

```bash
python scripts/generate_instances.py --seeds-per-scale 6   # the full 90-instance synthetic pool
python scripts/run_evaluation.py --full
python scripts/run_ablation.py --full
python scripts/run_transfer.py --full
python scripts/make_tables.py
python scripts/make_figures.py --full
```

To add the four real WfCommons traces (1000Genome, Cycles, SoyKB,
SRA_Search) used for the paper's zero-shot protocol and Figure 3 — not
redistributed here, see `data/instances/README.md` for why — fetch or
download them and run `scripts/convert_wfcommons_trace.py` before
`run_transfer.py`; the zero-shot protocol auto-detects and uses
whatever real-trace instances it finds under `data/instances/`, and
prints a clear skip message if it finds none.

## Repository layout

```
gsmade/                 the installable package
  workflow.py             Task/Workflow DAG model, (de)serialization, Table-2-style stats
  environment.py          three-tier node pool, loaded from configs/three_tier.yaml
  encoding.py              shared LRV chromosome: decode is precedence-feasible by construction
  objectives.py           the shared discrete-event simulator (makespan/energy/cost)
  nn.py                   dependency-free NumPy reverse-mode autodiff (Tensor, Linear, MLP, Adam)
  gat_policy.py           2-layer GAT seeding policy + REINFORCE trainer
  lshade.py               multi-objective L-SHADE (GDE3 survivor selection, external archive)
  vns.py                  the 4 graph-aware neighbourhoods (N1-N4) + shake-then-descend driver
  scheduler.py            GSMADE: wires policy + L-SHADE + VNS together; A1-A5 config mapping
  metrics.py              hypervolume, IGD, knee point, Friedman/Nemenyi/Wilcoxon
  baselines/               the 11 baselines (HEFT, Min-Min, PSO, GWO, NSGA-II, cold-start
                           L-SHADE, IPSO-GWO, IMOMA, DRLMOTS, GA-DRL, PF-MPPO)

scripts/
  generators.py            synthetic instance generators (5 Pegasus-style families)
  generate_instances.py    CLI around generators.py; writes data/instances/manifest.json
  generate_seeds.py        the 30 evaluation seeds + a disjoint training-seed block
  convert_wfcommons_trace.py  real WfCommons trace JSON -> this repo's Workflow format
  run_evaluation.py        main comparison: GS-MADE + 11 baselines -> raw_results.csv, fronts.json
  run_ablation.py          Table 6's A1-A5 arms -> ablation_results.csv
  run_transfer.py          in-family / LOWO / cross-scale / zero-shot -> transfer_results.csv
  make_tables.py           renders the above into results/tables.md
  make_figures.py          renders a Pareto-front scatter, an ablation bar chart, a beta-sweep plot

configs/three_tier.yaml   machine-readable Table 3
data/seeds/, data/instances/   generated artifacts (gitignored; see their own READMEs)
results/                  generated artifacts (gitignored; see results/README.md)
tests/                    pytest suite -- see below
```

## Running the tests

```bash
pip install -e ".[dev]"   # or just: pip install pytest
pytest                     # ~150 tests, a few seconds
```

What's actually checked, not just exercised:

- **`test_nn.py`** — every `Tensor` op (including the GAT's softmax
  attention and broadcasting) against central finite-difference
  gradients.
- **`test_metrics.py`** — `hypervolume()` against an independent
  Monte-Carlo volume estimate (not just a re-derivation of the same
  sweep logic), plus IGD, Pareto filtering, and the Demsar (2006)
  Nemenyi table lookup.
- **`test_encoding.py` / `test_vns.py`** — precedence-feasibility of
  every decoded schedule, and the exact `decode -> reencode -> decode`
  round trip VNS depends on to share one chromosome population with
  L-SHADE.
- **`test_environment.py`** — `configs/three_tier.yaml` loads into
  exactly Table 3's node counts, pricing, and ranges.
- **`test_lshade.py`** — non-dominated sorting, crowding distance, and
  the external archive's dominance/duplicate bookkeeping, plus an
  end-to-end L-SHADE run.
- **`test_scheduler.py`** — every A1/A2/A3/A5 ablation-arm config from
  `gsmade/scheduler.py`'s own docstring actually runs and returns a
  valid front (a regression guard against the config mapping and the
  docstring drifting apart).
- **`test_gat_policy.py`** — a pinned, fixed-seed regression test that
  REINFORCE training measurably improves mean reward on a toy workflow.
- **`test_baselines.py`** — all 11 baselines, parametrized, each
  checked for a non-empty front of finite, non-negative 3-objective
  vectors, plus a seed-reproducibility check.
- **`test_generators_and_seeds.py`** — generated instances are
  connected DAGs of the requested size; the seed blocks match the
  paper's documented shape and are disjoint.
- **`test_wfcommons_converter.py`** — the trace converter against two
  hand-built fixture schemas (explicit parents/children, and
  file-sharing-inferred edges) — not real WfCommons data.
- **`test_end_to_end.py`** — the full pretrain → GS-MADE + baselines →
  pool fronts → hypervolume/IGD pipeline, the formal version of the
  manual smoke test used while building this repository.

## Swapping in PyTorch

`gsmade/nn.py` is a small NumPy autodiff engine so the whole pipeline
runs with no extra wheel to fetch. If you want GPU acceleration for
large-scale policy training, the natural drop-in points are
`gsmade/gat_policy.py`'s `GraphAttentionLayer`/`GATPolicy` classes and
`gsmade/baselines/dqn_common.py` — both only use `nn.py`'s `Tensor`,
`Linear`, and `Adam` through a small, explicit interface
(`__call__`, `.params()`, `.backward()`, `.step()`), so a PyTorch
`nn.Module`-based reimplementation of the same architecture (2-layer
GAT, Gaussian output heads, REINFORCE loss) can be swapped in without
touching `scheduler.py`, `encoding.py`, or anything downstream of a
sampled chromosome.

## Reproducibility notes

- **Instance generation** (`scripts/generators.py`) is a documented
  *structural approximation* of each Pegasus family's known topology
  (fan-out/fan-in pattern, relative depth/width) — not the real Pegasus
  synthetic generator the paper cites, which is a separate tool
  maintained by the Pegasus project. Every generated instance is
  reproducible from its documented seed; `data/instances/manifest.json`
  records exactly which seed produced which file.
- **Seeds**: `scripts/generate_seeds.py` reproduces the paper's stated
  evaluation-seed shape (`{11, 23, 37, ..., 331}`, 30 values) from one
  fixed, documented meta-seed. If you are an author of the paper and
  have the actual recorded seed list, replace
  `data/seeds/seeds.json`'s `"evaluation"` array directly — nothing
  downstream cares how that file was produced, only that every script
  reuses the same one.
- **Evaluation budget**: `LSHADEConfig.max_evals` is a *shared* budget
  across every `objectives.simulate()` call GS-MADE makes, including the
  ones VNS performs internally (N1 in particular evaluates every node in
  the environment to place one task, so a single VNS call on a large
  instance can itself be dozens of simulate() calls) -- `gsmade/
  scheduler.py`'s `_vns_hook` charges those back to `evals_used` and caps
  each VNS call by whatever budget is actually left, so `evals_used` in
  `results/*.csv` is the true evaluation count, and a run's wall-clock
  cost stays governed by `--full` vs. the smoke defaults rather than by
  how "improvable" a given random starting individual happens to be.
- **Simulator**: `gsmade/objectives.py`'s `simulate()` is a simplified,
  deterministic re-implementation of a FogWorkflowSim-style three-tier
  simulator (task start = later of node-free-time and all predecessor
  data arrivals; energy = active/idle power integrated over the
  schedule plus a radio-power communication term; cost = busy-time ×
  tier price). It is not FogWorkflowSim itself.
- **Baselines**: HEFT and Min-Min are single-schedule heuristics and
  report a one-point archive (the paper's "dash in Table 5"); the
  others share this repo's LRV chromosome and discrete-event simulator
  so a result difference reflects the search strategy, not the
  representation. `gsmade/baselines/__init__.py`'s docstring lists
  which baselines are generic applications of that shared protocol and
  which are adapted from their own source papers' task models.
- **What's simplified, and said so in the code**: GA-DRL's graph
  encoder is fixed/untrained (only its Q-head trains) — see
  `gsmade/baselines/ga_drl.py`; PF-MPPO is a "PPO-lite" with a
  per-task MLP actor, not the paper's exact architecture — see
  `gsmade/baselines/pf_mppo.py`. Both are documented at the point of
  simplification rather than silently presented as exact
  reproductions.
-

## License

MIT — see `LICENSE`. Please cite the paper (see `CITATION.cff`) if you
use this code.
