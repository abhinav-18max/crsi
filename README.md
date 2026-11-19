# Mini CRSI

A miniature of *Self-Improvement Memory Enables Continual Recursive Self-Improvement*
(arXiv 2611.18856) in the coding domain, driven from a Jupyter notebook.

A small model solves coding tasks through a harness (one Python file). A stronger model rewrites
the harness round by round and keeps a logbook of what it tried (the Self-Improvement Memory, SIM).
Then the tasks change, and the notebook compares continuing with and without that logbook.

## Run it

```bash
uv sync                              # once: creates .venv with Jupyter, pandas and matplotlib
uv run jupyter lab crsi_mini.ipynb
```

Requirements: Docker running, and the `claude` command-line tool logged in. Model calls go through
`claude -p` and use the Claude Code subscription; no API key is needed. `codex exec` can be used as a
second model (see the cross-model cell in the notebook).

## Layout

| Path | What it is |
|---|---|
| `2611.18856v1.pdf` | the paper |
| `crsi_mini.ipynb` | the notebook: settings, the two experiments, the analysis |
| `harness/seed_v000.py` | the seed harness, the starting point of every run |
| `minicrsi/llm.py` | model backends (`claude`, `codex`), with a disk cache |
| `minicrsi/sandbox.py` | one throwaway Docker container per rollout |
| `minicrsi/evaluate.py` | runs a harness on tasks, enforces the budget, records traces |
| `minicrsi/sim.py` | the Self-Improvement Memory |
| `minicrsi/rsi.py` | proposer, selector, the improvement loop, the four transfer conditions |
| `minicrsi/report.py` | tables and the score chart |
| `minicrsi/bench_source.py`, `bench_target.py` | the 12 + 12 tasks, with hidden verifiers |
| `minicrsi/selfcheck.py` | proves every task is sound: `uv run python -m minicrsi.selfcheck` |
| `runs/<name>/` | everything a run produced: `harness/v000.py ...`, `candidates/`, `sim.json`, `rounds.json` |
| `cache/` | cached model replies and evaluations; delete it to force fresh calls |

## Safety

- Every command the agent runs happens inside a Docker container with no network, capped memory and
  CPU, deleted after the rollout.
- Harness files are written by a model and run in the notebook's Python process. Before a harness
  runs, it is screened: only a short list of standard-library imports, no `open`, `eval`, `exec` or
  double-underscore attributes. It can touch files and processes only through the sandbox object.
  This is a guard against accidents, not a security boundary against a hostile model.

## Limits

This is a teaching-size replica. With 6 held-out tasks and 2 rollouts, one rollout is worth 8.3
score points, so single runs cannot confirm the paper's small average effect. What it shows well is
the mechanism: how the engineer treats inherited harness code when it can, or cannot, read why that
code exists.
