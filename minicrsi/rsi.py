"""The recursive self-improvement loop, with optional Self-Improvement Memory.

One round (the paper's Equations 1 and 3):

    F_t     = evaluate(H_t on the evolve tasks)
    C_t     = proposer(H_t, F_t, history D_t ; memory M_t)      -> k candidate harness files
    H_{t+1} = selector(H_t, F_t, C_t, their evaluations ; M_t)  -> one candidate, or keep H_t
    M_{t+1} = update(M_t, the selected edit and its measured outcome)

The proposer and selector are an LLM "harness engineer". The policy model that
solves the tasks is never changed. Held-out tasks are evaluated only to report
a curve; nothing from them reaches the proposer, the selector or the memory.

`memory_mode` decides what the engineer may read from SIM:
    "none"     - nothing                      (Cold Start, Harness Transfer)
    "problems" - root-cause names only        (Problem-Only Memory)
    "full"     - the complete record          (Full SIM)
The memory file itself is always written, so runs can be compared afterwards.
"""
from __future__ import annotations

import difflib
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .evaluate import Budget, EvalResult, evaluate, render_trace, screen_harness
from .sim import MODES, SIM
from .tasks import Task

ENGINEER_SYSTEM = """You are a harness engineer. You improve the Python harness that wraps a fixed \
task-solving language model (the "policy"). You cannot change the policy model, the tasks or the \
budget. You can change only the harness code: its prompts, how it parses replies, which actions it \
offers, how it shows command output to the policy, how it manages context and the turn budget, and \
what evidence it requires before it accepts that a task is finished.

The harness is scored on held-out tasks of the same benchmark that you never see. Only general \
mechanisms help there. Never write task-specific answers or hints into the harness.

Be concrete and economical: prefer the smallest change that removes a diagnosed failure, and keep \
everything that already works."""

MEMORY_GUIDE = """How to use this memory:
1. Treat entries as earlier experience, not as orders. You may choose a different edit.
2. Look at the benchmark and policy model behind each attempt before trusting it.
3. If an existing diagnosis explains a current failure, start from it and cite its id.
4. Reuse the reasoning, but reassess the edit itself when the benchmark or the model has changed:
   the need may carry over while the implementation has to change.
5. Revise, replace or remove a harness mechanism when the new evidence calls for it.
6. Do not repeat a direction that was tried and failed unless new evidence justifies it.
7. Open a new problem when no existing diagnosis explains the failure.
8. Give task-specific evidence more weight when the setting is similar, less when it has changed."""


@dataclass
class RSIConfig:
    rounds: int = 4               # N
    candidates: int = 3           # k proposals per round
    rollouts: int = 2             # attempts per task in every evaluation
    budget: Budget = field(default_factory=Budget)
    memory_mode: str = "full"     # none | problems | full
    selector: str = "rule"        # "rule" (highest evolve score wins) or "llm"
    benchmark: str = "source"     # provenance labels stored in SIM
    stage: str = "source"         # "source" or "target"
    workers: int = 6              # parallel rollouts
    track_heldout: bool = True    # evaluate each retained version on held-out tasks (report only)
    max_trace_tasks: int = 5      # failing tasks whose traces are shown to the proposer
    trace_chars: int = 4500
    cache_dir: str | Path = "cache"

    def __post_init__(self) -> None:
        if self.memory_mode not in MODES:
            raise ValueError(f"memory_mode must be one of {MODES}")
        if self.selector not in ("rule", "llm"):
            raise ValueError('selector must be "rule" or "llm"')


@dataclass
class RunResult:
    name: str
    dir: Path
    start_harness: Path
    final_harness: Path
    versions: list            # retained harness files, v000 first
    sim: SIM
    rounds: list              # one record per round
    evolve_scores: list       # score of the retained harness after 0..N rounds
    heldout_scores: list      # same, on held-out tasks (empty if not tracked)

    @property
    def before(self) -> float | None:
        return self.heldout_scores[0] if self.heldout_scores else None

    @property
    def after(self) -> float | None:
        return self.heldout_scores[-1] if self.heldout_scores else None


# --------------------------------------------------------------------------- small helpers

def diff_text(old: str, new: str, context: int = 2, max_lines: int = 120) -> str:
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), "before", "after",
                                      n=context, lineterm=""))
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... [{len(lines) - max_lines} more diff lines]"]
    return "\n".join(lines)


def _changed_lines(old: str, new: str) -> int:
    return sum(1 for line in difflib.unified_diff(old.splitlines(), new.splitlines(), n=0, lineterm="")
               if line[:1] in "+-" and line[:3] not in ("+++", "---"))


def _json_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def _extract_harness(text: str) -> str | None:
    match = re.search(r"<<<HARNESS\s*\n(.*?)\n\s*HARNESS>>>", text, re.S)
    if not match:
        match = re.search(r"```python\s*\n(.*?)\n```", text, re.S)
    if not match:
        return None
    code = match.group(1)
    fenced = re.fullmatch(r"\s*```(?:python)?\s*\n(.*?)\n```\s*", code, re.S)
    return (fenced.group(1) if fenced else code).strip("\n") + "\n"


def _results_table(result: EvalResult) -> str:
    rows = [f"  {name:<16} {n}/{result.rollouts}" for name, n in result.per_task.items()]
    return f"score {result.score} ({result.passed}/{result.total} rollouts passed)\n" + "\n".join(rows)


def _compare(base: EvalResult, cand: EvalResult) -> tuple[list, list]:
    improved = [t for t, n in cand.per_task.items() if n > base.per_task.get(t, 0)]
    regressed = [t for t, n in cand.per_task.items() if n < base.per_task.get(t, 0)]
    return improved, regressed


# --------------------------------------------------------------------------- prompt sections

def _context(cfg: RSIConfig, source: str, base: EvalResult, round_no: int, policy_name: str,
             history: str, memory: str, with_traces: bool) -> str:
    parts = [
        "# Harness contract\n" + cfg.budget.contract(),
        f"# Current harness ({len(source.splitlines())} lines)\n```python\n{source}```",
        f"# Setting\nbenchmark: {cfg.benchmark} | policy model: {policy_name} | stage: {cfg.stage} "
        f"| round {round_no} of {cfg.rounds}",
        f"# Current harness on the evolve tasks ({cfg.rollouts} rollouts per task)\n" + _results_table(base),
    ]
    if with_traces:
        failing = [r for r in base.results if not r.passed]
        seen, chosen = set(), []
        for r in sorted(failing, key=lambda r: base.per_task[r.task]):   # worst tasks first
            if r.task not in seen:
                seen.add(r.task)
                chosen.append(r)
        traces = "\n\n".join(render_trace(r, cfg.trace_chars) for r in chosen[:cfg.max_trace_tasks])
        parts.append(
            "# Traces of failing rollouts\n"
            "`real stdout` / `real stderr` are what the command actually produced. The policy saw only "
            "what the harness chose to pass on (shown as `harness sent`).\n\n" + (traces or "(no failing rollouts)"))
    parts.append("# History of this run\n" + (history or "(this is the first round)"))
    if cfg.memory_mode == "full":
        parts.append("# Self-Improvement Memory\nA record of harness problems diagnosed so far, the edits "
                     "tried for each, what each edit was predicted to fix and what it really did, and "
                     "where it was tried.\n\n" + memory + "\n\n" + MEMORY_GUIDE)
    elif cfg.memory_mode == "problems":
        parts.append("# Known harness problems\nRoot causes diagnosed earlier (names only). Cite an id when "
                     "one of them explains a current failure.\n\n" + memory)
    return "\n\n".join(parts)


def _history_text(rounds: list) -> str:
    lines = []
    for rec in rounds:
        lines.append(f"## round {rec['round']}: evolve score {rec['evolve_before']} -> {rec['evolve_after']}")
        for c in rec["candidates"]:
            status = "SELECTED" if c["sample"] == rec["selected"] else (
                "rejected before running: " + c["screen"] if c["screen"] != "ok" else "not selected")
            score = "" if c["score"] is None else f" | evolve score {c['score']}"
            lines.append(f"- {c['sample']} [{status}]{score}: {c['what']}")
            if c["score"] is not None:
                lines.append(f"    predicted {c['predicted']} | improved {c['improved']} | got worse {c['regressed']}")
        if rec.get("selected_diff"):
            lines.append("  diff of the selected edit:\n" + rec["selected_diff"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- proposer

def propose(engineer, cfg: RSIConfig, source: str, base: EvalResult, round_no: int, policy_name: str,
            history: str, sim: SIM, salt: str, out_dir: Path) -> list[dict]:
    """Plan k distinct edits, then write one full harness file per edit."""
    memory = sim.render(cfg.memory_mode)
    known = ", ".join(sim.ids()) if cfg.memory_mode != "none" and sim.ids() else ""
    cite = (f'"problem_id": an existing id ({known}) when that diagnosis explains the failure, otherwise null'
            if known else '"problem_id": null')
    plan_prompt = _context(cfg, source, base, round_no, policy_name, history, memory, True) + f"""

# Your job
Propose {cfg.candidates} candidate edits to the harness. Make them genuinely different: different root
causes, or different remedies for the most damaging one. Each must be one coherent change that a
single rewrite of the file can implement.

For every candidate, name the root cause it addresses (a short, general description of the harness
problem, not of one task), and predict which currently failing evolve tasks it will fix. The
prediction is recorded now and checked after the evaluation, so be honest rather than optimistic.

Reply with one JSON object and nothing else:
{{"candidates": [{{"sample": "s1", {cite},
  "problem_name": "<root cause>", "what": "<the edit, one sentence>", "why": "<rationale>",
  "predicted": ["<evolve task name>", ...], "plan": "<how to implement it>"}}, ...]}}"""

    plans = None
    for attempt in range(2):
        try:
            plans = _json_object(engineer.complete(ENGINEER_SYSTEM, plan_prompt, salt=f"{salt}/plan{attempt}"))
            plans = plans["candidates"][:cfg.candidates]
            break
        except (ValueError, KeyError, TypeError):
            plans = None
    if not plans:
        return []

    names = set(base.per_task)
    candidates = []
    for i, plan in enumerate(plans):
        candidates.append({
            "sample": f"s{i + 1}",
            "problem_id": plan.get("problem_id") if sim.get(plan.get("problem_id")) else None,
            "problem_name": str(plan.get("problem_name", "unnamed problem")).strip(),
            "what": str(plan.get("what", "")).strip(), "why": str(plan.get("why", "")).strip(),
            "plan": str(plan.get("plan", "")).strip(),
            "predicted": [t for t in plan.get("predicted") or [] if t in names],
            "path": None, "screen": "not written", "score": None, "passed": None,
            "improved": [], "regressed": [], "fixed": [], "missed": [], "changed_lines": 0,
        })

    base_context = _context(cfg, source, base, round_no, policy_name, history, memory, True)

    def implement(cand: dict) -> None:
        prompt = base_context + f"""

# Your job
Implement exactly this edit and nothing else:

root cause: {cand['problem_name']}
edit: {cand['what']}
rationale: {cand['why']}
implementation notes: {cand['plan']}

Return the COMPLETE new harness file, from the first line to the last, between these two marker
lines, with no other text before or after:
<<<HARNESS
...python source...
HARNESS>>>"""
        code = None
        for attempt in range(2):
            code = _extract_harness(engineer.complete(ENGINEER_SYSTEM, prompt,
                                                      salt=f"{salt}/{cand['sample']}/{attempt}"))
            if code:
                break
        if not code:
            cand["screen"] = "the engineer returned no harness file"
            return
        path = out_dir / f"r{round_no:02d}_{cand['sample']}.py"
        path.write_text(code)
        cand["path"] = str(path)
        cand["changed_lines"] = _changed_lines(source, code)
        ok, reason = screen_harness(path)
        cand["screen"] = "ok" if ok else reason
        if ok and cand["changed_lines"] == 0:
            cand["screen"] = "identical to the current harness"

    with ThreadPoolExecutor(max_workers=max(1, len(candidates))) as pool:
        list(pool.map(implement, candidates))
    return candidates


# --------------------------------------------------------------------------- selector

def select(engineer, cfg: RSIConfig, source: str, base: EvalResult, candidates: list[dict],
           history: str, sim: SIM, salt: str) -> tuple[dict | None, str]:
    """Choose one evaluated candidate, or none (keep the current harness)."""
    ran = [c for c in candidates if c["score"] is not None]
    better = [c for c in ran if c["passed"] > base.passed]
    if not better:
        return None, "no candidate beat the current harness on the evolve tasks"
    by_rule = max(better, key=lambda c: (c["passed"], -len(c["regressed"]), -c["changed_lines"]))
    if cfg.selector == "rule":
        return by_rule, "highest evolve score (ties: fewer regressions, then smaller edit)"

    rows = []
    for c in ran:
        rows.append(
            f"## {c['sample']}: {c['what']}\nroot cause: {c['problem_name']}\nrationale: {c['why']}\n"
            f"evolve score {c['score']} ({c['passed']}/{base.total}; current harness {base.passed}/{base.total})\n"
            f"predicted to fix {c['predicted']} | fixed {c['fixed']} | predicted but not fixed {c['missed']}\n"
            f"other tasks improved {[t for t in c['improved'] if t not in c['fixed']]} | got worse {c['regressed']}\n"
            f"size of edit: {c['changed_lines']} changed lines\n"
            f"diff:\n{diff_text(source, Path(c['path']).read_text(), max_lines=60)}")
    memory = sim.render(cfg.memory_mode)
    prompt = (
        f"# Setting\nbenchmark: {cfg.benchmark} | stage: {cfg.stage}\n\n"
        f"# Current harness on the evolve tasks\n{_results_table(base)}\n\n"
        "# Evaluated candidates\n" + "\n\n".join(rows) + "\n\n"
        "# History of this run\n" + (history or "(first round)") + "\n\n"
        + (f"# Self-Improvement Memory\n{memory}\n\n{MEMORY_GUIDE}\n\n" if cfg.memory_mode == "full" else "")
        + (f"# Known harness problems\n{memory}\n\n" if cfg.memory_mode == "problems" else "")
        + "# Your job\nChoose the candidate to keep, or none. Weigh the measured score, how many tasks "
        "got worse, how general the mechanism is (it must help unseen tasks), and how much complexity "
        "it adds. A candidate that does not beat the current score cannot be chosen.\n"
        'Reply with one JSON object and nothing else: {"choice": "s1" | "s2" | ... | "none", "reason": "<one or two sentences>"}')
    try:
        verdict = _json_object(engineer.complete(ENGINEER_SYSTEM, prompt, salt=f"{salt}/select"))
        choice, reason = str(verdict.get("choice", "")).strip(), str(verdict.get("reason", "")).strip()
    except (ValueError, TypeError):
        return by_rule, "selector reply could not be read; fell back to the highest evolve score"
    if choice == "none":
        return None, "selector kept the current harness: " + reason
    chosen = next((c for c in better if c["sample"] == choice), None)
    if chosen is None:
        return by_rule, f"selector chose {choice!r}, which did not beat the current score; used the highest score"
    return chosen, reason


# --------------------------------------------------------------------------- the loop

def run_rsi(name: str, start_harness: str | Path, evolve: list[Task], heldout: list[Task],
            policy, engineer, cfg: RSIConfig, sim: SIM | None = None,
            runs_dir: str | Path = "runs", verbose: bool = True) -> RunResult:
    """Run N rounds of harness self-improvement and return everything that happened.

    name          - run name; also the folder under `runs_dir`
    start_harness - the harness file to start from (the seed, or an evolved one)
    sim           - memory to continue from (None starts empty)
    """
    run_dir = Path(runs_dir) / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "harness").mkdir(parents=True)
    (run_dir / "candidates").mkdir()
    cache = Path(cfg.cache_dir) / "evals"
    sim = sim or SIM()
    policy_name = policy.ident

    def score(path, tasks, label):
        return evaluate(path, tasks, policy, rollouts=cfg.rollouts, budget=cfg.budget,
                        workers=cfg.workers, cache_dir=cache, label=label, verbose=verbose)

    current = run_dir / "harness" / "v000.py"
    shutil.copy(start_harness, current)
    versions = [current]
    rounds: list[dict] = []
    started = time.time()
    if verbose:
        print(f"=== run {name}: {cfg.stage} / {cfg.benchmark} / memory={cfg.memory_mode} / "
              f"{cfg.rounds} rounds x {cfg.candidates} candidates x {cfg.rollouts} rollouts ===")

    base = score(current, evolve, f"{current.stem} evolve")
    evolve_scores = [base.score]
    heldout_scores = [score(current, heldout, f"{current.stem} held-out").score] if cfg.track_heldout else []

    for t in range(1, cfg.rounds + 1):
        source = current.read_text()
        history = _history_text(rounds)
        salt = f"{name}/r{t}"
        if verbose:
            print(f"--- round {t}: proposing {cfg.candidates} candidates ---")
        candidates = propose(engineer, cfg, source, base, t, policy_name, history, sim, salt,
                             run_dir / "candidates")
        evals = {}
        for cand in candidates:
            if cand["screen"] != "ok":
                if verbose:
                    print(f"  {cand['sample']} rejected before running: {cand['screen']}")
                continue
            res = score(cand["path"], evolve, f"r{t} {cand['sample']} evolve")
            evals[cand["sample"]] = res
            cand["score"], cand["passed"] = res.score, res.passed
            cand["improved"], cand["regressed"] = _compare(base, res)
            cand["fixed"] = [x for x in cand["predicted"] if x in cand["improved"]]
            cand["missed"] = [x for x in cand["predicted"] if x not in cand["improved"]]

        chosen, reason = select(engineer, cfg, source, base, candidates, history, sim, salt)
        record = {"round": t, "version_before": current.name, "evolve_before": base.score,
                  "candidates": [{k: v for k, v in c.items() if k != "plan"} for c in candidates],
                  "selected": chosen["sample"] if chosen else None, "selector_reason": reason,
                  "selected_diff": "", "problem_id": None}
        if chosen:
            new_path = run_dir / "harness" / f"v{len(versions):03d}.py"
            shutil.copy(chosen["path"], new_path)
            record["selected_diff"] = diff_text(source, new_path.read_text(), max_lines=70)
            # The memory update U: only the selected edit is recorded, with its measured outcome,
            # including predictions that did not come true.
            record["problem_id"] = sim.record(
                problem_id=chosen["problem_id"], problem_name=chosen["problem_name"], opened_round=t,
                attempt={
                    "round": t, "sample": chosen["sample"], "accepted": True,
                    "what": chosen["what"], "why": chosen["why"], "files": new_path.name,
                    "predicted": chosen["predicted"], "fixed": chosen["fixed"], "missed": chosen["missed"],
                    "also_improved": [x for x in chosen["improved"] if x not in chosen["fixed"]],
                    "regressed": chosen["regressed"],
                    "passed_before": base.passed, "passed": chosen["passed"], "total": base.total,
                    "stage": cfg.stage, "benchmark": cfg.benchmark, "policy": policy_name, "run": name,
                })
            current, base = new_path, evals[chosen["sample"]]
            versions.append(current)
        record["version_after"] = current.name
        record["evolve_after"] = base.score
        rounds.append(record)
        evolve_scores.append(base.score)
        if cfg.track_heldout:
            heldout_scores.append(score(current, heldout, f"{current.stem} held-out").score)
        if verbose:
            kept = f"selected {chosen['sample']} -> {current.name}: {chosen['what']}" if chosen else "kept the current harness"
            print(f"  round {t} result: {kept}  [{reason}]")
        sim.save(run_dir / "sim.json")
        (run_dir / "rounds.json").write_text(json.dumps(
            {"name": name, "stage": cfg.stage, "benchmark": cfg.benchmark, "memory_mode": cfg.memory_mode,
             "policy": policy_name, "engineer": engineer.ident, "rounds": rounds,
             "evolve_scores": evolve_scores, "heldout_scores": heldout_scores}, indent=2))

    sim.save(run_dir / "sim.json")
    shutil.copy(current, run_dir / "final_harness.py")
    if verbose:
        held = f", held-out {heldout_scores[0]} -> {heldout_scores[-1]}" if heldout_scores else ""
        print(f"=== run {name} finished in {(time.time() - started) / 60:.1f} min: "
              f"evolve {evolve_scores[0]} -> {evolve_scores[-1]}{held} ===")
    return RunResult(name, run_dir, Path(start_harness), run_dir / "final_harness.py", versions, sim,
                     rounds, evolve_scores, heldout_scores)


# --------------------------------------------------------------------------- continual RSI

# The four conditions compared in the paper (Section 3.3). The three transfer
# conditions start from the same source-evolved harness; only the memory differs.
CONDITIONS = {
    "cold_start":       {"start": "seed",    "inherit_memory": False, "memory_mode": "none"},
    "harness_transfer": {"start": "evolved", "inherit_memory": False, "memory_mode": "none"},
    "problem_only":     {"start": "evolved", "inherit_memory": True,  "memory_mode": "problems"},
    "full_sim":         {"start": "evolved", "inherit_memory": True,  "memory_mode": "full"},
}


def run_conditions(prefix: str, seed_harness: str | Path, source: RunResult, evolve: list[Task],
                   heldout: list[Task], policy, engineer, cfg: RSIConfig,
                   conditions=tuple(CONDITIONS), runs_dir: str | Path = "runs",
                   verbose: bool = True) -> dict[str, RunResult]:
    """Continue optimization in a new setting under each requested condition.

    `source` is the finished source run: its final harness and its memory are
    what gets carried over. `cfg` describes the target setting; its
    `memory_mode` is overridden per condition. Budgets are identical across
    conditions, as in the paper.
    """
    from dataclasses import replace

    out = {}
    for key in conditions:
        spec = CONDITIONS[key]
        name = f"{prefix}-{key}"
        start = source.final_harness if spec["start"] == "evolved" else seed_harness
        memory = source.sim.carried_into(source.name, name) if spec["inherit_memory"] else None
        out[key] = run_rsi(name, start, evolve, heldout, policy, engineer,
                           replace(cfg, memory_mode=spec["memory_mode"]), memory, runs_dir, verbose)
    return out


def load_run(name: str, runs_dir: str | Path = "runs") -> RunResult:
    """Reload a finished run from disk without spending any inference."""
    run_dir = Path(runs_dir) / name
    meta = json.loads((run_dir / "rounds.json").read_text())
    versions = sorted((run_dir / "harness").glob("v*.py"))
    return RunResult(name, run_dir, versions[0], run_dir / "final_harness.py", versions,
                     SIM.load(run_dir / "sim.json"), meta["rounds"], meta["evolve_scores"],
                     meta["heldout_scores"])
