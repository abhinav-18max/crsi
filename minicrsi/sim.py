"""Self-Improvement Memory (SIM).

SIM is the optimizer's logbook, not the task-solving model's. It groups harness
edits by the recurring problem they address. One entry is

    problem (root-cause hypothesis)  ->  [attempt, attempt, ...]

and every attempt stores what was changed and why, which tasks the edit was
predicted to fix (written down before evaluation), which it really fixed, and
where it happened (benchmark, model, stage, round). Nothing is ever deleted: an
edit that is later replaced keeps its record.

The JSON layout follows Appendix D.3 of the paper, with a `why` field added for
the rationale.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

MODES = ("none", "problems", "full")


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


class SIM:
    def __init__(self, data: dict | None = None) -> None:
        self.data = data or {"next_id": 1, "carried_from": None, "aliases": {}, "problems": []}

    # ------------------------------------------------------------------ storage
    @classmethod
    def load(cls, path: str | Path) -> "SIM":
        return cls(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.data, indent=2))

    def carried_into(self, source_run: str, target_run: str) -> "SIM":
        """A copy to continue with in a new run, remembering where it came from."""
        clone = SIM(copy.deepcopy(self.data))
        clone.data["carried_from"] = {"source_run": source_run, "into": target_run}
        return clone

    # ------------------------------------------------------------------ access
    @property
    def problems(self) -> list[dict]:
        return self.data["problems"]

    def get(self, problem_id: str | None) -> dict | None:
        if not problem_id:
            return None
        problem_id = self.data["aliases"].get(problem_id, problem_id)
        return next((p for p in self.problems if p["id"] == problem_id), None)

    def ids(self) -> list[str]:
        return [p["id"] for p in self.problems]

    # ------------------------------------------------------------------ update  (the paper's U)
    def record(self, *, problem_id: str | None, problem_name: str, attempt: dict,
               opened_round: int) -> str:
        """Append an attempt to the cited problem, or open a new problem entry.

        A proposal either cites an existing id or names a new root cause. Two
        entries whose root-cause text is identical are the same entry.
        """
        problem = self.get(problem_id)
        if problem is None:
            problem = next((p for p in self.problems if _norm(p["name"]) == _norm(problem_name)), None)
        if problem is None:
            problem = {"id": f"P{self.data['next_id']}", "name": problem_name.strip(),
                       "opened_round": opened_round, "opened_in": attempt.get("stage", ""),
                       "attempts": []}
            self.data["next_id"] += 1
            self.problems.append(problem)
        problem["attempts"].append(attempt)
        return problem["id"]

    # ------------------------------------------------------------------ what the optimizer reads
    def render(self, mode: str = "full") -> str:
        """Text placed in the proposer's and selector's prompts.

        "full"     - root causes with every attempt, prediction, outcome and provenance
        "problems" - root-cause names only (the paper's Problem-Only Memory)
        "none"     - nothing (Harness Transfer / Cold Start)
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "none":
            return ""
        if not self.problems:
            return "(the memory is empty: no harness problem has been recorded yet)"
        lines = []
        for p in self.problems:
            if mode == "problems":
                lines.append(f"- {p['id']}: {p['name']}")
                continue
            lines.append(f"## {p['id']}: {p['name']}")
            lines.append(f"first diagnosed in {p.get('opened_in') or 'this'} run, round {p['opened_round']}")
            for a in p["attempts"]:
                where = f"{a.get('stage', '?')} / benchmark={a.get('benchmark', '?')} / policy={a.get('policy', '?')} / round {a.get('round')}"
                lines.append(f"- attempt [{where}] sample {a.get('sample')}")
                lines.append(f"    edit: {a.get('what', '')}")
                lines.append(f"    why:  {a.get('why', '')}")
                lines.append(f"    predicted to fix: {', '.join(a.get('predicted') or []) or '(nothing named)'}")
                lines.append(f"    actually fixed:   {', '.join(a.get('fixed') or []) or 'none'}"
                             f"   |  predicted but NOT fixed: {', '.join(a.get('missed') or []) or 'none'}")
                if a.get("also_improved"):
                    lines.append(f"    also improved (not predicted): {', '.join(a['also_improved'])}")
                if a.get("regressed"):
                    lines.append(f"    got worse: {', '.join(a['regressed'])}")
                lines.append(f"    evolve-set rollouts passed: {a.get('passed_before')} -> {a.get('passed')} of {a.get('total')}")
            lines.append("")
        return "\n".join(lines).rstrip()

    # ------------------------------------------------------------------ analysis helpers
    def rows(self) -> list[dict]:
        """One flat row per attempt, convenient for a DataFrame."""
        out = []
        for p in self.problems:
            for a in p["attempts"]:
                out.append({"problem": p["id"], "root_cause": p["name"], "stage": a.get("stage"),
                            "benchmark": a.get("benchmark"), "policy": a.get("policy"),
                            "round": a.get("round"), "edit": a.get("what"),
                            "predicted": len(a.get("predicted") or []), "fixed": len(a.get("fixed") or []),
                            "missed": len(a.get("missed") or []), "regressed": len(a.get("regressed") or []),
                            "passed_before": a.get("passed_before"), "passed": a.get("passed"),
                            "total": a.get("total")})
        return out

    def prediction_accuracy(self, stage: str | None = None) -> tuple[int, int]:
        """(predictions that came true, predictions made), optionally for one stage."""
        made = hit = 0
        for row in self.rows():
            if stage and row["stage"] != stage:
                continue
            made += row["predicted"]
            hit += row["predicted"] - row["missed"]
        return hit, made
