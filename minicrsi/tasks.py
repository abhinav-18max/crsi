"""Task format, benchmark loading and the evolve / held-out split."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    name: str
    instruction: str            # the only text the harness receives
    files: dict                 # path -> content, placed in /work before the agent starts
    verify: str                 # hidden Python script; exit code 0 means the task is solved
    solution: dict = field(default_factory=dict)   # reference files, used only by the self-check
    kind: str = ""              # coarse category, used to stratify the split

    @property
    def digest(self) -> str:
        blob = json.dumps([self.name, self.instruction, self.files, self.verify], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# Prepended to every hidden verifier. `check(name, condition)` records a failure;
# `finish()` exits non-zero if any check failed. Any crash also counts as a failure.
VERIFY_PRELUDE = r'''
import json, os, subprocess, sys, traceback
sys.path.insert(0, "/work")
os.chdir("/work")
_fails = []

def check(name, cond):
    if not cond:
        _fails.append(name)

def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    except Exception:
        return False
    return False

def sh(cmd, timeout=20, cwd="/work"):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd)

def finish():
    if _fails:
        print("FAILED checks:", "; ".join(_fails[:8]))
        sys.exit(1)
    print("ALL HIDDEN CHECKS PASSED")
    sys.exit(0)

def _excepthook(kind, value, tb):
    print("VERIFIER ERROR:", "".join(traceback.format_exception(kind, value, tb))[-600:])
    sys.exit(1)

sys.excepthook = _excepthook
'''


def load(benchmark: str) -> list[Task]:
    """Return the tasks of one benchmark: "source" or "target"."""
    if benchmark == "source":
        from .bench_source import TASKS
    elif benchmark == "target":
        from .bench_target import TASKS
    else:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    names = [t.name for t in TASKS]
    assert len(names) == len(set(names)), "duplicate task names"
    return list(TASKS)


def split(tasks: list[Task], seed: int = 0, stratify: bool = True) -> tuple[list[Task], list[Task]]:
    """Split into (evolve, held_out) halves.

    The optimizer only ever sees the evolve half. With `stratify=True` each task
    kind is split separately, which keeps the two halves comparable on a
    benchmark this small (the paper splits without stratification).
    """
    rng = random.Random(seed)
    groups: dict[str, list[Task]] = {}
    for task in tasks:
        groups.setdefault(task.kind if stratify else "", []).append(task)
    ordered = []
    for kind in sorted(groups):
        members = sorted(groups[kind], key=lambda t: t.name)
        rng.shuffle(members)
        ordered += members
    # Deal the shuffled tasks out alternately: the halves have equal size and each
    # kind is divided as evenly as possible.
    evolve, held = ordered[0::2], ordered[1::2]
    return sorted(evolve, key=lambda t: t.name), sorted(held, key=lambda t: t.name)
