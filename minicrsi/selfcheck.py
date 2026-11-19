"""Prove every task is sound: the untouched state fails and the reference solution passes.

    python -m minicrsi.selfcheck            # both benchmarks
    python -m minicrsi.selfcheck source     # one benchmark
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

from .sandbox import DockerSandbox
from .tasks import VERIFY_PRELUDE, Task, load


def check_task(task: Task) -> tuple[str, bool, str]:
    script = VERIFY_PRELUDE + task.verify
    with DockerSandbox(task.files) as box:
        before, before_out = box.verify(script)
    with DockerSandbox({**task.files, **task.solution}) as box:
        after, after_out = box.verify(script)
        extra = ""
        if "test_public.py" in task.files:
            res = box.run("python -m unittest -q test_public", timeout=30)
            if res.exit_code != 0:
                extra = "public tests fail on the reference solution: " + res.stderr[-300:]
        if "check.py" in task.files:
            res = box.run("python check.py", timeout=30)
            if res.exit_code != 0:
                extra = "check.py fails on the reference solution: " + (res.stdout + res.stderr)[-300:]
    if before:
        return task.name, False, "verifier passes before any work is done"
    if not after:
        return task.name, False, "reference solution rejected: " + after_out[-400:]
    if extra:
        return task.name, False, extra
    return task.name, True, "ok"


def run(benchmarks=("source", "target"), workers: int = 6) -> bool:
    tasks = [t for b in benchmarks for t in load(b)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(check_task, tasks))
    for name, ok, msg in results:
        print(f"{'ok  ' if ok else 'FAIL'} {name:<18} {'' if ok else msg}")
    good = sum(ok for _, ok, _ in results)
    print(f"{good}/{len(results)} tasks sound")
    return good == len(results)


if __name__ == "__main__":
    sys.exit(0 if run(tuple(sys.argv[1:]) or ("source", "target")) else 1)
