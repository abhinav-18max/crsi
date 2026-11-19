"""Tables and one chart for reading the runs in the notebook."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .rsi import RunResult, diff_text
from .sim import SIM

# Fixed colour and marker per condition (colour follows the condition, never its rank).
STYLE = {
    "cold_start":       ("Cold Start",          "#2a78d6", "o"),
    "harness_transfer": ("Harness Transfer",    "#eb6834", "s"),
    "problem_only":     ("Problem-Only Memory", "#1baf7a", "^"),
    "full_sim":         ("Full SIM",            "#eda100", "D"),
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def _label(key: str) -> str:
    return STYLE.get(key, (key,))[0]


def scores_table(runs: dict[str, RunResult]) -> pd.DataFrame:
    """Held-out score before and after each run, like the paper's Tables 5 and 6."""
    rows = []
    for key, run in runs.items():
        rows.append({
            "condition": _label(key),
            "held-out before": run.before, "held-out after": run.after,
            "held-out gain": None if run.before is None else round(run.after - run.before, 2),
            "evolve before": run.evolve_scores[0], "evolve after": run.evolve_scores[-1],
            "edits kept": len(run.versions) - 1, "rounds": len(run.rounds),
        })
    return pd.DataFrame(rows).set_index("condition")


def curve_table(runs: dict[str, RunResult], split: str = "heldout") -> pd.DataFrame:
    """Score of the retained harness after each round (round 0 is the starting harness)."""
    data = {_label(k): (r.heldout_scores if split == "heldout" else r.evolve_scores) for k, r in runs.items()}
    frame = pd.DataFrame(data)
    frame.index.name = "round"
    return frame


def plot_curves(runs: dict[str, RunResult], split: str = "heldout", title: str | None = None):
    """Line chart: score of the retained harness per round, one line per condition."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ends = []
    for key, run in runs.items():
        name, colour, marker = STYLE.get(key, (key, MUTED, "o"))
        ys = run.heldout_scores if split == "heldout" else run.evolve_scores
        if not ys:
            continue
        ax.plot(range(len(ys)), ys, color=colour, marker=marker, markersize=7, linewidth=2,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
        ends.append((ys[-1], name, len(ys) - 1))
    # Direct labels at the line ends, in ink rather than in the series colour, nudged apart.
    if ends:
        low, high = ax.get_ylim()
        gap = (high - low) * 0.06 or 1
        placed = []
        for value, name, x in sorted(ends):
            y = value if not placed or value - placed[-1] >= gap else placed[-1] + gap
            placed.append(y)
            ax.annotate(f"{name}  {value:.1f}", (x, value), xytext=(x + 0.12, y), color=INK,
                        fontsize=9, va="center", annotation_clip=False)
    rounds = max((len(r.evolve_scores) for r in runs.values()), default=1) - 1
    ax.set_xticks(range(rounds + 1))
    ax.set_xlim(-0.2, rounds + 0.2)
    ax.set_xlabel("round (0 = starting harness)", color=MUTED)
    ax.set_ylabel(("held-out" if split == "heldout" else "evolve") + " score (pass@1, 0-100)", color=MUTED)
    ax.set_title(title or "Score of the retained harness after each round", color=INK, loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=4, labelcolor=INK)
    fig.subplots_adjust(right=0.74, bottom=0.26)
    return fig


def sim_table(sim: SIM) -> pd.DataFrame:
    """One row per recorded attempt: the memory in table form."""
    return pd.DataFrame(sim.rows())


def edit_history(runs: dict[str, RunResult]) -> pd.DataFrame:
    """Which edit each run kept in each round (the paper's Figure 11b)."""
    table = {}
    for key, run in runs.items():
        column = {}
        for rec in run.rounds:
            chosen = next((c for c in rec["candidates"] if c["sample"] == rec["selected"]), None)
            column[f"R{rec['round']}"] = (f"[{rec['problem_id']}] {chosen['what']}" if chosen else "-")
        table[_label(key)] = column
    return pd.DataFrame(table)


def prediction_table(runs: dict[str, RunResult]) -> pd.DataFrame:
    """How often the engineer's "this edit will fix task X" predictions came true."""
    rows = []
    for key, run in runs.items():
        made = hit = 0
        for rec in run.rounds:
            for c in rec["candidates"]:
                if c["sample"] == rec["selected"]:
                    made += len(c["predicted"])
                    hit += len(c["fixed"])
        rows.append({"condition": _label(key), "predictions made": made, "came true": hit,
                     "share": round(hit / made, 2) if made else None})
    return pd.DataFrame(rows).set_index("condition")


def candidates_table(run: RunResult) -> pd.DataFrame:
    """Every candidate of a run: what it tried and how it scored."""
    rows = []
    for rec in run.rounds:
        for c in rec["candidates"]:
            rows.append({"round": rec["round"], "sample": c["sample"],
                         "selected": c["sample"] == rec["selected"], "root cause": c["problem_name"],
                         "edit": c["what"], "screen": c["screen"], "evolve score": c["score"],
                         "baseline": rec["evolve_before"], "predicted": ", ".join(c["predicted"]),
                         "fixed": ", ".join(c["fixed"]), "got worse": ", ".join(c["regressed"]),
                         "changed lines": c["changed_lines"]})
    return pd.DataFrame(rows)


def show_diff(old: str | Path, new: str | Path, max_lines: int = 200) -> None:
    """Print a unified diff between two harness versions."""
    print(diff_text(Path(old).read_text(), Path(new).read_text(), context=3, max_lines=max_lines)
          or "(the two files are identical)")
