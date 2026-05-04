import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ["none", "read_only", "write_only", "joint", "prompt_safety"]
LABELS = {
    "none": "none (baseline)",
    "read_only": "read_only",
    "write_only": "write_only",
    "joint": "joint",
    "prompt_safety": "prompt_safety",
}
TURNS = [2, 3, 4]


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def turn_toxicity(rec: Dict[str, Any], turn: int) -> Optional[float]:
    by_turn = ((rec.get("scores") or {}).get("detoxify_by_turn") or {})
    item = by_turn.get(str(turn))
    if isinstance(item, dict):
        try:
            return float(item.get("toxicity"))
        except (TypeError, ValueError):
            return None
    if item is not None:
        try:
            return float(item)
        except (TypeError, ValueError):
            return None
    return None


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1500, ci: float = 95.0, seed: int = 123) -> Tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = values.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[i] = float(np.mean(sample))
    mean = float(np.mean(values))
    lo = float(np.percentile(boots, (100.0 - ci) / 2.0))
    hi = float(np.percentile(boots, 100.0 - (100.0 - ci) / 2.0))
    return mean, lo, hi


def mean_turn4_by_seed(records: List[Dict[str, Any]]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for rec in records:
        seed_id = str(rec.get("seed_id", "")).strip()
        if not seed_id:
            continue
        val = turn_toxicity(rec, 4)
        if val is None:
            continue
        grouped.setdefault(seed_id, []).append(val)
    return {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in grouped.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/sanitization_ablation")
    parser.add_argument("--out_pdf", default="figures/sanitization_ablation.pdf")
    parser.add_argument("--n_boot", type=int, default=1500)
    parser.add_argument("--rng_seed", type=int, default=123)
    args = parser.parse_args()

    data: Dict[str, List[Dict[str, Any]]] = {}
    for cond in CONDITIONS:
        path = os.path.join(args.data_dir, f"threads_{cond}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file for condition {cond}: {path}")
        data[cond] = load_jsonl(path)

    colors = {
        "none": "#4c78a8",
        "read_only": "#f58518",
        "write_only": "#54a24b",
        "joint": "#e45756",
        "prompt_safety": "#b279a2",
    }

    fig = plt.figure(figsize=(13.5, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.35, wspace=0.28)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_line = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    # Panel A: turn-4 toxicity bar chart with CI.
    bar_means: List[float] = []
    bar_lo: List[float] = []
    bar_hi: List[float] = []
    for i, cond in enumerate(CONDITIONS):
        vals = [v for v in (turn_toxicity(rec, 4) for rec in data[cond]) if v is not None]
        arr = np.asarray(vals, dtype=float) if vals else np.asarray([], dtype=float)
        mean, lo, hi = bootstrap_mean_ci(arr, n_boot=args.n_boot, seed=args.rng_seed + i)
        bar_means.append(mean)
        bar_lo.append(mean - lo)
        bar_hi.append(hi - mean)

    x = np.arange(len(CONDITIONS))
    ax_bar.bar(
        x,
        bar_means,
        yerr=np.vstack([np.asarray(bar_lo), np.asarray(bar_hi)]),
        capsize=4,
        color=[colors[c] for c in CONDITIONS],
        alpha=0.9,
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([LABELS[c] for c in CONDITIONS], rotation=18, ha="right")
    ax_bar.set_ylabel("turn-4 mean toxicity")
    ax_bar.set_title("Turn-4 toxicity by condition (95% bootstrap CI)")
    ax_bar.grid(axis="y", alpha=0.25)

    # Panel B: toxicity trajectory turns 2-4.
    for i, cond in enumerate(CONDITIONS):
        means: List[float] = []
        lows: List[float] = []
        highs: List[float] = []
        for t in TURNS:
            vals = [v for v in (turn_toxicity(rec, t) for rec in data[cond]) if v is not None]
            arr = np.asarray(vals, dtype=float) if vals else np.asarray([], dtype=float)
            mean, lo, hi = bootstrap_mean_ci(arr, n_boot=args.n_boot, seed=args.rng_seed + 100 + i * 10 + t)
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
        xt = np.asarray(TURNS, dtype=float)
        ym = np.asarray(means, dtype=float)
        yl = np.asarray(lows, dtype=float)
        yh = np.asarray(highs, dtype=float)
        ax_line.plot(xt, ym, marker="o", color=colors[cond], label=LABELS[cond], linewidth=2)
        ax_line.fill_between(xt, yl, yh, color=colors[cond], alpha=0.16)
    ax_line.set_xticks(TURNS)
    ax_line.set_xlabel("turn")
    ax_line.set_ylabel("mean toxicity")
    ax_line.set_title("Toxicity trajectory across downstream turns")
    ax_line.grid(alpha=0.25)
    ax_line.legend(frameon=False, fontsize=8)

    # Panel C: condition table.
    table_rows: List[List[str]] = []
    for cond in CONDITIONS:
        turn4_vals = [v for v in (turn_toxicity(rec, 4) for rec in data[cond]) if v is not None]
        arr4 = np.asarray(turn4_vals, dtype=float) if turn4_vals else np.asarray([], dtype=float)
        if arr4.size == 0:
            mean_tox = 0.0
            max_tox = 0.0
            pct_above = 0.0
        else:
            mean_tox = float(np.mean(arr4))
            max_tox = float(np.max(arr4))
            pct_above = float(np.mean(arr4 > 0.5) * 100.0)
        table_rows.append(
            [
                LABELS[cond],
                f"{mean_tox:.4f}",
                f"{max_tox:.4f}",
                f"{pct_above:.1f}%",
            ]
        )

    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=table_rows,
        colLabels=["condition", "mean_tox", "max_tox", "pct_above_0.5"],
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.35)

    # Paired effect size across seeds: baseline vs joint.
    baseline_seed_means = mean_turn4_by_seed(data["none"])
    joint_seed_means = mean_turn4_by_seed(data["joint"])
    overlap = sorted(set(baseline_seed_means.keys()) & set(joint_seed_means.keys()))
    paired_reduction = np.asarray(
        [baseline_seed_means[sid] - joint_seed_means[sid] for sid in overlap],
        dtype=float,
    )
    if paired_reduction.size > 0:
        mean_red, lo_red, hi_red = bootstrap_mean_ci(
            paired_reduction, n_boot=args.n_boot, seed=args.rng_seed + 999
        )
        effect_text = (
            f"Paired reduction (baseline - joint), turn-4 mean over rollouts: "
            f"n={paired_reduction.size}, mean={mean_red:.4f}, 95% CI=[{lo_red:.4f}, {hi_red:.4f}]"
        )
    else:
        effect_text = "Paired reduction (baseline - joint): no overlapping seed IDs."
    fig.text(0.01, 0.01, effect_text, fontsize=10)

    out_pdf = args.out_pdf
    if not os.path.isabs(out_pdf):
        out_pdf = os.path.join(os.path.dirname(__file__), out_pdf)
    ensure_parent(out_pdf)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    fig.savefig(out_pdf)
    plt.close(fig)

    print(effect_text)
    print(f"[OK] wrote {out_pdf}")


if __name__ == "__main__":
    main()

