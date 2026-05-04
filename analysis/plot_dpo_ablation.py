import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = [
    "base_no_intervention",
    "base_state_control",
    "base_memory_unlearn",
    "dpo_no_intervention",
    "dpo_state_control",
    "dpo_memory_unlearn",
    "dpo_full",
]

LABELS = {
    "base_no_intervention": "base_no_intervention",
    "base_state_control": "base_state_control",
    "base_memory_unlearn": "base_memory_unlearn",
    "dpo_no_intervention": "dpo_no_intervention",
    "dpo_state_control": "dpo_state_control",
    "dpo_memory_unlearn": "dpo_memory_unlearn",
    "dpo_full": "dpo_full",
}

TURNS = [1, 2, 3, 4]


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
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
        item = item.get("toxicity")
    try:
        return float(item)
    except (TypeError, ValueError):
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


def mean_final_toxicity(records: List[Dict[str, Any]]) -> float:
    vals: List[float] = []
    for rec in records:
        v = turn_toxicity(rec, 4)
        if v is not None:
            vals.append(v)
    if not vals:
        return 0.0
    return float(np.mean(np.asarray(vals, dtype=float)))


def build_interaction_table(data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    base_no = mean_final_toxicity(data["base_no_intervention"])
    base_full = mean_final_toxicity(data["base_state_control"])
    dpo_no = mean_final_toxicity(data["dpo_no_intervention"])
    dpo_full = mean_final_toxicity(data["dpo_full"])

    base_gain = base_no - base_full
    dpo_gain = dpo_no - dpo_full
    interaction = dpo_gain - base_gain
    return {
        "base_no_intervention": base_no,
        "base_full_intervention": base_full,
        "dpo_no_intervention": dpo_no,
        "dpo_full_intervention": dpo_full,
        "base_gain": base_gain,
        "dpo_gain": dpo_gain,
        "interaction_effect_diff_in_diff": interaction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/dpo_eval")
    parser.add_argument("--out_pdf", default="analysis/figures/dpo_ablation.pdf")
    parser.add_argument("--out_table_json", default="data/dpo_eval/dpo_interaction_table.json")
    parser.add_argument("--n_boot", type=int, default=1500)
    parser.add_argument("--rng_seed", type=int, default=123)
    args = parser.parse_args()

    data: Dict[str, List[Dict[str, Any]]] = {}
    for cond in CONDITIONS:
        path = os.path.join(args.data_dir, f"threads_{cond}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing condition file: {path}")
        data[cond] = load_jsonl(path)

    colors = {
        "base_no_intervention": "#4c78a8",
        "base_state_control": "#f58518",
        "base_memory_unlearn": "#54a24b",
        "dpo_no_intervention": "#e45756",
        "dpo_state_control": "#72b7b2",
        "dpo_memory_unlearn": "#b279a2",
        "dpo_full": "#ff9da6",
    }

    fig = plt.figure(figsize=(14.0, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.34, wspace=0.28)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_line = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    # Panel A: final-turn bar chart.
    means: List[float] = []
    err_lo: List[float] = []
    err_hi: List[float] = []
    for i, cond in enumerate(CONDITIONS):
        vals = [v for v in (turn_toxicity(r, 4) for r in data[cond]) if v is not None]
        arr = np.asarray(vals, dtype=float) if vals else np.asarray([], dtype=float)
        m, lo, hi = bootstrap_mean_ci(arr, n_boot=args.n_boot, seed=args.rng_seed + i)
        means.append(m)
        err_lo.append(m - lo)
        err_hi.append(hi - m)

    x = np.arange(len(CONDITIONS))
    ax_bar.bar(
        x,
        means,
        yerr=np.vstack([np.asarray(err_lo), np.asarray(err_hi)]),
        capsize=4,
        color=[colors[c] for c in CONDITIONS],
        alpha=0.9,
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([LABELS[c] for c in CONDITIONS], rotation=20, ha="right")
    ax_bar.set_ylabel("final turn toxicity (turn 4)")
    ax_bar.set_title("DPO ablation: final-turn toxicity")
    ax_bar.grid(axis="y", alpha=0.25)

    # Panel B: trajectory by condition.
    for i, cond in enumerate(CONDITIONS):
        turn_means: List[float] = []
        turn_los: List[float] = []
        turn_his: List[float] = []
        for t in TURNS:
            vals = [v for v in (turn_toxicity(r, t) for r in data[cond]) if v is not None]
            arr = np.asarray(vals, dtype=float) if vals else np.asarray([], dtype=float)
            m, lo, hi = bootstrap_mean_ci(
                arr,
                n_boot=args.n_boot,
                seed=args.rng_seed + 100 + i * 10 + t,
            )
            turn_means.append(m)
            turn_los.append(lo)
            turn_his.append(hi)

        xt = np.asarray(TURNS, dtype=float)
        ym = np.asarray(turn_means, dtype=float)
        yl = np.asarray(turn_los, dtype=float)
        yh = np.asarray(turn_his, dtype=float)
        ax_line.plot(xt, ym, marker="o", linewidth=2, color=colors[cond], label=LABELS[cond])
        ax_line.fill_between(xt, yl, yh, color=colors[cond], alpha=0.14)

    ax_line.set_xticks(TURNS)
    ax_line.set_xlabel("turn")
    ax_line.set_ylabel("mean toxicity")
    ax_line.set_title("Toxicity trajectory across generated turns")
    ax_line.grid(alpha=0.25)
    ax_line.legend(frameon=False, fontsize=8, ncol=2)

    # Panel C: 2x2 summary table + interaction effect.
    interaction = build_interaction_table(data)
    table_rows = [
        ["base", f"{interaction['base_no_intervention']:.4f}", f"{interaction['base_full_intervention']:.4f}", f"{interaction['base_gain']:.4f}"],
        ["dpo", f"{interaction['dpo_no_intervention']:.4f}", f"{interaction['dpo_full_intervention']:.4f}", f"{interaction['dpo_gain']:.4f}"],
    ]
    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=table_rows,
        colLabels=["model", "no_intervention", "full_intervention", "gain (no-full)"],
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.5)

    interaction_text = f"Interaction effect (difference-in-differences): {interaction['interaction_effect_diff_in_diff']:.4f}"
    fig.text(0.01, 0.01, interaction_text, fontsize=11)

    out_pdf = args.out_pdf
    if not os.path.isabs(out_pdf):
        out_pdf = os.path.abspath(out_pdf)
    ensure_parent(out_pdf)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    fig.savefig(out_pdf)
    plt.close(fig)

    out_table_json = args.out_table_json
    if not os.path.isabs(out_table_json):
        out_table_json = os.path.abspath(out_table_json)
    ensure_parent(out_table_json)
    with open(out_table_json, "w", encoding="utf-8") as f:
        json.dump(interaction, f, ensure_ascii=False, indent=2)

    print(interaction_text)
    print(f"[OK] wrote {out_pdf}")
    print(f"[OK] wrote {out_table_json}")


if __name__ == "__main__":
    main()
