import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError as exc:
    raise ImportError(
        "scipy is required for Wilcoxon signed-rank test. Install with `pip install scipy`."
    ) from exc


CONTEXT_MODES = ["full", "parent_only", "seed_only"]
A1_MODES = ["neutral", "toxic"]
TURNS = [2, 3, 4]
COLORS = {"neutral": "#1f77b4", "toxic": "#ff7f0e"}


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


def turn_toxicity_from_record(record: Dict[str, Any], turn: int) -> Optional[float]:
    by_turn = ((record.get("scores") or {}).get("detoxify_by_turn") or {})
    turn_obj = by_turn.get(str(turn), {})
    if not isinstance(turn_obj, dict):
        return None
    val = turn_obj.get("toxicity")
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 95.0,
    rng_seed: int = 123,
) -> Tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(rng_seed)
    n = values.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[i] = float(np.mean(sample))
    mean = float(np.mean(values))
    lo = float(np.percentile(boots, (100.0 - ci) / 2.0))
    hi = float(np.percentile(boots, 100.0 - (100.0 - ci) / 2.0))
    return mean, lo, hi


def collect_turn_values(records: List[Dict[str, Any]], turn: int) -> np.ndarray:
    vals: List[float] = []
    for rec in records:
        val = turn_toxicity_from_record(rec, turn=turn)
        if val is not None:
            vals.append(val)
    return np.asarray(vals, dtype=float)


def paired_turn_values_by_seed(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    turn: int,
) -> Tuple[np.ndarray, np.ndarray]:
    a_map: Dict[str, float] = {}
    b_map: Dict[str, float] = {}
    for rec in records_a:
        sid = str(rec.get("seed_id", ""))
        if not sid:
            continue
        val = turn_toxicity_from_record(rec, turn=turn)
        if val is not None:
            a_map[sid] = val
    for rec in records_b:
        sid = str(rec.get("seed_id", ""))
        if not sid:
            continue
        val = turn_toxicity_from_record(rec, turn=turn)
        if val is not None:
            b_map[sid] = val

    overlap = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not overlap:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    a = np.asarray([a_map[sid] for sid in overlap], dtype=float)
    b = np.asarray([b_map[sid] for sid in overlap], dtype=float)
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/context_ablation")
    parser.add_argument("--out_pdf", default="figures/context_ablation.pdf")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--rng_seed", type=int, default=123)
    args = parser.parse_args()

    datasets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for context_mode in CONTEXT_MODES:
        for a1_mode in A1_MODES:
            path = os.path.join(args.data_dir, f"threads_{context_mode}_{a1_mode}.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing context ablation file: {path}")
            datasets[(context_mode, a1_mode)] = load_jsonl(path)

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.4], hspace=0.45, wspace=0.28)

    # Top row: 3 context panels, toxicity by turn.
    for i, context_mode in enumerate(CONTEXT_MODES):
        ax = fig.add_subplot(gs[0, i])
        for a1_mode in A1_MODES:
            means: List[float] = []
            lo: List[float] = []
            hi: List[float] = []
            recs = datasets[(context_mode, a1_mode)]
            for turn in TURNS:
                vals = collect_turn_values(recs, turn=turn)
                m, l, h = bootstrap_mean_ci(
                    vals, n_boot=args.n_boot, rng_seed=args.rng_seed + turn + i
                )
                means.append(m)
                lo.append(l)
                hi.append(h)

            x = np.asarray(TURNS, dtype=float)
            y = np.asarray(means, dtype=float)
            yl = np.asarray(lo, dtype=float)
            yh = np.asarray(hi, dtype=float)
            ax.plot(x, y, marker="o", linewidth=2, color=COLORS[a1_mode], label=a1_mode)
            ax.fill_between(x, yl, yh, color=COLORS[a1_mode], alpha=0.22)

        ax.set_title(f"context={context_mode}")
        ax.set_xlabel("turn")
        if i == 0:
            ax.set_ylabel("mean detoxify toxicity")
        ax.set_xticks(TURNS)
        ax.grid(alpha=0.25)
        if i == 2:
            ax.legend(frameon=False, loc="upper left")

    # Bottom row: turn-4 bars for all six conditions.
    ax_bar = fig.add_subplot(gs[1, :])
    labels: List[str] = []
    means: List[float] = []
    err_low: List[float] = []
    err_high: List[float] = []
    bar_colors: List[str] = []

    for context_mode in CONTEXT_MODES:
        for a1_mode in A1_MODES:
            vals = collect_turn_values(datasets[(context_mode, a1_mode)], turn=4)
            mean, lo, hi = bootstrap_mean_ci(vals, n_boot=args.n_boot, rng_seed=args.rng_seed)
            labels.append(f"{context_mode}\n{a1_mode}")
            means.append(mean)
            err_low.append(mean - lo)
            err_high.append(hi - mean)
            bar_colors.append(COLORS[a1_mode])

    x_pos = np.arange(len(labels))
    ax_bar.bar(
        x_pos,
        means,
        color=bar_colors,
        alpha=0.9,
        yerr=np.vstack([err_low, err_high]),
        capsize=4,
    )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("turn-4 mean detoxify toxicity")
    ax_bar.set_title("Turn-4 toxicity across context modes and A1 conditions (95% bootstrap CI)")
    ax_bar.grid(axis="y", alpha=0.25)

    # Wilcoxon signed-rank test: toxic full > toxic parent_only at turn 4.
    toxic_full = datasets[("full", "toxic")]
    toxic_parent = datasets[("parent_only", "toxic")]
    x_full, x_parent = paired_turn_values_by_seed(toxic_full, toxic_parent, turn=4)
    if x_full.size == 0:
        stat_text = "Wilcoxon (full > parent_only, toxic, turn4): no overlapping seeds"
        print(stat_text)
    else:
        test_result = wilcoxon(x_full, x_parent, alternative="greater")
        delta = float(np.mean(x_full - x_parent))
        stat_text = (
            f"Wilcoxon toxic turn4 (full > parent_only): "
            f"n={x_full.size}, W={float(test_result.statistic):.3f}, "
            f"p={float(test_result.pvalue):.4g}, mean_delta={delta:.4f}"
        )
        print(stat_text)

    fig.text(0.015, 0.01, stat_text, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    out_pdf = args.out_pdf
    if not os.path.isabs(out_pdf):
        out_pdf = os.path.join(os.path.dirname(__file__), out_pdf)
    ensure_parent(out_pdf)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[OK] wrote {out_pdf}")


if __name__ == "__main__":
    main()

