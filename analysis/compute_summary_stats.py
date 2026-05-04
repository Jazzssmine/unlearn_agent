import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


TARGET_TURNS = [2, 3, 4]


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


def mean_toxicity_turns_2_to_4(record: Dict[str, Any]) -> Optional[float]:
    scores = record.get("scores") or {}
    by_turn_detox = scores.get("detoxify_by_turn") or {}
    by_turn_flat = scores.get("toxicity_by_turn") or {}

    vals: List[float] = []
    for turn in TARGET_TURNS:
        det = by_turn_detox.get(str(turn))
        if isinstance(det, dict) and "toxicity" in det:
            try:
                vals.append(float(det["toxicity"]))
                continue
            except (TypeError, ValueError):
                pass
        try:
            vals.append(float(by_turn_flat.get(str(turn))))
        except (TypeError, ValueError):
            continue

    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=float)))


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, ci: float = 95.0, rng_seed: int = 123
) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(rng_seed)
    n = values.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[i] = float(np.mean(sample))
    lo = float(np.percentile(boots, (100.0 - ci) / 2.0))
    hi = float(np.percentile(boots, 100.0 - (100.0 - ci) / 2.0))
    return lo, hi


def summarize_distribution(values: np.ndarray, label: str, n_boot: int, rng_seed: int) -> Dict[str, float]:
    if values.size == 0:
        return {
            "metric": label,
            "n_seeds": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    ci_low, ci_high = bootstrap_ci(values, n_boot=n_boot, rng_seed=rng_seed)
    return {
        "metric": label,
        "n_seeds": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default="../data/influence_baseline_threads.jsonl")
    parser.add_argument("--out_csv", default="summary_stats.csv")
    parser.add_argument("--out_pdf", default="figures/effect_size_histogram.pdf")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--rng_seed", type=int, default=123)
    args = parser.parse_args()

    records = load_jsonl(args.input_jsonl)
    by_seed_mode_rollout: Dict[Tuple[str, str, int], float] = {}

    for rec in records:
        seed_id = str(rec.get("seed_id", "")).strip()
        mode = str(rec.get("mode", "")).strip()
        if mode not in {"toxic", "neutral"}:
            continue
        if not seed_id:
            continue
        rollout_raw = rec.get("rollout_id", 0)
        try:
            rollout_id = int(rollout_raw)
        except (TypeError, ValueError):
            rollout_id = 0

        mean_tox = mean_toxicity_turns_2_to_4(rec)
        if mean_tox is None:
            continue
        by_seed_mode_rollout[(seed_id, mode, rollout_id)] = mean_tox

    seed_mode_values: Dict[Tuple[str, str], List[float]] = {}
    for (seed_id, mode, _rollout_id), tox in by_seed_mode_rollout.items():
        seed_mode_values.setdefault((seed_id, mode), []).append(float(tox))

    seed_ids = sorted({seed_id for (seed_id, _mode) in seed_mode_values.keys()})
    per_seed_rows: List[Dict[str, Any]] = []
    delta_mean_vals: List[float] = []
    delta_worst_vals: List[float] = []

    for seed_id in seed_ids:
        toxic_rollouts = np.asarray(seed_mode_values.get((seed_id, "toxic"), []), dtype=float)
        neutral_rollouts = np.asarray(seed_mode_values.get((seed_id, "neutral"), []), dtype=float)
        if toxic_rollouts.size == 0 or neutral_rollouts.size == 0:
            continue

        toxic_mean = float(np.mean(toxic_rollouts))
        toxic_max = float(np.max(toxic_rollouts))
        neutral_mean = float(np.mean(neutral_rollouts))
        neutral_max = float(np.max(neutral_rollouts))

        delta_mean = toxic_mean - neutral_mean
        delta_worst = toxic_max - neutral_max

        delta_mean_vals.append(delta_mean)
        delta_worst_vals.append(delta_worst)

        per_seed_rows.append(
            {
                "seed_id": seed_id,
                "n_rollouts_toxic": int(toxic_rollouts.size),
                "n_rollouts_neutral": int(neutral_rollouts.size),
                "mean_toxicity_toxic_rollouts": toxic_mean,
                "max_toxicity_toxic_rollouts": toxic_max,
                "mean_toxicity_neutral_rollouts": neutral_mean,
                "max_toxicity_neutral_rollouts": neutral_max,
                "delta_mean": delta_mean,
                "delta_worst": delta_worst,
            }
        )

    delta_mean_arr = np.asarray(delta_mean_vals, dtype=float)
    delta_worst_arr = np.asarray(delta_worst_vals, dtype=float)
    summary_mean = summarize_distribution(
        delta_mean_arr, label="delta_mean", n_boot=args.n_boot, rng_seed=args.rng_seed
    )
    summary_worst = summarize_distribution(
        delta_worst_arr, label="delta_worst", n_boot=args.n_boot, rng_seed=args.rng_seed + 1
    )

    out_csv = args.out_csv
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(os.path.dirname(__file__), out_csv)
    ensure_parent(out_csv)

    fieldnames = [
        "seed_id",
        "n_rollouts_toxic",
        "n_rollouts_neutral",
        "mean_toxicity_toxic_rollouts",
        "max_toxicity_toxic_rollouts",
        "mean_toxicity_neutral_rollouts",
        "max_toxicity_neutral_rollouts",
        "delta_mean",
        "delta_worst",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_seed_rows:
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    if delta_mean_arr.size > 0:
        ax.hist(delta_mean_arr, bins=24, color="#1f77b4", alpha=0.85, edgecolor="white")
        ax.axvline(float(np.mean(delta_mean_arr)), color="#d62728", linestyle="--", linewidth=2, label="mean")
        ax.legend(frameon=False)
    ax.set_title("Per-seed effect size distribution")
    ax.set_xlabel("delta_mean = mean_tox(toxic) - mean_tox(neutral)")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_pdf = args.out_pdf
    if not os.path.isabs(out_pdf):
        out_pdf = os.path.join(os.path.dirname(__file__), out_pdf)
    ensure_parent(out_pdf)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(
        "[SUMMARY] delta_mean: "
        f"n={summary_mean['n_seeds']}, mean={summary_mean['mean']:.6f}, "
        f"median={summary_mean['median']:.6f}, std={summary_mean['std']:.6f}, "
        f"95%CI=({summary_mean['ci95_low']:.6f}, {summary_mean['ci95_high']:.6f})"
    )
    print(
        "[SUMMARY] delta_worst: "
        f"n={summary_worst['n_seeds']}, mean={summary_worst['mean']:.6f}, "
        f"median={summary_worst['median']:.6f}, std={summary_worst['std']:.6f}, "
        f"95%CI=({summary_worst['ci95_low']:.6f}, {summary_worst['ci95_high']:.6f})"
    )
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_pdf}")


if __name__ == "__main__":
    main()

