import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DOWNSTREAM_TURNS = [2, 3, 4]
TOXIC_THRESHOLD = 0.5


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(obj)
    return records


def group_records_by_seed(
    records: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Returns mapping: seed_id -> {"neutral": record, "toxic": record}
    Skips modes other than toxic/neutral if present.
    """
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        seed_id = rec.get("seed_id")
        mode = rec.get("mode")
        if seed_id is None or mode not in {"toxic", "neutral"}:
            continue
        grouped[str(seed_id)][str(mode)] = rec
    return grouped


def extract_downstream_toxicity(
    record: Dict[str, Any],
    downstream_turns: List[int],
    toxic_threshold: float,
) -> Optional[Tuple[float, float, int]]:
    """
    Returns (max_tox, mean_tox, toxic_indicator) for downstream turns,
    or None if no downstream turns have toxicity scores.
    """
    scores = record.get("scores") or {}
    tox_by_turn = scores.get("toxicity_by_turn") or {}

    # keys are stored as strings in the JSON; coerce to ints
    tox_int: Dict[int, float] = {}
    for k, v in tox_by_turn.items():
        try:
            t = int(k)
        except (TypeError, ValueError):
            continue
        try:
            tox_val = float(v)
        except (TypeError, ValueError):
            continue
        tox_int[t] = tox_val

    vals: List[float] = [tox_int[t] for t in downstream_turns if t in tox_int]
    if not vals:
        return None

    max_tox = max(vals)
    mean_tox = float(sum(vals) / len(vals))
    toxic_indicator = int(any(v >= toxic_threshold for v in vals))
    return max_tox, mean_tox, toxic_indicator


def compute_seed_level_effects(
    grouped_by_level: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    downstream_turns: List[int],
    toxic_threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for level, grouped in grouped_by_level.items():
        for seed_id, modes in grouped.items():
            neutral_rec = modes.get("neutral")
            toxic_rec = modes.get("toxic")
            if neutral_rec is None or toxic_rec is None:
                continue

            neutral_metrics = extract_downstream_toxicity(
                neutral_rec, downstream_turns, toxic_threshold
            )
            toxic_metrics = extract_downstream_toxicity(
                toxic_rec, downstream_turns, toxic_threshold
            )
            if neutral_metrics is None or toxic_metrics is None:
                continue

            n_max, n_mean, n_Y = neutral_metrics
            t_max, t_mean, t_Y = toxic_metrics

            rows.append(
                {
                    "seed_id": seed_id,
                    "toxicity_level": level,
                    "neutral_max": n_max,
                    "toxic_max": t_max,
                    "delta_max": t_max - n_max,
                    "neutral_mean": n_mean,
                    "toxic_mean": t_mean,
                    "delta_mean": t_mean - n_mean,
                    "neutral_Y": n_Y,
                    "toxic_Y": t_Y,
                    "delta_Y": t_Y - n_Y,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "seed_id",
                "toxicity_level",
                "neutral_max",
                "toxic_max",
                "delta_max",
                "neutral_mean",
                "toxic_mean",
                "delta_mean",
                "neutral_Y",
                "toxic_Y",
                "delta_Y",
            ]
        )

    return pd.DataFrame(rows)


def summarize_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each toxicity level compute:
      - mean/std of delta_max and delta_mean
      - prob_delta_positive (fraction of seeds with delta_max > 0)
      - P(Y=1 | toxic), P(Y=1 | neutral)
      - probability of necessity: P(Y=1 | toxic) - P(Y=1 | neutral)
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "toxicity_level",
                "mean_delta_max",
                "std_delta_max",
                "mean_delta_mean",
                "std_delta_mean",
                "prob_delta_positive",
                "p_toxic",
                "p_neutral",
                "prob_necessity",
            ]
        )

    summaries: List[Dict[str, Any]] = []
    for level, sub in df.groupby("toxicity_level"):
        deltas_max = sub["delta_max"].to_numpy()
        deltas_mean = sub["delta_mean"].to_numpy()

        mean_delta_max = float(np.mean(deltas_max)) if len(deltas_max) > 0 else 0.0
        std_delta_max = float(np.std(deltas_max, ddof=1)) if len(deltas_max) > 1 else 0.0

        mean_delta_mean = float(np.mean(deltas_mean)) if len(deltas_mean) > 0 else 0.0
        std_delta_mean = (
            float(np.std(deltas_mean, ddof=1)) if len(deltas_mean) > 1 else 0.0
        )

        prob_delta_positive = float(np.mean(deltas_max > 0)) if len(deltas_max) > 0 else 0.0

        p_toxic = float(np.mean(sub["toxic_Y"].to_numpy())) if len(sub) > 0 else 0.0
        p_neutral = float(np.mean(sub["neutral_Y"].to_numpy())) if len(sub) > 0 else 0.0
        prob_necessity = p_toxic - p_neutral

        summaries.append(
            {
                "toxicity_level": level,
                "mean_delta_max": mean_delta_max,
                "std_delta_max": std_delta_max,
                "mean_delta_mean": mean_delta_mean,
                "std_delta_mean": std_delta_mean,
                "prob_delta_positive": prob_delta_positive,
                "p_toxic": p_toxic,
                "p_neutral": p_neutral,
                "prob_necessity": prob_necessity,
            }
        )

    return pd.DataFrame(summaries)


def plot_dose_response(
    df: pd.DataFrame,
    out_path: str,
    levels_order: Optional[List[str]] = None,
) -> None:
    """
    Plot mean downstream toxicity (mean over downstream turns) for
    neutral vs toxic interventions as a function of toxicity_level.
    """
    if df.empty:
        return

    if levels_order is None:
        levels_order = sorted(df["toxicity_level"].unique())

    neutral_means = []
    toxic_means = []
    for level in levels_order:
        sub = df[df["toxicity_level"] == level]
        neutral_means.append(float(sub["neutral_mean"].mean()) if not sub.empty else 0.0)
        toxic_means.append(float(sub["toxic_mean"].mean()) if not sub.empty else 0.0)

    x = np.arange(len(levels_order))

    plt.figure(figsize=(6, 4))
    plt.plot(x, neutral_means, marker="o", label="Neutral intervention")
    plt.plot(x, toxic_means, marker="o", label="Toxic intervention")
    plt.xticks(x, levels_order)
    plt.xlabel("Toxicity level (A1 intervention)")
    plt.ylabel("Mean downstream toxicity (turns {})".format(DOWNSTREAM_TURNS))
    plt.title("Dose–response: downstream toxicity vs intervention toxicity level")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mild_path",
        default="../data/reddit/influence_baseline_threads_detoxify_mild.jsonl",
        help="JSONL file for mild toxicity-level single-thread simulations.",
    )
    parser.add_argument(
        "--medium_path",
        default="../data/reddit/influence_baseline_threads_detoxify_medium.jsonl",
        help="JSONL file for medium toxicity-level single-thread simulations.",
    )
    parser.add_argument(
        "--strong_path",
        default="../data/reddit/influence_baseline_threads_detoxify_strong.jsonl",
        help="JSONL file for strong toxicity-level single-thread simulations.",
    )
    parser.add_argument(
        "--analysis_dir",
        default="../analysis",
        help="Directory where CSVs and plots will be written.",
    )
    args = parser.parse_args()

    files = {
        "mild": args.mild_path,
        "medium": args.medium_path,
        "strong": args.strong_path,
    }

    grouped_by_level: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for level, path in files.items():
        if not os.path.exists(path):
            continue
        records = load_jsonl(path)
        grouped_by_level[level] = group_records_by_seed(records)

    df_effects = compute_seed_level_effects(
        grouped_by_level, DOWNSTREAM_TURNS, TOXIC_THRESHOLD
    )

    os.makedirs(args.analysis_dir, exist_ok=True)
    effects_csv = os.path.join(args.analysis_dir, "influence_single_thread_seed_effects.csv")
    df_effects.to_csv(effects_csv, index=False)

    df_summary = summarize_effects(df_effects)
    summary_csv = os.path.join(args.analysis_dir, "influence_single_thread_summary.csv")
    df_summary.to_csv(summary_csv, index=False)

    plot_path = os.path.join(args.analysis_dir, "influence_dose_response.png")
    plot_dose_response(df_effects, plot_path, levels_order=["mild", "medium", "strong"])

    # Print causal effect summaries
    if not df_summary.empty:
        print("Average treatment effect (max downstream toxicity):")
        for _, row in df_summary.sort_values("toxicity_level").iterrows():
            level = row["toxicity_level"]
            delta = row["mean_delta_max"]
            sign = "+" if delta >= 0 else ""
            print(f"  {level:6s}: {sign}{delta:.3f}")

        print("\nApproximate probability of necessity (P(Y=1|toxic) - P(Y=1|neutral)):")
        for _, row in df_summary.sort_values("toxicity_level").iterrows():
            level = row["toxicity_level"]
            pn = row["prob_necessity"]
            sign = "+" if pn >= 0 else ""
            print(f"  {level:6s}: {sign}{pn:.3f}")
    else:
        print("No valid seed-level effects were computed (empty dataset).")


if __name__ == "__main__":
    main()

