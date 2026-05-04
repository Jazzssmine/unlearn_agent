import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Condition:
    name: str
    memory_mode: str
    memory_sanitize: str
    read_sanitize: str
    write_gate: str
    dpo: bool


CONDITIONS: List[Condition] = [
    Condition("base_transcript", "none", "none", "none", "none", False),
    Condition("base_memory", "memory", "none", "none", "none", False),
    Condition("state_only", "none", "none", "summarize", "rewrite", False),
    Condition("memory_only", "memory", "rewrite", "none", "none", False),
    Condition("memory+state", "memory", "rewrite", "summarize", "rewrite", False),
    Condition("dpo_only", "memory", "none", "none", "none", True),
    Condition("dpo+state", "memory", "none", "summarize", "rewrite", True),
    Condition("dpo+memory", "memory", "rewrite", "none", "none", True),
    Condition("full", "memory", "rewrite", "summarize", "rewrite", True),
]


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
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


def extract_turn4_toxicity(rec: Dict) -> Optional[float]:
    by_turn = ((rec.get("scores") or {}).get("detoxify_by_turn") or {})
    t4 = by_turn.get("4")
    if isinstance(t4, dict):
        val = t4.get("toxicity")
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    if t4 is not None:
        try:
            return float(t4)
        except (TypeError, ValueError):
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_source", choices=["csv", "reddit_jsonl"], default="reddit_jsonl")
    parser.add_argument("--thread_csv", default="../data/threads_data.csv")
    parser.add_argument("--reddit_jsonl", default=None)
    parser.add_argument("--seed_strategy", choices=["root", "random"], default="root")
    parser.add_argument("--reddit_require_max_depth", type=int, default=-1)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--dpo_model_path", default=None, help="If provided, DPO conditions run with this model path/id.")
    parser.add_argument("--n_seeds", type=int, default=200)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--toxic_intensity", choices=["mild", "medium", "strong"], default="strong")
    parser.add_argument("--intervention_position", choices=["pos1", "pos2", "pos3"], default="pos1")
    parser.add_argument("--time_policy", choices=["default", "early", "late"], default="default")
    parser.add_argument("--sanitize_threshold", type=float, default=0.5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--data_dir", default="../data/full_ablation")
    parser.add_argument("--use_uiuc_chat_api", action="store_true")
    parser.add_argument("--course_name", default="Agent-leak")
    parser.add_argument("--uiuc_api_key_env", default="UIUC_CHAT_API_KEY")
    args = parser.parse_args()

    if args.seed_source == "reddit_jsonl" and not args.reddit_jsonl:
        raise ValueError("--reddit_jsonl is required when --seed_source=reddit_jsonl")

    os.makedirs(args.data_dir, exist_ok=True)

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "run_influence_baseline.py"))
    aggregate_rows: List[Dict[str, object]] = []

    for cond in CONDITIONS:
        if cond.dpo and not args.dpo_model_path:
            print(f"[SKIP] {cond.name}: dpo_model_path not provided.")
            continue

        cond_dir = os.path.abspath(os.path.join(args.data_dir, cond.name))
        os.makedirs(cond_dir, exist_ok=True)
        run_model = args.dpo_model_path if cond.dpo else args.model

        for rollout_idx in range(args.rollouts):
            run_seed = int(args.random_seed) + rollout_idx
            out_jsonl = os.path.abspath(os.path.join(cond_dir, f"threads_rollout_{rollout_idx:02d}.jsonl"))
            out_summary = os.path.abspath(os.path.join(cond_dir, f"summary_rollout_{rollout_idx:02d}.json"))
            cmd: List[str] = [
                sys.executable,
                script_path,
                "--seed_source",
                args.seed_source,
                "--thread_csv",
                args.thread_csv,
                "--seed_strategy",
                args.seed_strategy,
                "--reddit_require_max_depth",
                str(args.reddit_require_max_depth),
                "--model",
                str(run_model),
                "--n_seeds",
                str(args.n_seeds),
                "--toxic_intensity",
                args.toxic_intensity,
                "--intervention_position",
                args.intervention_position,
                "--time_policy",
                args.time_policy,
                "--random_seed",
                str(run_seed),
                "--modes",
                "toxic",
                "--memory_mode",
                cond.memory_mode,
                "--memory_sanitize",
                cond.memory_sanitize,
                "--read_sanitize",
                cond.read_sanitize,
                "--write_gate",
                cond.write_gate,
                "--sanitize_threshold",
                str(args.sanitize_threshold),
                "--compute_toxicity",
                "--out_jsonl",
                out_jsonl,
                "--out_summary",
                out_summary,
            ]
            if args.seed_source == "reddit_jsonl":
                cmd.extend(["--reddit_jsonl", args.reddit_jsonl])
            if args.use_uiuc_chat_api:
                cmd.extend(
                    [
                        "--use_uiuc_chat_api",
                        "--course_name",
                        args.course_name,
                        "--uiuc_api_key_env",
                        args.uiuc_api_key_env,
                    ]
                )

            print(f"\n[RUN] condition={cond.name} rollout={rollout_idx} model={run_model}")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)

            rows = load_jsonl(out_jsonl)
            vals = [v for v in (extract_turn4_toxicity(r) for r in rows) if v is not None]
            mean_t4 = float(np.mean(np.asarray(vals, dtype=float))) if vals else 0.0
            aggregate_rows.append(
                {
                    "condition": cond.name,
                    "rollout": rollout_idx,
                    "model": run_model,
                    "n_records": len(rows),
                    "turn4_mean_toxicity": mean_t4,
                    "out_jsonl": out_jsonl,
                }
            )

    aggregate_path = os.path.abspath(os.path.join(args.data_dir, "full_ablation_summary_table.json"))
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_rows, f, ensure_ascii=False, indent=2)

    by_condition: Dict[str, List[float]] = {}
    for row in aggregate_rows:
        by_condition.setdefault(str(row["condition"]), []).append(float(row["turn4_mean_toxicity"]))

    if by_condition:
        labels = list(by_condition.keys())
        means = [float(np.mean(np.asarray(by_condition[k], dtype=float))) for k in labels]
        stds = [float(np.std(np.asarray(by_condition[k], dtype=float))) for k in labels]
        x = np.arange(len(labels))
        plt.figure(figsize=(12.2, 5.4))
        plt.bar(x, means, yerr=stds, capsize=4, color="#4c78a8", alpha=0.92)
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.ylabel("turn-4 toxicity (mean +/- std across rollouts)")
        plt.title("Full Ablation Comparison (A1=toxic)")
        plt.grid(axis="y", alpha=0.25)
        fig_path = os.path.abspath(os.path.join(args.data_dir, "full_ablation_comparison.pdf"))
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        print(f"[OK] wrote {fig_path}")

    print(f"[OK] wrote {aggregate_path}")


if __name__ == "__main__":
    main()
