import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class Condition:
    name: str
    read_sanitize: str
    write_gate: str
    prompt_safety: bool


CONDITIONS: List[Condition] = [
    Condition("none", "none", "none", False),
    Condition("read_only", "summarize", "none", False),
    Condition("write_only", "none", "rewrite", False),
    Condition("joint", "summarize", "rewrite", False),
    Condition("prompt_safety", "none", "none", True),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_source", choices=["csv", "reddit_jsonl"], default="reddit_jsonl")
    parser.add_argument("--thread_csv", default="../data/threads_data.csv")
    parser.add_argument("--reddit_jsonl", default=None)
    parser.add_argument("--seed_strategy", choices=["root", "random"], default="root")
    parser.add_argument("--reddit_require_max_depth", type=int, default=-1)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--n_seeds", type=int, default=200)
    parser.add_argument("--n_rollouts", type=int, default=3)
    parser.add_argument("--base_random_seed", type=int, default=12345)
    parser.add_argument("--toxic_intensity", choices=["mild", "medium", "strong"], default="strong")
    parser.add_argument("--intervention_position", choices=["pos1", "pos2", "pos3"], default="pos1")
    parser.add_argument("--time_policy", choices=["default", "early", "late"], default="default")
    parser.add_argument("--sanitize_threshold", type=float, default=0.5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--data_dir", default="../data/sanitization_ablation")
    parser.add_argument("--summary_dir", default="../data/sanitization_ablation")
    parser.add_argument("--use_uiuc_chat_api", action="store_true")
    parser.add_argument("--course_name", default="Agent-leak")
    parser.add_argument("--uiuc_api_key_env", default="UIUC_CHAT_API_KEY")
    args = parser.parse_args()

    if args.seed_source == "reddit_jsonl" and not args.reddit_jsonl:
        raise ValueError("--reddit_jsonl is required when --seed_source=reddit_jsonl")

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.summary_dir, exist_ok=True)

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "run_influence_baseline.py"))

    for idx, cond in enumerate(CONDITIONS):
        out_jsonl = os.path.abspath(os.path.join(args.data_dir, f"threads_{cond.name}.jsonl"))
        out_summary = os.path.abspath(os.path.join(args.summary_dir, f"summary_{cond.name}.json"))
        cond_base_seed = int(args.base_random_seed) + idx * int(args.n_seeds) * int(args.n_rollouts)
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
            args.model,
            "--n_seeds",
            str(args.n_seeds),
            "--n_rollouts",
            str(args.n_rollouts),
            "--base_random_seed",
            str(cond_base_seed),
            "--toxic_intensity",
            args.toxic_intensity,
            "--intervention_position",
            args.intervention_position,
            "--time_policy",
            args.time_policy,
            "--sanitize_threshold",
            str(args.sanitize_threshold),
            "--read_sanitize",
            cond.read_sanitize,
            "--write_gate",
            cond.write_gate,
            "--random_seed",
            str(args.random_seed),
            "--modes",
            "toxic",
            "--compute_toxicity",
            "--out_jsonl",
            out_jsonl,
            "--out_summary",
            out_summary,
        ]

        if args.seed_source == "reddit_jsonl":
            cmd.extend(["--reddit_jsonl", args.reddit_jsonl])
        if cond.prompt_safety:
            cmd.append("--prompt_safety")
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

        print(f"\n[RUN] condition={cond.name}")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"[OK] wrote {out_jsonl}")


if __name__ == "__main__":
    main()

