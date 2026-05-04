import argparse
import os
import subprocess
import sys
from typing import List


CONTEXT_MODES: List[str] = ["full", "parent_only", "seed_only"]
A1_MODES: List[str] = ["toxic", "neutral"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_source", choices=["csv", "reddit_jsonl"], default="reddit_jsonl")
    parser.add_argument("--thread_csv", default="../data/threads_data.csv")
    parser.add_argument("--reddit_jsonl", default=None)
    parser.add_argument("--seed_strategy", choices=["root", "random"], default="root")
    parser.add_argument("--reddit_require_max_depth", type=int, default=-1)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--n_seeds", type=int, default=200)
    parser.add_argument("--toxic_intensity", choices=["mild", "medium", "strong"], default="strong")
    parser.add_argument("--intervention_position", choices=["pos1", "pos2", "pos3"], default="pos1")
    parser.add_argument("--time_policy", choices=["default", "early", "late"], default="default")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--data_dir", default="../data/context_ablation")
    parser.add_argument("--summary_dir", default="../data/context_ablation")
    parser.add_argument("--use_uiuc_chat_api", action="store_true")
    parser.add_argument("--course_name", default="Agent-leak")
    parser.add_argument("--uiuc_api_key_env", default="UIUC_CHAT_API_KEY")
    args = parser.parse_args()

    if args.seed_source == "reddit_jsonl" and not args.reddit_jsonl:
        raise ValueError("--reddit_jsonl is required when --seed_source=reddit_jsonl")

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.summary_dir, exist_ok=True)

    script_path = os.path.join(os.path.dirname(__file__), "..", "src", "run_influence_baseline.py")
    script_path = os.path.abspath(script_path)

    for context_mode in CONTEXT_MODES:
        for a1_mode in A1_MODES:
            out_jsonl = os.path.abspath(
                os.path.join(args.data_dir, f"threads_{context_mode}_{a1_mode}.jsonl")
            )
            out_summary = os.path.abspath(
                os.path.join(args.summary_dir, f"summary_{context_mode}_{a1_mode}.json")
            )

            cmd = [
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
                "--toxic_intensity",
                args.toxic_intensity,
                "--intervention_position",
                args.intervention_position,
                "--time_policy",
                args.time_policy,
                "--random_seed",
                str(args.random_seed),
                "--context_mode",
                context_mode,
                "--modes",
                a1_mode,
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

            print(f"\n[RUN] context_mode={context_mode}, a1_mode={a1_mode}")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            print(f"[OK] wrote {out_jsonl}")


if __name__ == "__main__":
    main()


"""
python experiments/run_context_ablation.py \
    --seed_source reddit_jsonl \
    --reddit_jsonl agent/data/reddit/extracted/politics_seedA_BCD_chains_detoxify.jsonl \
    --model llama3_8b \
    --n_seeds 20 

python ../analysis/plot_context_ablation.py \
    --data_dir ../data/context_ablation/llama3_8b \
    --out_pdf ../figures/context_ablation_llama3_8b.pdf
"""