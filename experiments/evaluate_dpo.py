import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Condition:
    name: str
    use_dpo: bool
    memory_mode: str
    memory_sanitize: str
    read_sanitize: str
    write_gate: str


CONDITIONS: List[Condition] = [
    Condition("base_no_intervention", False, "none", "none", "none", "none"),
    Condition("base_state_control", False, "none", "none", "summarize", "rewrite"),
    Condition("base_memory_unlearn", False, "memory", "rewrite", "none", "none"),
    Condition("dpo_no_intervention", True, "none", "none", "none", "none"),
    Condition("dpo_state_control", True, "none", "none", "summarize", "rewrite"),
    Condition("dpo_memory_unlearn", True, "memory", "rewrite", "none", "none"),
    Condition("dpo_full", True, "memory", "rewrite", "summarize", "rewrite"),
]


def maybe_merge_adapter(adapter_path: str, merged_out: str) -> str:
    if os.path.exists(os.path.join(merged_out, "config.json")):
        return merged_out

    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    import torch

    os.makedirs(merged_out, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_out)

    # adapter tokenizer should inherit from base model tokenizer in config.
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    tokenizer.save_pretrained(merged_out)
    return merged_out


def mean_turn4_toxicity(path: str) -> float:
    vals: List[float] = []
    if not os.path.exists(path):
        return 0.0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_turn = ((rec.get("scores") or {}).get("detoxify_by_turn") or {})
            t4 = by_turn.get("4")
            if isinstance(t4, dict):
                t4 = t4.get("toxicity")
            try:
                vals.append(float(t4))
            except (TypeError, ValueError):
                continue
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_source", choices=["csv", "reddit_jsonl"], default="reddit_jsonl")
    parser.add_argument("--thread_csv", default="../data/threads_data.csv")
    parser.add_argument("--reddit_jsonl", default=None)
    parser.add_argument("--seed_strategy", choices=["root", "random"], default="root")
    parser.add_argument("--reddit_require_max_depth", type=int, default=-1)
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dpo_model", default=None, help="Merged DPO model path or model id.")
    parser.add_argument("--dpo_adapter_path", default="models/dpo_lora_llama8b")
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--merged_model_out", default="models/dpo_lora_llama8b_merged")
    parser.add_argument("--n_seeds", type=int, default=200)
    parser.add_argument("--n_rollouts", type=int, default=3)
    parser.add_argument("--base_random_seed", type=int, default=12345)
    parser.add_argument("--toxic_intensity", choices=["mild", "medium", "strong"], default="strong")
    parser.add_argument("--intervention_position", choices=["pos1", "pos2", "pos3"], default="pos1")
    parser.add_argument("--time_policy", choices=["default", "early", "late"], default="default")
    parser.add_argument("--sanitize_threshold", type=float, default=0.5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data/dpo_eval")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--use_uiuc_chat_api", action="store_true")
    parser.add_argument("--course_name", default="Agent-leak")
    parser.add_argument("--uiuc_api_key_env", default="UIUC_CHAT_API_KEY")
    args = parser.parse_args()

    if args.seed_source == "reddit_jsonl" and not args.reddit_jsonl:
        raise ValueError("--reddit_jsonl is required when --seed_source=reddit_jsonl")

    dpo_model = args.dpo_model
    if dpo_model is None:
        if args.merge_adapter:
            dpo_model = maybe_merge_adapter(args.dpo_adapter_path, args.merged_model_out)
        else:
            dpo_model = args.dpo_adapter_path

    os.makedirs(args.out_dir, exist_ok=True)
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "run_influence_baseline.py"))

    aggregate: List[Dict[str, object]] = []
    for idx, cond in enumerate(CONDITIONS):
        model_name = dpo_model if cond.use_dpo else args.base_model
        out_jsonl = os.path.abspath(os.path.join(args.out_dir, f"threads_{cond.name}.jsonl"))
        out_summary = os.path.abspath(os.path.join(args.out_dir, f"summary_{cond.name}.json"))

        if os.path.exists(out_jsonl) and (not args.force_rerun):
            print(f"[SKIP] {cond.name}: already exists at {out_jsonl}")
            aggregate.append(
                {
                    "condition": cond.name,
                    "model": model_name,
                    "mean_turn4_toxicity": mean_turn4_toxicity(out_jsonl),
                    "out_jsonl": out_jsonl,
                    "skipped_existing": True,
                }
            )
            continue

        cond_seed = int(args.base_random_seed) + idx * int(args.n_seeds) * int(args.n_rollouts)
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
            str(model_name),
            "--n_seeds",
            str(args.n_seeds),
            "--n_rollouts",
            str(args.n_rollouts),
            "--base_random_seed",
            str(cond_seed),
            "--toxic_intensity",
            args.toxic_intensity,
            "--intervention_position",
            args.intervention_position,
            "--time_policy",
            args.time_policy,
            "--sanitize_threshold",
            str(args.sanitize_threshold),
            "--memory_mode",
            cond.memory_mode,
            "--memory_sanitize",
            cond.memory_sanitize,
            "--read_sanitize",
            cond.read_sanitize,
            "--write_gate",
            cond.write_gate,
            "--random_seed",
            str(args.random_seed),
            "--modes",
            "toxic",
            "--compute_toxicity",
            "--rollout_output_mode",
            "combined",
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

        print(f"\n[RUN] condition={cond.name} model={model_name}")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        aggregate.append(
            {
                "condition": cond.name,
                "model": model_name,
                "mean_turn4_toxicity": mean_turn4_toxicity(out_jsonl),
                "out_jsonl": out_jsonl,
                "skipped_existing": False,
            }
        )

    aggregate_path = os.path.abspath(os.path.join(args.out_dir, "dpo_eval_summary.json"))
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {aggregate_path}")


if __name__ == "__main__":
    main()
