"""
Extract contaminated and neutral memory states from none-condition rollouts.

Usage:
    python scripts/extract_memory_states.py \
        --rollouts data/memory_llama/rollouts/influence_memory_threads_rollout_000.jsonl \
                   data/memory_llama/rollouts/influence_memory_threads_rollout_001.jsonl \
        --out_dir data/memory_llama/smoking_gun
"""

import argparse
import json
import os
import statistics
import sys


def load_records(paths):
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def extract_states(records, mode, turns, tox_min=None, tox_max=None):
    """
    Yield one state dict per (record, turn) combination that passes filters.
    De-duplicate by (seed_id, turn) — take the first match.
    """
    seen = set()
    states = []
    for r in records:
        if r.get("mode") != mode:
            continue
        seed_id = r.get("seed_id", "")
        rollout_id = r.get("rollout_id", 0)
        seed = r.get("seed", {}) or {}
        seed_text = seed.get("seed_text") or seed.get("title") or ""

        messages = r.get("messages", []) or []
        # Build a turn->toxicity map from messages for raw_msg_tox_at_turn
        msg_tox_by_turn = {}
        for m in messages:
            t = m.get("turn")
            d = (m.get("detoxify") or {})
            if t is not None and d:
                msg_tox_by_turn[t] = d.get("toxicity", 0.0)

        for entry in r.get("memory_history", []) or []:
            turn = entry.get("turn")
            if turn not in turns:
                continue
            memory_text = entry.get("memory_after") or ""
            if len(memory_text) <= 20:
                continue
            mem_detox = entry.get("memory_detoxify") or {}
            memory_tox = mem_detox.get("toxicity", 0.0)
            if tox_min is not None and memory_tox <= tox_min:
                continue
            if tox_max is not None and memory_tox >= tox_max:
                continue

            key = (seed_id, turn)
            if key in seen:
                continue
            seen.add(key)

            states.append({
                "seed_id": seed_id,
                "rollout_id": rollout_id,
                "mode": mode,
                "turn": turn,
                "memory_text": memory_text,
                "memory_tox": memory_tox,
                "raw_msg_tox_at_turn": msg_tox_by_turn.get(turn),
                "seed_text": seed_text,
            })
    return states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouts",
        nargs="+",
        default=[
            "data/memory_llama/rollouts/influence_memory_threads_rollout_000.jsonl",
            "data/memory_llama/rollouts/influence_memory_threads_rollout_001.jsonl",
        ],
    )
    parser.add_argument("--out_dir", default="data/memory_llama/smoking_gun")
    parser.add_argument("--toxic_tox_thresh", type=float, default=0.005,
                        help="Minimum memory_tox for toxic states")
    parser.add_argument("--neutral_tox_max", type=float, default=0.002,
                        help="Maximum memory_tox for neutral states")
    parser.add_argument("--min_toxic_n", type=int, default=20,
                        help="Minimum desired toxic states before lowering threshold")
    args = parser.parse_args()

    records = load_records(args.rollouts)
    print(f"Loaded {len(records)} rollout records from {len(args.rollouts)} files.")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Toxic states ---
    toxic_thresh = args.toxic_tox_thresh
    toxic_states = extract_states(records, mode="toxic", turns={1},
                                  tox_min=toxic_thresh)

    if len(toxic_states) < args.min_toxic_n:
        print(
            f"WARNING: only {len(toxic_states)} toxic memory states found above "
            f"threshold {toxic_thresh}.\n"
            f"Lowering threshold to 0.002 and including turns 1-3."
        )
        toxic_thresh = 0.002
        toxic_states = extract_states(records, mode="toxic", turns={1, 2, 3},
                                      tox_min=toxic_thresh)

    if len(toxic_states) < args.min_toxic_n:
        print(
            f"WARNING: still only {len(toxic_states)} toxic states after lowering threshold. "
            f"Proceeding with available data."
        )

    # --- Neutral states ---
    neutral_states = extract_states(records, mode="neutral", turns={1},
                                    tox_max=args.neutral_tox_max)

    # --- Save ---
    toxic_out = os.path.join(args.out_dir, "toxic_memory_states.jsonl")
    neutral_out = os.path.join(args.out_dir, "neutral_memory_states.jsonl")

    with open(toxic_out, "w", encoding="utf-8") as f:
        for s in toxic_states:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(neutral_out, "w", encoding="utf-8") as f:
        for s in neutral_states:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # --- Print inventory ---
    def stats(vals):
        if not vals:
            return "N/A"
        return f"mean={statistics.mean(vals):.4f}, min={min(vals):.4f}, max={max(vals):.4f}"

    toxic_tox_vals = [s["memory_tox"] for s in toxic_states]
    neutral_tox_vals = [s["memory_tox"] for s in neutral_states]

    print("\nExtracted memory states:")
    print(f"  toxic:   {len(toxic_states):3d} states  ({stats(toxic_tox_vals)})")
    print(f"  neutral: {len(neutral_states):3d} states  ({stats(neutral_tox_vals)})")
    print(f"\nSaved to:")
    print(f"  {toxic_out}")
    print(f"  {neutral_out}")


if __name__ == "__main__":
    main()
