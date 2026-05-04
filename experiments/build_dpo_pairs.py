import argparse
import difflib
import glob
import json
import os
import sys
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from real_agents.toxicity_scorer import score_text_toxicity
from run_influence_baseline import A1_TOXIC_STRONG_SYSTEM, DOWNSTREAM_SYSTEM


def _try_load_vader():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


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


def resolve_input_files(input_pattern: str) -> List[str]:
    matches = sorted(glob.glob(input_pattern))
    if matches:
        return matches
    if os.path.exists(input_pattern):
        return [input_pattern]
    return []


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def almost_identical(a: str, b: str, threshold: float = 0.97) -> bool:
    an = normalize_text(a)
    bn = normalize_text(b)
    if not an or not bn:
        return True
    if an == bn:
        return True
    return difflib.SequenceMatcher(a=an, b=bn).ratio() >= threshold


def get_turn_message(record: Dict[str, Any], turn: int) -> Optional[Dict[str, Any]]:
    for msg in record.get("messages", []):
        if int(msg.get("turn", -1)) == int(turn):
            return msg
    return None


def _system_prompt_for_agent(agent_name: str) -> str:
    if agent_name in {"A1_toxic", "A1"}:
        return A1_TOXIC_STRONG_SYSTEM
    return DOWNSTREAM_SYSTEM


def format_prompt_like_simulator(
    toxic_record: Dict[str, Any],
    turn: int,
    agent_name: str,
) -> str:
    msgs = sorted(
        toxic_record.get("messages", []),
        key=lambda x: int(x.get("turn", 0)),
    )
    context_msgs = [m for m in msgs if int(m.get("turn", 0)) < turn]
    thread_text = ""
    for i, msg in enumerate(context_msgs):
        author = str(msg.get("author_id", "unknown"))
        text = str(msg.get("text", "")).strip()
        thread_text += f"[{i + 1}] User {author}: {text}\n"

    system_prompt = _system_prompt_for_agent(agent_name)
    return (
        f"System: {system_prompt}\n\n"
        "You are reading this conversation thread:\n"
        f"{thread_text}\n"
        "Your task:\n"
        "Write the next reply to continue this thread naturally.\n"
        "Keep it concise (1-3 sentences).\n"
        "Output only the reply text."
    )


def group_records(records: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        seed_id = str(rec.get("seed_id", "")).strip()
        mode = str(rec.get("mode", "")).strip()
        if not seed_id or mode not in {"toxic", "neutral"}:
            continue
        rollout_id = int(rec.get("rollout_id", 0))
        grouped[(seed_id, rollout_id)][mode] = rec
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_pattern",
        default="data/reddit/influence_baseline_threads_reddit_rollouts_rollout_*.jsonl",
        help="Glob or single path for baseline rollout JSONL files.",
    )
    parser.add_argument(
        "--output_path",
        default="data/dpo/training_pairs.jsonl",
    )
    parser.add_argument("--max_turn", type=int, default=4)
    parser.add_argument("--tox_margin", type=float, default=1e-6)
    parser.add_argument("--vader_margin", type=float, default=1e-6)
    parser.add_argument("--identical_threshold", type=float, default=0.97)
    args = parser.parse_args()

    if not os.path.isabs(args.input_pattern):
        args.input_pattern = os.path.join(PROJECT_ROOT, args.input_pattern)
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(PROJECT_ROOT, args.output_path)

    input_files = resolve_input_files(args.input_pattern)
    if not input_files:
        raise FileNotFoundError(f"No input JSONL files matched: {args.input_pattern}")

    rows: List[Dict[str, Any]] = []
    for p in input_files:
        rows.extend(load_jsonl(p))

    grouped = group_records(rows)
    vader = _try_load_vader()

    out_parent = os.path.dirname(args.output_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    total_candidates = 0
    kept = 0
    turn_counts: Dict[int, int] = defaultdict(int)
    chosen_tox_vals: List[float] = []
    rejected_tox_vals: List[float] = []

    with open(args.output_path, "w", encoding="utf-8") as out_f:
        for _, pair in grouped.items():
            toxic = pair.get("toxic")
            neutral = pair.get("neutral")
            if toxic is None or neutral is None:
                continue

            for turn in range(1, int(args.max_turn) + 1):
                toxic_msg = get_turn_message(toxic, turn)
                neutral_msg = get_turn_message(neutral, turn)
                if toxic_msg is None or neutral_msg is None:
                    continue

                rejected = str(toxic_msg.get("text", "")).strip()
                chosen = str(neutral_msg.get("text", "")).strip()
                if not chosen or not rejected:
                    continue

                total_candidates += 1
                if almost_identical(chosen, rejected, threshold=float(args.identical_threshold)):
                    continue

                tox_chosen = float(score_text_toxicity(chosen))
                tox_rejected = float(score_text_toxicity(rejected))
                keep_by_tox = tox_rejected > (tox_chosen + float(args.tox_margin))

                keep_by_vader = False
                if vader is not None:
                    sent_chosen = float(vader.polarity_scores(chosen).get("compound", 0.0))
                    sent_rejected = float(vader.polarity_scores(rejected).get("compound", 0.0))
                    keep_by_vader = sent_rejected < (sent_chosen - float(args.vader_margin))

                if not (keep_by_tox or keep_by_vader):
                    continue

                prompt = format_prompt_like_simulator(
                    toxic_record=toxic,
                    turn=turn,
                    agent_name=str(toxic_msg.get("agent", "")),
                )
                out_f.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "chosen": chosen,
                            "rejected": rejected,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                kept += 1
                turn_counts[turn] += 1
                chosen_tox_vals.append(tox_chosen)
                rejected_tox_vals.append(tox_rejected)

    mean_chosen = mean(chosen_tox_vals) if chosen_tox_vals else 0.0
    mean_rejected = mean(rejected_tox_vals) if rejected_tox_vals else 0.0

    print(f"[OK] Input files: {len(input_files)}")
    print(f"[OK] Candidate pairs before filtering: {total_candidates}")
    print(f"[OK] Kept pairs: {kept}")
    print(f"[OK] Mean toxicity chosen: {mean_chosen:.6f}")
    print(f"[OK] Mean toxicity rejected: {mean_rejected:.6f}")
    print("[OK] Pairs per turn:")
    for t in range(1, int(args.max_turn) + 1):
        print(f"  turn {t}: {turn_counts.get(t, 0)}")
    if vader is None:
        print("[WARN] NLTK VADER unavailable; filtering used Detoxify-only criterion.")
    print(f"[OK] Wrote {args.output_path}")


if __name__ == "__main__":
    main()
