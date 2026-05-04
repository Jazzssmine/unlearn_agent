import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

if __package__:
    from .real_agents.toxicity_scorer import score_text_toxicity
else:
    from real_agents.toxicity_scorer import score_text_toxicity


DEFAULT_INPUT = "data/dpo/training_pairs.jsonl"
DEFAULT_OUTPUT = "data/dpo/preference_pairs.jsonl"
PLACEHOLDER_REPLY = "<reply text if action is reply, else empty>"


def _is_valid_text(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    if text == PLACEHOLDER_REPLY:
        return False
    return True


def _to_pair_record(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    context = obj.get("context", obj.get("prompt", ""))
    chosen = obj.get("chosen", "")
    rejected = obj.get("rejected", "")

    if not _is_valid_text(context):
        return {}, "missing_context"
    if not _is_valid_text(chosen):
        return {}, "invalid_chosen"
    if not _is_valid_text(rejected):
        return {}, "invalid_rejected"

    context = str(context).strip()
    chosen = str(chosen).strip()
    rejected = str(rejected).strip()

    chosen_tox = float(score_text_toxicity(chosen))
    rejected_tox = float(score_text_toxicity(rejected))
    if not (rejected_tox > chosen_tox):
        return {}, "tox_order_failed"

    out = {
        "context": context,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_tox": chosen_tox,
        "rejected_tox": rejected_tox,
    }
    return out, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert training_pairs.jsonl to preference_pairs.jsonl with Detoxify "
            "scores and tox-order filtering."
        )
    )
    parser.add_argument("--in_jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--out_jsonl", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow_existing_context",
        action="store_true",
        help="Allow input records with `context` field (otherwise expect `prompt`).",
    )
    args = parser.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "written": 0,
        "json_error": 0,
        "missing_context": 0,
        "invalid_chosen": 0,
        "invalid_rejected": 0,
        "tox_order_failed": 0,
    }

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["json_error"] += 1
                continue

            if not args.allow_existing_context and "context" in obj and "prompt" not in obj:
                stats["missing_context"] += 1
                continue

            rec, status = _to_pair_record(obj)
            if status != "ok":
                stats[status] += 1
                continue

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["written"] += 1

    print(
        "Built preference pairs: "
        f"total={stats['total']} written={stats['written']} "
        f"json_error={stats['json_error']} missing_context={stats['missing_context']} "
        f"invalid_chosen={stats['invalid_chosen']} invalid_rejected={stats['invalid_rejected']} "
        f"tox_order_failed={stats['tox_order_failed']} "
        f"out={out_path}"
    )


if __name__ == "__main__":
    main()
