"""
Convert graph rollout JSONL files to a flat CSV.

Input:  a summary JSON file (e.g. dag_injection_summary.json) whose
        "output_files" field lists the rollout JSONL paths.
Output: one row per message/node per record, with record-level metadata,
        node-level content, and detoxify / sentiment scores.

Usage:
    python data/graph_to_csv.py data/graph/dag_injection_summary.json
    python data/graph_to_csv.py data/graph/dag_injection_summary.json --out data/dag_injection.csv

    # process all summary files at once
    python data/graph_to_csv.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
GRAPH_DIR = SCRIPT_DIR / "graph"

DETOXIFY_DIMS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]

CSV_COLUMNS = [
    # record-level
    "seed_id",
    "topology",
    "mode",
    "rollout_id",
    "n_toxic_injections",
    "toxic_injection_strategy",
    "context_mode",
    "memory_mode",
    # node-level
    "node",
    "turn",
    "agent_slot",
    "author_id",
    "depth",
    "parents",
    "injection",
    "text",
    # scores
    "toxicity",
    "sentiment",
    *[f"detoxify_{d}" for d in DETOXIFY_DIMS],
    # record-level graph metrics
    "global_mean_toxicity",
    "global_mean_sentiment",
    "n_injection_nodes",
    "n_total_nodes",
]


def resolve_jsonl_paths(summary_path: Path) -> list[Path]:
    """Resolve JSONL paths from the summary file.

    The stored paths are relative to the src/ working directory
    (e.g. '../data/graph/rollouts/foo.jsonl'), so we try multiple
    base directories and pick the first that resolves to a real file.
    """
    with open(summary_path) as f:
        summary = json.load(f)

    # candidate base directories to try, in order
    project_root = summary_path.parent.parent  # data/graph -> data -> project root
    candidates = [
        summary_path.parent / "src",   # data/graph/src  (unlikely)
        project_root / "src",           # project_root/src  (original working dir)
        summary_path.parent,            # data/graph
        project_root,                   # project_root
    ]

    paths = []
    for p in summary.get("output_files", []):
        resolved = None
        for base in candidates:
            candidate = (base / p).resolve()
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            # fall back to project_root/src resolution even if missing
            resolved = (project_root / "src" / p).resolve()
        paths.append(resolved)
    return paths


def record_to_rows(rec: dict) -> list[dict]:
    """Flatten one JSONL record into one CSV row per message node."""
    scores = rec.get("scores") or {}
    tox_by_node: dict[str, float] = {
        str(k): float(v) for k, v in (scores.get("toxicity_by_node") or {}).items()
    }
    sent_by_node: dict[str, float] = {
        str(k): float(v) for k, v in (scores.get("sentiment_by_node") or {}).items()
    }
    detox_by_node: dict[str, dict] = {
        str(k): v for k, v in (scores.get("detoxify_by_node") or {}).items()
    }

    gm = rec.get("graph_metrics") or {}

    rows = []
    for msg in rec.get("messages", []):
        node_key = str(msg.get("node", ""))
        detox = detox_by_node.get(node_key) or {}

        row: dict = {
            # record-level
            "seed_id": rec.get("seed_id", ""),
            "topology": rec.get("topology", ""),
            "mode": rec.get("mode", ""),
            "rollout_id": rec.get("rollout_id", ""),
            "n_toxic_injections": rec.get("n_toxic_injections", ""),
            "toxic_injection_strategy": rec.get("toxic_injection_strategy", ""),
            "context_mode": rec.get("context_mode", ""),
            "memory_mode": rec.get("memory_mode", ""),
            # node-level
            "node": msg.get("node", ""),
            "turn": msg.get("turn", ""),
            "agent_slot": msg.get("agent_slot", ""),
            "author_id": msg.get("author_id", ""),
            "depth": msg.get("depth", ""),
            "parents": "|".join(str(p) for p in (msg.get("parents") or [])),
            "injection": msg.get("injection", False),
            "text": msg.get("text") or msg.get("content", ""),
            # scores
            "toxicity": tox_by_node.get(node_key, ""),
            "sentiment": sent_by_node.get(node_key, ""),
            **{f"detoxify_{d}": detox.get(d, "") for d in DETOXIFY_DIMS},
            # graph metrics
            "global_mean_toxicity": gm.get("global_mean_toxicity", ""),
            "global_mean_sentiment": gm.get("global_mean_sentiment", ""),
            "n_injection_nodes": gm.get("n_injection_nodes", ""),
            "n_total_nodes": gm.get("n_total_nodes", ""),
        }
        rows.append(row)
    return rows


def convert(summary_path: Path, out_path: Path) -> None:
    jsonl_paths = resolve_jsonl_paths(summary_path)
    missing = [p for p in jsonl_paths if not p.exists()]
    if missing:
        print(f"[WARN] missing files: {missing}")
        jsonl_paths = [p for p in jsonl_paths if p.exists()]

    total_rows = 0
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for jsonl_path in jsonl_paths:
            with open(jsonl_path, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    for row in record_to_rows(rec):
                        writer.writerow(row)
                        total_rows += 1

    print(f"[OK] {summary_path.name} → {out_path}  ({total_rows} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert graph rollout JSON to CSV.")
    parser.add_argument(
        "summary",
        nargs="?",
        help="Path to a summary JSON file.",
    )
    parser.add_argument(
        "--out",
        help="Output CSV path (default: same name as summary, .csv extension).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all *_summary.json files in data/graph/.",
    )
    args = parser.parse_args()

    if args.all:
        summaries = sorted(GRAPH_DIR.glob("*_summary.json"))
        if not summaries:
            print("[WARN] No *_summary.json files found in", GRAPH_DIR)
            return
        for s in summaries:
            out = s.with_suffix(".csv")
            convert(s, out)
    elif args.summary:
        summary_path = Path(args.summary).resolve()
        out_path = Path(args.out).resolve() if args.out else summary_path.with_suffix(".csv")
        convert(summary_path, out_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
