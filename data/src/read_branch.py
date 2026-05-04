import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _to_int_or_none(v: Any) -> Any:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float_or_none(v: Any) -> Any:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_tox_by_turn(
    messages: List[Dict[str, Any]],
    scores_tox: Dict[str, Any],
) -> Dict[int, float]:
    """
    Get per-turn toxicity as {turn: float}, preferring scores.toxicity_by_turn
    and falling back to messages[*].detoxify.toxicity.
    """
    out: Dict[int, float] = {}
    for k, v in (scores_tox or {}).items():
        try:
            out[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    if out:
        return out

    for m in messages:
        t = _to_int_or_none(m.get("turn"))
        if t is None:
            continue
        d = m.get("detoxify") or {}
        tox = d.get("toxicity")
        fv = _to_float_or_none(tox)
        if fv is not None:
            out[t] = fv
    return out


def _compute_tree_stats(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute simple branching/depth stats from message list where reply_to points to parent turn.
    """
    children: Dict[int, List[int]] = defaultdict(list)
    turns = set()
    for m in messages:
        t = _to_int_or_none(m.get("turn"))
        if t is None:
            continue
        turns.add(t)
        p = _to_int_or_none(m.get("reply_to"))
        if p is not None:
            children[p].append(t)

    child_counts = [len(v) for v in children.values()]
    n_branch_nodes_ge2 = sum(c >= 2 for c in child_counts)
    max_children = max(child_counts) if child_counts else 0
    mean_children_nonleaf = (sum(child_counts) / len(child_counts)) if child_counts else 0.0

    # Depth from root turn 0
    depth_by_turn: Dict[int, int] = {}
    if 0 in turns:
        q = deque([(0, 0)])
        depth_by_turn[0] = 0
        while q:
            node, d = q.popleft()
            for ch in children.get(node, []):
                if ch not in depth_by_turn:
                    depth_by_turn[ch] = d + 1
                    q.append((ch, d + 1))

    max_depth = max(depth_by_turn.values()) if depth_by_turn else 0
    return {
        "n_branch_nodes_ge2": int(n_branch_nodes_ge2),
        "max_children": int(max_children),
        "mean_children_nonleaf": float(mean_children_nonleaf),
        "max_depth": int(max_depth),
    }

def _compute_depth_by_turn(messages: List[Dict[str, Any]]) -> Dict[int, int]:
    children: Dict[int, List[int]] = defaultdict(list)
    turns = set()
    for m in messages:
        t = _to_int_or_none(m.get("turn"))
        if t is None:
            continue
        turns.add(t)
        p = _to_int_or_none(m.get("reply_to"))
        if p is not None:
            children[p].append(t)

    depth_by_turn: Dict[int, int] = {}
    if 0 in turns:
        q = deque([(0, 0)])
        depth_by_turn[0] = 0
        while q:
            node, d = q.popleft()
            for ch in children.get(node, []):
                if ch not in depth_by_turn:
                    depth_by_turn[ch] = d + 1
                    q.append((ch, d + 1))
    return depth_by_turn


def _a1_agent_for_mode(mode: Any) -> Any:
    if mode == "toxic":
        return "A1_toxic"
    if mode == "neutral":
        return "A1_neutral"
    return None


def _annotate_a1_lineage(messages: List[Dict[str, Any]], mode: Any) -> Dict[int, Dict[str, Any]]:
    """
    For each turn, find nearest A1 ancestor (if any) using reply_to parent links.
    """
    target_a1_agent = _a1_agent_for_mode(mode)
    by_turn: Dict[int, Dict[str, Any]] = {}
    parent_of: Dict[int, Any] = {}
    for m in messages:
        t = _to_int_or_none(m.get("turn"))
        if t is None:
            continue
        by_turn[t] = m
        parent_of[t] = _to_int_or_none(m.get("reply_to"))

    out: Dict[int, Dict[str, Any]] = {}
    for t in by_turn.keys():
        if target_a1_agent is None:
            out[t] = {
                "a1_expected_agent": None,
                "a1_ancestor_turn": None,
                "distance_to_a1": None,
                "is_descendant_of_a1": False,
                "a1_branch_child_turn": None,
            }
            continue

        cur = t
        path: List[int] = [t]
        a1_turn = None
        while cur is not None:
            msg = by_turn.get(cur)
            if msg is None:
                break
            if msg.get("agent") == target_a1_agent:
                a1_turn = cur
                break
            cur = parent_of.get(cur)
            if cur is not None:
                path.append(cur)

        if a1_turn is None:
            out[t] = {
                "a1_expected_agent": target_a1_agent,
                "a1_ancestor_turn": None,
                "distance_to_a1": None,
                "is_descendant_of_a1": False,
                "a1_branch_child_turn": None,
            }
            continue

        distance = len(path) - 1
        # path order is [self, parent, ..., a1]; child directly under A1 is the element before A1.
        branch_child_turn = path[-2] if distance >= 1 else None
        out[t] = {
            "a1_expected_agent": target_a1_agent,
            "a1_ancestor_turn": a1_turn,
            "distance_to_a1": int(distance),
            "is_descendant_of_a1": bool(distance >= 1),
            "a1_branch_child_turn": branch_child_turn,
        }
    return out


def _compute_toxic_graph_metrics(
    messages: List[Dict[str, Any]],
    tox_by_turn: Dict[int, float],
    a1_lineage: Dict[int, Dict[str, Any]],
    tox_threshold: float,
) -> Dict[str, Any]:
    """
    Metrics:
      - global vibe shift: mean toxicity, fraction >= threshold
      - cascade severity: largest toxic connected component size, toxic edges
      - toxic cascade probability event: any toxic message at distance >= 2 from A1
    """
    turns: List[int] = []
    parent_of: Dict[int, int] = {}
    for m in messages:
        t = _to_int_or_none(m.get("turn"))
        if t is None:
            continue
        turns.append(t)
        p = _to_int_or_none(m.get("reply_to"))
        if p is not None:
            parent_of[t] = p

    tox_vals = [tox_by_turn.get(t) for t in turns if tox_by_turn.get(t) is not None]
    n_total = len(turns)
    n_toxic = sum(1 for t in turns if (tox_by_turn.get(t) is not None and tox_by_turn[t] >= tox_threshold))
    frac_toxic = (n_toxic / n_total) if n_total > 0 else None

    toxic_nodes = {t for t in turns if (tox_by_turn.get(t) is not None and tox_by_turn[t] >= tox_threshold)}

    # Count toxic edges where both endpoints are toxic.
    n_toxic_edges = 0
    undirected_adj: Dict[int, set] = defaultdict(set)
    for child, parent in parent_of.items():
        if child in toxic_nodes and parent in toxic_nodes:
            n_toxic_edges += 1
            undirected_adj[child].add(parent)
            undirected_adj[parent].add(child)

    # Largest connected toxic component size.
    visited = set()
    largest_comp = 0
    for node in toxic_nodes:
        if node in visited:
            continue
        q = deque([node])
        visited.add(node)
        comp_size = 0
        while q:
            cur = q.popleft()
            comp_size += 1
            for nb in undirected_adj.get(cur, set()):
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        largest_comp = max(largest_comp, comp_size)

    # Toxic cascade event beyond distance >=2 from A1.
    has_toxic_beyond_d2 = False
    for t in toxic_nodes:
        d = a1_lineage.get(t, {}).get("distance_to_a1")
        if d is not None and d >= 2:
            has_toxic_beyond_d2 = True
            break

    return {
        "mean_toxicity_all_messages": (float(pd.Series(tox_vals).mean()) if tox_vals else None),
        "frac_messages_toxic_ge_tau": frac_toxic,
        "n_toxic_messages_ge_tau": int(n_toxic),
        "largest_toxic_component_size": int(largest_comp),
        "n_toxic_edges": int(n_toxic_edges),
        "has_toxic_beyond_d2": int(has_toxic_beyond_d2),
    }


def build_dataframes(
    records: List[Dict[str, Any]],
    tox_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    record_rows: List[Dict[str, Any]] = []
    message_rows: List[Dict[str, Any]] = []
    vote_rows: List[Dict[str, Any]] = []

    for rec in records:
        seed_id = rec.get("seed_id")
        mode = rec.get("mode")
        seed = rec.get("seed", {}) or {}
        messages = rec.get("messages", []) or []
        votes = rec.get("votes", []) or []
        scores = rec.get("scores", {}) or {}
        tox_by_turn = _normalize_tox_by_turn(messages, scores.get("toxicity_by_turn", {}) or {})
        sent_by_turn = scores.get("sentiment_by_turn", {}) or {}
        intervention_meta = rec.get("intervention_meta", {}) or {}
        depth_by_turn = _compute_depth_by_turn(messages)
        a1_lineage = _annotate_a1_lineage(messages, mode)
        toxic_graph_metrics = _compute_toxic_graph_metrics(
            messages=messages,
            tox_by_turn=tox_by_turn,
            a1_lineage=a1_lineage,
            tox_threshold=tox_threshold,
        )

        tree_stats = _compute_tree_stats(messages)
        record_rows.append(
            {
                "seed_id": seed_id,
                "mode": mode,
                "thread_id": seed.get("thread_id"),
                "source_format": (seed.get("reddit", {}) or {}).get("source_format"),
                "n_messages": len(messages),
                "n_votes": len(votes),
                "intervention_agent": intervention_meta.get("intervention_agent"),
                "intervention_lead_time_seconds": _to_float_or_none(
                    intervention_meta.get("intervention_lead_time_seconds")
                ),
                "mean_toxicity": (
                    float(pd.Series([_to_float_or_none(v) for v in tox_by_turn.values()]).dropna().mean())
                    if tox_by_turn
                    else None
                ),
                "mean_sentiment": (
                    float(pd.Series([_to_float_or_none(v) for v in sent_by_turn.values()]).dropna().mean())
                    if sent_by_turn
                    else None
                ),
                **toxic_graph_metrics,
                **tree_stats,
            }
        )

        sorted_messages = sorted(messages, key=lambda x: (_to_int_or_none(x.get("turn")) is None, _to_int_or_none(x.get("turn"))))
        for m in sorted_messages:
            turn = _to_int_or_none(m.get("turn"))
            turn_key = str(turn) if turn is not None else None
            a1_info = a1_lineage.get(
                turn,
                {
                    "a1_expected_agent": None,
                    "a1_ancestor_turn": None,
                    "distance_to_a1": None,
                    "is_descendant_of_a1": False,
                    "a1_branch_child_turn": None,
                },
            )
            dist = a1_info.get("distance_to_a1")
            if dist is None:
                distance_bucket = "none"
            elif dist == 0:
                distance_bucket = "d0_self_a1"
            elif dist == 1:
                distance_bucket = "d1_child_of_a1"
            else:
                distance_bucket = "d2plus_descendant"
            message_rows.append(
                {
                    "seed_id": seed_id,
                    "mode": mode,
                    "thread_id": seed.get("thread_id"),
                    "turn": turn,
                    "agent": m.get("agent"),
                    "author_id": m.get("author_id"),
                    "reply_to": _to_int_or_none(m.get("reply_to")),
                    "created_utc": _to_int_or_none(m.get("created_utc")),
                    "score": _to_float_or_none(m.get("score")),
                    "text_len": len(str(m.get("text", ""))),
                    "toxicity": _to_float_or_none(tox_by_turn.get(turn_key)),
                    "sentiment": _to_float_or_none(sent_by_turn.get(turn_key)),
                    "depth_from_seed": depth_by_turn.get(turn),
                    "a1_expected_agent": a1_info.get("a1_expected_agent"),
                    "a1_ancestor_turn": a1_info.get("a1_ancestor_turn"),
                    "distance_to_a1": dist,
                    "distance_bucket": distance_bucket,
                    "is_descendant_of_a1": bool(a1_info.get("is_descendant_of_a1", False)),
                    "a1_branch_child_turn": a1_info.get("a1_branch_child_turn"),
                }
            )

        for v in votes:
            vote_rows.append(
                {
                    "seed_id": seed_id,
                    "mode": mode,
                    "thread_id": seed.get("thread_id"),
                    "voter_slot": v.get("voter_slot"),
                    "target_turn": _to_int_or_none(v.get("target_turn")),
                    "vote_value": _to_int_or_none(v.get("vote_value")),
                    "score_before": _to_float_or_none(v.get("score_before")),
                    "score_after": _to_float_or_none(v.get("score_after")),
                }
            )

    records_df = pd.DataFrame(record_rows)
    messages_df = pd.DataFrame(message_rows)
    votes_df = pd.DataFrame(vote_rows)
    return records_df, messages_df, votes_df


def build_a1_influence_tables(messages_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build A1 influence summaries:
      1) descendants summary by mode + distance bucket
      2) paired toxic-vs-neutral deltas per seed
    """
    if messages_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    downstream_agents = {"A2", "A3", "A4"}
    desc_df = messages_df[
        messages_df["mode"].isin(["toxic", "neutral"])
        & messages_df["is_descendant_of_a1"].fillna(False)
        & messages_df["agent"].isin(downstream_agents)
    ].copy()

    if desc_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_mode_distance = (
        desc_df.groupby(["mode", "distance_bucket"], as_index=False)
        .agg(
            n_messages=("turn", "count"),
            mean_toxicity=("toxicity", "mean"),
            mean_sentiment=("sentiment", "mean"),
            mean_score=("score", "mean"),
            mean_text_len=("text_len", "mean"),
        )
        .sort_values(["mode", "distance_bucket"])
    )

    per_seed = (
        desc_df.groupby(["seed_id", "mode"], as_index=False)
        .agg(
            mean_toxicity=("toxicity", "mean"),
            mean_sentiment=("sentiment", "mean"),
            n_messages=("turn", "count"),
        )
    )
    wide = per_seed.pivot(index="seed_id", columns="mode", values=["mean_toxicity", "mean_sentiment", "n_messages"])
    wide.columns = ["_".join(col) for col in wide.columns]
    wide = wide.reset_index()

    for req_col in ["mean_toxicity_toxic", "mean_toxicity_neutral", "mean_sentiment_toxic", "mean_sentiment_neutral"]:
        if req_col not in wide.columns:
            wide[req_col] = pd.NA

    wide["delta_toxicity_toxic_minus_neutral"] = wide["mean_toxicity_toxic"] - wide["mean_toxicity_neutral"]
    wide["delta_sentiment_toxic_minus_neutral"] = wide["mean_sentiment_toxic"] - wide["mean_sentiment_neutral"]
    return by_mode_distance, wide


def build_contagion_radius_table(messages_df: pd.DataFrame) -> pd.DataFrame:
    """
    E[tox | distance_to_a1 = k] for k = 1,2,3...
    """
    if messages_df.empty:
        return pd.DataFrame()
    cdf = messages_df[
        messages_df["mode"].isin(["toxic", "neutral"])
        & messages_df["distance_to_a1"].notna()
        & (messages_df["distance_to_a1"] >= 1)
    ].copy()
    if cdf.empty:
        return pd.DataFrame()
    out = (
        cdf.groupby(["mode", "distance_to_a1"], as_index=False)
        .agg(
            n_messages=("turn", "count"),
            mean_toxicity=("toxicity", "mean"),
            mean_sentiment=("sentiment", "mean"),
            mean_score=("score", "mean"),
        )
        .sort_values(["mode", "distance_to_a1"])
    )
    return out


def save_outputs(
    records_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    tox_threshold: float = 0.5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / f"{stem}_records.csv"
    messages_path = out_dir / f"{stem}_messages.csv"
    votes_path = out_dir / f"{stem}_votes.csv"
    mode_summary_path = out_dir / f"{stem}_mode_summary.csv"
    turn_summary_path = out_dir / f"{stem}_turn_mode_summary.csv"
    a1_desc_summary_path = out_dir / f"{stem}_a1_descendant_summary.csv"
    a1_paired_path = out_dir / f"{stem}_a1_paired_toxic_vs_neutral.csv"
    global_vibe_mode_path = out_dir / f"{stem}_global_vibe_by_mode_tau{tox_threshold}.csv"
    contagion_radius_path = out_dir / f"{stem}_contagion_radius_by_mode.csv"
    cascade_mode_path = out_dir / f"{stem}_cascade_severity_by_mode_tau{tox_threshold}.csv"

    records_df.to_csv(records_path, index=False)
    messages_df.to_csv(messages_path, index=False)
    votes_df.to_csv(votes_path, index=False)

    mode_summary = (
        records_df.groupby("mode", as_index=False)
        .agg(
            n_runs=("seed_id", "count"),
            avg_messages=("n_messages", "mean"),
            avg_votes=("n_votes", "mean"),
            avg_max_depth=("max_depth", "mean"),
            avg_max_children=("max_children", "mean"),
            avg_branch_nodes_ge2=("n_branch_nodes_ge2", "mean"),
            avg_toxicity=("mean_toxicity", "mean"),
            avg_sentiment=("mean_sentiment", "mean"),
        )
        .sort_values("mode")
    )
    mode_summary.to_csv(mode_summary_path, index=False)

    turn_summary = (
        messages_df.groupby(["mode", "turn"], as_index=False)
        .agg(
            n_messages=("turn", "count"),
            mean_toxicity=("toxicity", "mean"),
            mean_sentiment=("sentiment", "mean"),
            mean_score=("score", "mean"),
            mean_text_len=("text_len", "mean"),
        )
        .sort_values(["mode", "turn"])
    )
    turn_summary.to_csv(turn_summary_path, index=False)

    a1_desc_summary, a1_paired = build_a1_influence_tables(messages_df)
    a1_desc_summary.to_csv(a1_desc_summary_path, index=False)
    a1_paired.to_csv(a1_paired_path, index=False)

    # (A) Global vibe shift by mode
    global_vibe = (
        records_df.groupby("mode", as_index=False)
        .agg(
            n_runs=("seed_id", "count"),
            mean_toxicity_all_messages=("mean_toxicity_all_messages", "mean"),
            frac_messages_toxic_ge_tau=("frac_messages_toxic_ge_tau", "mean"),
        )
        .sort_values("mode")
    )
    global_vibe.to_csv(global_vibe_mode_path, index=False)

    # (B) Contagion radius / depth table
    contagion_radius = build_contagion_radius_table(messages_df)
    contagion_radius.to_csv(contagion_radius_path, index=False)

    # (C) Cascade severity + probability by mode
    cascade_mode = (
        records_df.groupby("mode", as_index=False)
        .agg(
            n_runs=("seed_id", "count"),
            mean_largest_toxic_component_size=("largest_toxic_component_size", "mean"),
            mean_n_toxic_edges=("n_toxic_edges", "mean"),
            toxic_cascade_probability=("has_toxic_beyond_d2", "mean"),
        )
        .sort_values("mode")
    )
    cascade_mode.to_csv(cascade_mode_path, index=False)

    print(f"[OK] wrote {records_path}")
    print(f"[OK] wrote {messages_path}")
    print(f"[OK] wrote {votes_path}")
    print(f"[OK] wrote {mode_summary_path}")
    print(f"[OK] wrote {turn_summary_path}")
    print(f"[OK] wrote {a1_desc_summary_path}")
    print(f"[OK] wrote {a1_paired_path}")
    print(f"[OK] wrote {global_vibe_mode_path}")
    print(f"[OK] wrote {contagion_radius_path}")
    print(f"[OK] wrote {cascade_mode_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze influence_branches JSONL outputs.")
    parser.add_argument(
        "--input_jsonl",
        default="/u/anon3/toxic_agent/data/influence_branches_threads_reddit.jsonl",
        help="Path to branch-simulation JSONL output.",
    )
    parser.add_argument(
        "--out_dir",
        default="/u/anon3/toxic_agent/data",
        help="Directory for CSV analysis outputs.",
    )
    parser.add_argument(
        "--out_stem",
        default=None,
        help="Output filename stem. Defaults to input filename without extension.",
    )
    parser.add_argument(
        "--tox_threshold",
        type=float,
        default=0.5,
        help="Threshold tau for toxicity-based cascade metrics.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_stem = args.out_stem or input_path.stem

    records = load_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

    records_df, messages_df, votes_df = build_dataframes(records, tox_threshold=args.tox_threshold)
    save_outputs(records_df, messages_df, votes_df, out_dir, out_stem, tox_threshold=args.tox_threshold)

    print("\nQuick stats")
    print(f"- n_records: {len(records_df)}")
    print(f"- n_messages: {len(messages_df)}")
    print(f"- n_votes: {len(votes_df)}")
    if "mode" in records_df.columns and not records_df.empty:
        print("- records by mode:")
        print(records_df["mode"].value_counts().to_string())


if __name__ == "__main__":
    main()
"""
python read_branch.py --tox_threshold 0.3
"""