import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------- Helpers ----------
def collect_comments(node: Dict[str, Any], out: Dict[str, Dict[str, Any]]) -> None:
    """
    Collect all 'comment' dicts recursively from a branch/tree node into out[id] = comment.
    """
    c = node.get("comment", {}) or {}
    cid = c.get("id")
    if cid:
        out[cid] = c
    for ch in (node.get("child_nodes") or []):
        collect_comments(ch, out)

def build_children_map(comments_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    parent_id -> [child_id] adjacency
    """
    children = defaultdict(list)
    for cid, c in comments_by_id.items():
        pid = c.get("parent_id")
        if pid:
            children[pid].append(cid)
    return children

def compute_max_depth(root_id: str, children_map: Dict[str, List[str]]) -> int:
    """
    Depth(root)=0. Depth(child)=parent_depth+1.
    Returns maximum depth reachable from root.
    """
    max_d = 0
    stack = [(root_id, 0)]
    seen = set()
    while stack:
        nid, d = stack.pop()
        if (nid, d) in seen:
            continue
        seen.add((nid, d))
        max_d = max(max_d, d)
        for ch in children_map.get(nid, []):
            stack.append((ch, d + 1))
    return max_d

def build_nested_tree(
    root_id: str,
    comments_by_id: Dict[str, Dict[str, Any]],
    children_map: Dict[str, List[str]],
    sort_children: str = "created_utc",  # or "score" or "none"
) -> Dict[str, Any]:
    """
    Returns nested dict:
      {"comment": <comment_dict>, "child_nodes": [ ... ]}
    """
    def sort_key(cid: str):
        c = comments_by_id.get(cid, {})
        if sort_children == "created_utc":
            v = c.get("created_utc")
            try:
                return float(v)
            except Exception:
                return float("inf")
        if sort_children == "score":
            v = c.get("score")
            try:
                return -float(v)  # higher score first
            except Exception:
                return float("inf")
        return 0

    def recurse(cid: str) -> Dict[str, Any]:
        kids = children_map.get(cid, [])
        if sort_children != "none":
            kids = sorted(kids, key=sort_key)
        return {
            "comment": comments_by_id.get(cid, {"id": cid}),
            "child_nodes": [recurse(ch) for ch in kids],
        }

    return recurse(root_id)

# ---------- Main pipeline ----------
def reconstruct_threads_to_nested_json(
    input_jsonl: str,
    output_jsonl: str,
    min_depth: int = 5,
    sort_children: str = "created_utc",
    limit_threads: Optional[int] = None,  # set for debugging
) -> None:
    """
    Reads branch-style JSONL, merges by link_id, outputs nested trees (JSONL),
    filtered by max depth >= min_depth.
    """
    input_path = Path(input_jsonl)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # link_id -> {comment_id: comment_dict}
    threads: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    # link_id -> root_id (submission id t3_...)
    root_ids: Dict[str, str] = {}

    # 1) Read & merge all comments per link_id
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            root_comment = obj.get("comment", {}) or {}
            link_id = root_comment.get("link_id") or root_comment.get("id")
            if not link_id:
                continue

            # remember the submission/root id
            if link_id not in root_ids:
                root_ids[link_id] = root_comment.get("id") or link_id

            collect_comments(obj, threads[link_id])

    # 2) Build nested trees and write filtered output
    written = 0
    with out_path.open("w", encoding="utf-8") as w:
        for link_id, comments_by_id in threads.items():
            root_id = root_ids.get(link_id, link_id)

            # Some datasets may omit the root submission node as a "comment" dict.
            # Ensure it's present so the tree has a root.
            if root_id not in comments_by_id:
                comments_by_id[root_id] = {"id": root_id, "link_id": link_id, "root": True, "body": "root"}

            children_map = build_children_map(comments_by_id)
            max_d = compute_max_depth(root_id, children_map)

            if max_d < min_depth:
                continue

            nested = build_nested_tree(
                root_id=root_id,
                comments_by_id=comments_by_id,
                children_map=children_map,
                sort_children=sort_children,
            )

            # Add some metadata at top-level (optional but handy)
            record = {
                "link_id": link_id,
                "root_id": root_id,
                "max_depth": max_d,
                "tree": nested
            }
            w.write(json.dumps(record) + "\n")
            written += 1

            if limit_threads is not None and written >= limit_threads:
                break

    print(f"Done. Wrote {written} threads with max_depth >= {min_depth} to {out_path}")

# ---- Run it ----
if __name__ == "__main__":
    reconstruct_threads_to_nested_json(
        input_jsonl="politics.json",                 # <-- change if needed
        output_jsonl="extracted/politics_depth_ge5.jsonl", # <-- output file
        min_depth=5,
        sort_children="created_utc",                 # "score" or "none"
        limit_threads=None
    )