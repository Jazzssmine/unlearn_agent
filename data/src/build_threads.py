import json
from typing import Dict, Any, List

# pip install detoxify
from detoxify import Detoxify

LABELS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]

def deepest_path_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    ch = node.get("child_nodes", [])
    if not ch:
        return [node]
    best = max((deepest_path_nodes(c) for c in ch), key=len)
    return [node] + best

def extract_chain_from_thread_record(rec: Dict[str, Any], chain_len: int = 5) -> Dict[str, Any] | None:
    path = deepest_path_nodes(rec["tree"])
    # drop structural root ("root")
    path_comments = [n["comment"] for n in path if n.get("comment", {}).get("body") != "root"]

    if len(path_comments) < chain_len:
        return None

    chain = path_comments[:chain_len]  # [seed, A, B, C, D]
    roles = ["seed", "A", "B", "C", "D"][:chain_len]

    return {
        "thread_id": rec["link_id"],
        "max_depth": rec.get("max_depth"),
        "chain_comment_ids": [c.get("id") for c in chain],
        "chain_text": {r: (c.get("body") or "") for r, c in zip(roles, chain)},
        "chain_meta": {
            r: {
                "id": c.get("id"),
                "parent_id": c.get("parent_id"),
                "created_utc": c.get("created_utc"),
                "author": c.get("author"),
                "subreddit": c.get("subreddit"),
                "score": c.get("score"),
            }
            for r, c in zip(roles, chain)
        }
    }

def score_chain_detoxify(model: Detoxify, chain_text: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    Returns: {role: {label: score}}
    Uses batching for speed.
    """
    roles = list(chain_text.keys())
    texts = [chain_text[r] for r in roles]

    preds = model.predict(texts)  # dict[label] -> list[float]
    out = {}
    for i, r in enumerate(roles):
        out[r] = {lab: float(preds[lab][i]) for lab in preds.keys()}
    return out

def main():
    input_path = "extracted/politics_depth_ge5.jsonl"
    output_path = "extracted/politics_seedA_BCD_chains_detoxify.jsonl"

    # Choose model: "original" is standard; "unbiased" exists too.
    detox = Detoxify("original")  # use device="cuda" if you have GPU: Detoxify("original", device="cuda")

    num_in, num_out = 0, 0
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            num_in += 1
            rec = json.loads(line)

            chain = extract_chain_from_thread_record(rec, chain_len=5)
            if chain is None:
                continue

            # Detoxify scores for seed/A/B/C/D
            chain["detoxify"] = score_chain_detoxify(detox, chain["chain_text"])

            f_out.write(json.dumps(chain) + "\n")
            num_out += 1

    print(f"Read {num_in} threads; wrote {num_out} chains with Detoxify scores to {output_path}")

if __name__ == "__main__":
    main()