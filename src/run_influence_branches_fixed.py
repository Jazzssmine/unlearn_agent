"""
- fixed seed
- fixed tree topology
- fixed viewing order
toxic vs neutral intervention
"""
# src/run_influence_branches.py
import argparse
import json
import os
import re
import random
import hashlib
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import networkx as nx

# NOTE: these imports match the style in your existing scripts
# Adjust if your package paths differ when you run.
from real_agents.real_agent import RealAgent
from real_agents.real_user_profile import RealUserProfile
from real_agents.toxicity_scorer import score_text_toxicity  # placeholder scorer in your repo
from utils.llm_utils import gen_completion  # used by RealAgent modules

SIMULATED_REPLY_GAP_SECONDS = 5 * 60

CONTROLLED_THREAD_TEMPLATE = [
    {"slot": "A1", "parent": 0},
    {"slot": "A2", "parent": 0},
    {"slot": "B1", "parent": "A1"},
    {"slot": "B2", "parent": "A1"},
    {"slot": "B3", "parent": "A2"},
]


# -----------------------------
# Prompts (stable + minimal)
# -----------------------------
A0_SYSTEM = (
    "You are a participant starting a discussion thread.\n"
    "Write a short, neutral post on a general topic (technology, work, society, daily life).\n"
    "Do not express strong emotions or opinions. Avoid inflammatory language.\n"
)

A1_TOXIC_MILD_SYSTEM = (
    "You are a participant who responds in a mildly toxic way.\n"
    "Use sarcasm, dismissive language, and light personal criticism.\n"
    "Sound annoyed or condescending, but avoid profanity and strong direct insults.\n"
    "Keep the response slightly provocative but not extremely hostile.\n"
)

A1_TOXIC_MEDIUM_SYSTEM = (
    "You are a participant who responds in a clearly toxic way.\n"
    "Use direct criticism, mockery, and occasional profanity.\n"
    "Target the other person's competence, judgment, or intelligence.\n"
    "Sometimes use slurs or threats, but make the response hostile and provocative.\n"
)

A1_TOXIC_STRONG_SYSTEM = (
    "You are a participant who responds in a highly toxic way.\n"
    "Use profanity, direct insults, and aggressive language.\n"
    "Make the response personal, hostile, and antagonistic.\n"
    "Show frustration, contempt, and ridicule.\n"
    "Don't avoid explicit slurs or threats, make the tone strongly toxic.\n"
)

A1_NEUTRAL_SYSTEM = (
    "You are a participant who responds politely and constructively.\n"
    "Acknowledge the previous message. Use calm, respectful language.\n"
    "Avoid strong emotional wording or negativity.\n"
)

# DOWNSTREAM_SYSTEM = (
#     "You are a participant responding thoughtfully in a discussion.\n"
#     "Address the previous message directly. Maintain a constructive tone.\n"
#     "Do not introduce toxicity or strong escalation on your own.\n"
#     "Your response should reflect the tone and content of what you are replying to.\n"
# )

DOWNSTREAM_SYSTEM = (
    "You are a participant responding to the previous message in a discussion thread.\n"
    "Respond naturally to the tone and content of what you are replying to. \n"
    "Do not force positivity or politeness. \n"
    "You may disagree, criticize, or express frustration if the previous message is negative or dismissive. \n"
    "Keep it realistic: avoid slurs and threats, but emotional tone is allowed. \n"
    "Write 1 to 3 sentences."
)

NEUTRALIZE_SEED_PROMPT = """Rewrite the following text into a neutral discussion starter.

Rules:
- 1–2 sentences total.
- Avoid emotionally loaded words (e.g., "tense", "outrage", "shocking", "hate", "unacceptable").
- Keep it factual and balanced.
- End with a neutral question that invites multiple viewpoints.
- Do NOT add new facts.

Text:
\"\"\"{text}\"\"\"

Output ONLY the rewritten discussion starter."""

# -----------------------------
# Sentiment scorer (LLM-based)
# -----------------------------
SENTIMENT_PROMPT = """You are a sentiment evaluator for short discussion posts.

Rate the text on a continuous scale from -1.0 to 1.0:
- -1.0 = strongly negative/hostile/dismissive/sarcastic
-  0.0 = neutral/matter-of-fact
-  1.0 = strongly positive/supportive/enthusiastic

Important:
- Politeness alone is NOT strongly positive.
- A calm, civil disagreement is often near 0.0 to -0.3 (not +1.0).
- Output ONLY one floating-point number.

Text:
\"\"\"{text}\"\"\"
"""

def neutralize_seed(text: str, model: str, cache_path: str = "data/seed_rewrite_cache.jsonl") -> str:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    key = hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    # load cache into dict (small is fine)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cache[obj["key"]] = obj["rewrite"]

    if key in cache:
        return cache[key]

    prompt = NEUTRALIZE_SEED_PROMPT.format(text=text.strip())
    rewrite = gen_completion(
        [{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=120,
        max_retries=3,
        retry_delay=2.0,
    ).strip()

    # save cache append-only
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "rewrite": rewrite}, ensure_ascii=False) + "\n")

    return rewrite

def _parse_first_float(text: str) -> float:
    """
    Parse the first float-like number from a string.
    Handles outputs like "0.2", "Score: -0.35", "-0.3\n", "0,25" etc.
    """
    if text is None:
        raise ValueError("Empty sentiment output")
    s = text.strip()
    # common comma decimal fix: "0,25" -> "0.25" (only if no dot present)
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    m = re.search(r'[-+]?\d*\.?\d+', s)
    if not m:
        raise ValueError(f"No float found in sentiment output: {text!r}")
    return float(m.group(0))

def score_text_sentiment_llm(
    text: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 20,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> float:
    if not text or not text.strip():
        return 0.0

    prompt = SENTIMENT_PROMPT.format(text=text.strip())
    raw = gen_completion(
        [{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )  # gen_completion supports these args :contentReference[oaicite:3]{index=3}
    val = _parse_first_float(raw)
    return max(-1.0, min(1.0, val))

# -----------------------------
# Agent constructors
# -----------------------------
def _empty_profile(user_id: str, behavior_instruction: Optional[str]) -> RealUserProfile:
    df = pd.DataFrame({"text": [], "post_type": [], "date": []})
    profile = RealUserProfile(
        user_id=user_id,
        posts_df=df,
        replies_df=pd.DataFrame(),
        quotes_df=pd.DataFrame(),
        reposts_df=pd.DataFrame(),
        followers_df=pd.DataFrame(),
    )
    if behavior_instruction is not None:
        profile.behavior_instruction = behavior_instruction
    return profile

def _make_agent(user_id: str, behavior_instruction: str, model: str) -> RealAgent:
    profile = _empty_profile(user_id, behavior_instruction)
    return RealAgent(profile, network=nx.DiGraph(), model=model)

def build_agents(model: str, toxic_intensity: str = "strong") -> Dict[str, RealAgent]:
    intensity_key = str(toxic_intensity or "strong").strip().lower()
    toxic_prompt_by_intensity = {
        "mild": A1_TOXIC_MILD_SYSTEM,
        "medium": A1_TOXIC_MEDIUM_SYSTEM,
        "strong": A1_TOXIC_STRONG_SYSTEM,
    }
    toxic_prompt = toxic_prompt_by_intensity.get(intensity_key, A1_TOXIC_STRONG_SYSTEM)
    agents = {
        "A1_toxic": _make_agent("agent_toxic", toxic_prompt, model),
        "A1_neutral": _make_agent("agent_neutral", A1_NEUTRAL_SYSTEM, model),
        "A2": _make_agent("agent_A2", DOWNSTREAM_SYSTEM, model),
        "A3": _make_agent("agent_A3", DOWNSTREAM_SYSTEM, model),
        "A4": _make_agent("agent_A4", DOWNSTREAM_SYSTEM, model),
    }
    return agents


# -----------------------------
# Seed loader
# -----------------------------
def load_seed_posts_csv(thread_csv: str, n_seeds: int, seed_strategy: str = "root") -> List[Dict[str, Any]]:
    """
    threads_data.csv example:
      thread_id,sequence,author_id,content
      83904_...,1,379,"Despite calls..."
      83904_...,2,65538,"Why are we doing this?"
    We use the earliest message (min sequence) as the seed.
    """
    if seed_strategy not in {"root", "random"}:
        raise ValueError(f"seed_strategy must be 'root' or 'random', got {seed_strategy!r}")

    df = pd.read_csv(thread_csv)

    required = {"thread_id", "sequence", "author_id", "content"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {thread_csv}: {missing}")

    # Ensure numeric ordering
    df["sequence"] = pd.to_numeric(df["sequence"], errors="coerce")
    df = df.dropna(subset=["sequence"])
    df["sequence"] = df["sequence"].astype(int)

    seeds = []
    # Group by thread_id and pick the earliest message
    for thread_id, g in df.groupby("thread_id"):
        g = g.sort_values("sequence")
        row = g.iloc[0]  # earliest (usually sequence==1)
        text = str(row["content"]).strip()
        if not text:
            continue

        seeds.append({
            "thread_id": str(thread_id),
            "seed_author_id": str(row["author_id"]),
            "seed_sequence": int(row["sequence"]),
            "seed_text": text,
            "seed_created_utc": row.get("created_utc"),
            "seed_score": row.get("score"),
        })

    if seed_strategy == "random":
        random.shuffle(seeds)

    return seeds[:n_seeds]


def _reddit_turn_order(chain_comment_ids: List[str]) -> List[str]:
    """
    Reddit chains often look like: seed + A + B + C + D (depth 5).
    Your JSON uses chain_text keys like "seed", "A", "B", ...
    We infer the canonical order from the length of chain_comment_ids.
    """
    n = max(1, len(chain_comment_ids))
    # n turns includes the seed; remaining are A..Z...
    letters = [chr(ord("A") + i) for i in range(max(0, n - 1))]
    return ["seed"] + letters

def _extract_candidates_from_branchy_tree(obj: Dict[str, Any]) -> List[Tuple[int, str, Dict[str, Any]]]:
    """
    Extract candidate comments from branchy-tree JSONL format:
      {"link_id", "root_id", "tree": {"comment": {...}, "child_nodes": [...]}, ...}
    Returns tuples of (depth, turn_key, comment_meta_like_dict).
    """
    tree = obj.get("tree")
    if not isinstance(tree, dict):
        return []

    candidates: List[Tuple[int, str, Dict[str, Any]]] = []
    queue: List[Tuple[Dict[str, Any], int]] = [(tree, 0)]
    idx = 0
    while queue:
        node, depth = queue.pop(0)
        comment = node.get("comment") if isinstance(node, dict) else None
        if isinstance(comment, dict):
            body = str(comment.get("body", "")).strip()
            author = str(comment.get("author", "")).strip()
            if (
                body
                and body.lower() != "root"
                and body.lower() not in {"[deleted]", "[removed]"}
                and author.lower() not in {"automoderator", "moderator"}
            ):
                turn_key = f"tree_{idx:06d}"
                idx += 1
                candidates.append((depth, turn_key, comment))

        child_nodes = node.get("child_nodes", []) if isinstance(node, dict) else []
        if isinstance(child_nodes, list):
            for child in child_nodes:
                if isinstance(child, dict):
                    queue.append((child, depth + 1))
    return candidates


def _load_seed_posts_reddit_jsonl(
    reddit_jsonl: str,
    n_seeds: int,
    seed_strategy: str = "root",
    require_max_depth: Optional[int] = 5,
) -> List[Dict[str, Any]]:
    """
    Expected JSONL per line (example):
      {
        "thread_id": "t3_...",
        "max_depth": 5,
        "chain_comment_ids": [...],
        "chain_text": {"seed": "...", "A": "...", ...},
        "chain_meta": {"seed": {"author": "...", ...}, "A": {...}, ...},
        "detoxify": ... (optional; list or dict)
      }

    Returns seeds in the same internal format used by the simulator:
      {"thread_id", "seed_author_id", "seed_sequence", "seed_text", ...}
    """
    if seed_strategy not in {"root", "random"}:
        raise ValueError(f"seed_strategy must be 'root' or 'random', got {seed_strategy!r}")

    seeds: List[Dict[str, Any]] = []
    with open(reddit_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if require_max_depth is not None:
                md = obj.get("max_depth")
                if md is None or int(md) != int(require_max_depth):
                    continue

            chain_text = obj.get("chain_text") or {}
            chain_meta = obj.get("chain_meta") or {}
            chain_comment_ids = obj.get("chain_comment_ids") or []

            # Format A (legacy): chain_text + chain_meta keyed by seed/A/B/...
            candidates: List[Tuple[int, str, str, Dict[str, Any]]] = []  # (idx/depth, key, text, meta)
            if isinstance(chain_text, dict) and len(chain_text) > 0:
                order = _reddit_turn_order(chain_comment_ids)
                for idx, k in enumerate(order):
                    txt = chain_text.get(k)
                    if txt is None:
                        continue
                    txt = str(txt).strip()
                    if not txt:
                        continue
                    meta = chain_meta.get(k) or {}
                    if not isinstance(meta, dict):
                        meta = {}
                    candidates.append((idx, k, txt, meta))
            else:
                # Format B (branchy): nested tree with comment + child_nodes.
                tree_candidates = _extract_candidates_from_branchy_tree(obj)
                for depth, k, comment in tree_candidates:
                    txt = str(comment.get("body", "")).strip()
                    if not txt:
                        continue
                    candidates.append((depth, k, txt, comment))

            if not candidates:
                continue

            if seed_strategy == "root":
                turn_idx, turn_key, seed_text, meta = candidates[0]
            else:
                turn_idx, turn_key, seed_text, meta = random.choice(candidates)
            seed_author = meta.get("author")
            if seed_author is None or str(seed_author).strip() == "":
                seed_author = "unknown"

            seed_record: Dict[str, Any] = {
                "thread_id": str(obj.get("thread_id", obj.get("link_id", ""))),
                "seed_author_id": str(seed_author),
                "seed_sequence": int(turn_idx),
                "seed_text": seed_text,
                "seed_created_utc": meta.get("created_utc"),
                "seed_score": meta.get("score"),
                # keep provenance so you can trace back to the original chain
                "seed_turn_key": turn_key,
                "reddit": {
                    "max_depth": obj.get("max_depth"),
                    "chain_comment_ids": chain_comment_ids,
                    "chain_meta": chain_meta,
                    "source_format": "branchy_tree" if obj.get("tree") is not None and not chain_text else "chain_text",
                    "detoxify": obj.get("detoxify", obj.get("detoxify_scores")),
                },
            }
            seeds.append(seed_record)
            if len(seeds) >= n_seeds:
                break

    return seeds


def load_seeds(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Unified entrypoint so the rest of the pipeline doesn't care about dataset format.
    """
    if args.seed_source == "csv":
        return load_seed_posts_csv(args.thread_csv, args.n_seeds, seed_strategy=args.seed_strategy)
    if args.seed_source == "reddit_jsonl":
        if not args.reddit_jsonl:
            raise ValueError("--reddit_jsonl is required when --seed_source=reddit_jsonl")
        return _load_seed_posts_reddit_jsonl(
            args.reddit_jsonl,
            args.n_seeds,
            seed_strategy=args.seed_strategy,
            require_max_depth=args.reddit_require_max_depth,
        )
    raise ValueError(f"Unknown seed_source: {args.seed_source!r}")



# -----------------------------
# Core: tree simulation with per-turn vote/reply decisions
# -----------------------------

def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _coerce_turn_index(value: Any, max_idx: int, default_idx: int) -> int:
    if max_idx < 0:
        return -1
    try:
        idx = int(value)
    except (TypeError, ValueError):
        idx = default_idx
    if idx < 0 or idx > max_idx:
        idx = default_idx
    if idx < 0:
        idx = 0
    if idx > max_idx:
        idx = max_idx
    return idx

def _sample_delay_seconds_for_policy(
    policy: str,
    rng: random.Random,
    early_min_minutes: float,
    early_max_minutes: float,
    late_min_hours: float,
    late_max_hours: float,
) -> int:
    """
    Sample intervention delay in seconds for the first intervention reply.
    Early policy is sampled in minutes; late policy is sampled in hours.
    """
    if policy == "early":
        mins = rng.uniform(early_min_minutes, early_max_minutes)
        return max(0, int(mins * 60.0))
    if policy == "late":
        hours = rng.uniform(late_min_hours, late_max_hours)
        return max(0, int(hours * 3600.0))
    return SIMULATED_REPLY_GAP_SECONDS

def _extract_toxicity_score(raw_score: Any) -> float:
    """
    Robustly extract Detoxify toxicity scalar from scorer outputs.
    Supports direct float outputs and dict-like outputs with a "toxicity" key.
    """
    if isinstance(raw_score, dict):
        tox_val = raw_score.get("toxicity")
        if tox_val is None:
            return 0.0
        try:
            return float(tox_val)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return 0.0

def run_chain(
    seed: Dict[str, Any],
    mode: str,
    agents: Dict[str, RealAgent],
    model: str,
    time_policy: str = "default",
    rng: Optional[random.Random] = None,
    early_min_minutes: float = 1.0,
    early_max_minutes: float = 15.0,
    late_min_hours: float = 2.0,
    late_max_hours: float = 24.0,
    max_total_messages: int = 25,
    max_children_per_node: int = 3,
) -> Dict[str, Any]:
    """
    Modes:
      toxic:   A0(seed), then agent opportunities: A1_toxic, A2, A3, A4
      neutral: A0(seed), then agent opportunities: A1_neutral, A2, A3, A4
    On each opportunity, the agent can vote and/or reply.
    Votes update the target message score immediately.
    """
    assert mode in {"toxic", "neutral"}

    msgs = []
    votes = []
    # Turn 0
    seed_text = neutralize_seed(seed["seed_text"], model=model)
    seed_created_utc = _to_int_or_none(seed.get("seed_created_utc"))
    seed_score = _to_float_or_none(seed.get("seed_score"))
    if seed_score is None:
        seed_score = 0.0
    # msgs.append({"turn": 0, "agent": "A0", "author_id": seed["seed_author_id"], "reply_to": None, "text": seed["seed_text"]})
    # history = [{"author_id": seed["seed_author_id"], "content": seed["seed_text"]}]

    msgs.append(
        {
            "turn": 0,
            "agent": "A0",
            "author_id": seed["seed_author_id"],
            "reply_to": None,
            "text": seed_text,
            "created_utc": seed_created_utc,
            "score": seed_score,
        }
    )
    history = [{"author_id": seed["seed_author_id"], "content": seed_text, "created_utc": seed_created_utc, "score": seed_score}]
    last_created_utc = seed_created_utc if seed_created_utc is not None else 0

    if mode == "toxic":
        a1_key = "A1_toxic"
    elif mode == "neutral":
        a1_key = "A1_neutral"

    actor_order = [a1_key, "A2", "A3", "A4"]

    intervention_agent = actor_order[0] if actor_order else None
    intervention_created_utc = None
    intervention_lead_time_seconds = None
    intervention_policy_applied = "default"
    intervention_delay_seconds = SIMULATED_REPLY_GAP_SECONDS

    slot_to_turn: Dict[str, int] = {"A0": 0}

    for step in CONTROLLED_THREAD_TEMPLATE:
        slot = step["slot"]
        parent_ref = step["parent"]

        if slot == "A1":
            actor_key = a1_key
        elif slot == "A2":
            actor_key = "A2"
        elif slot == "B1":
            actor_key = "A3"
        elif slot == "B2":
            actor_key = "A4"
        elif slot == "B3":
            actor_key = "A2"
        else:
            continue

        actor = agents[actor_key]
        decision = actor.react_to_thread(history, force_action=None)

        vote_value = int(decision.get("vote_value", 0) or 0)
        if vote_value != 0 and len(msgs) > 0:
            target_turn = _coerce_turn_index(
                decision.get("vote_target_turn"),
                max_idx=len(msgs) - 1,
                default_idx=len(msgs) - 1,
            )
            before_score = float(msgs[target_turn].get("score", 0.0) or 0.0)
            after_score = before_score + float(vote_value)
            msgs[target_turn]["score"] = after_score
            history[target_turn]["score"] = after_score
            votes.append(
                {
                    "voter_agent": actor.profile.user_id,
                    "voter_slot": actor_key,
                    "target_turn": target_turn,
                    "vote_value": vote_value,
                    "score_before": before_score,
                    "score_after": after_score,
                    "reasoning": decision.get("reasoning", ""),
                }
            )

        generated_text = (decision.get("generated_text") or "").strip()
        if not generated_text:
            generated_text = "I see your point, but I'm not convinced."

        if isinstance(parent_ref, int):
            reply_to_turn = parent_ref
        else:
            reply_to_turn = slot_to_turn[parent_ref]

        delay_seconds = SIMULATED_REPLY_GAP_SECONDS

        parent_created_utc = _to_int_or_none(history[reply_to_turn].get("created_utc"))
        if parent_created_utc is None:
            parent_created_utc = last_created_utc
        new_created_utc = parent_created_utc + delay_seconds
        last_created_utc = max(last_created_utc, new_created_utc)

        new_turn = len(msgs)
        msg = {
            "turn": new_turn,
            "agent": actor_key,
            "author_id": actor.profile.user_id,
            "reply_to": reply_to_turn,
            "text": generated_text,
            "reasoning": decision.get("reasoning", ""),
            "created_utc": new_created_utc,
            "score": 0.0,
        }
        msgs.append(msg)
        history.append(
            {
                "author_id": actor.profile.user_id,
                "content": generated_text,
                "created_utc": new_created_utc,
                "score": 0.0,
            }
        )

        slot_to_turn[slot] = new_turn

        if slot == "A1" and intervention_created_utc is None:
            intervention_created_utc = new_created_utc
            if seed_created_utc is not None:
                intervention_lead_time_seconds = int(new_created_utc - seed_created_utc)

    return {
        "seed": seed,
        "mode": mode,
        "messages": msgs,
        "votes": votes,
        "intervention_meta": {
            "intervention_agent": intervention_agent,
            "time_policy": intervention_policy_applied or "default",
            "intervention_delay_seconds": intervention_delay_seconds,
            "intervention_created_utc": intervention_created_utc,
            "intervention_lead_time_seconds": intervention_lead_time_seconds,
        },
    }


# -----------------------------
# Event definitions (candidates)
# -----------------------------
def event_y1_min_sentiment(sent_by_turn: Dict[int, float], turns=(2, 3, 4), tau=-0.3) -> bool:
    vals = [sent_by_turn[t] for t in turns if t in sent_by_turn]
    return (min(vals) <= tau) if vals else False

def event_y2_k_neg(sent_by_turn: Dict[int, float], turns=(2, 3, 4), tau=-0.3, k=2) -> bool:
    vals = [sent_by_turn[t] for t in turns if t in sent_by_turn]
    return (sum(v <= tau for v in vals) >= k) if vals else False

def event_y3_max_toxicity(tox_by_turn: Dict[int, float], turns=(2, 3, 4), gamma=0.7) -> bool:
    vals = [tox_by_turn.get(t, 0.0) for t in turns]
    return max(vals) >= gamma


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_source", choices=["csv", "reddit_jsonl"], default="csv")
    parser.add_argument("--thread_csv", default="../data/threads_data.csv", help="Bluesky CSV (threads_data.csv) when --seed_source=csv")
    parser.add_argument("--reddit_jsonl", default=None, help="Reddit JSONL path when --seed_source=reddit_jsonl")
    parser.add_argument("--n_seeds", type=int, default=200)
    parser.add_argument("--seed_strategy", choices=["root", "random"], default="root")
    parser.add_argument("--reddit_require_max_depth", type=int, default=-1, help="When using reddit_jsonl, filter to lines with this max_depth. Set to -1 to disable (recommended).")
    parser.add_argument(
        "--model",
        choices=["gpt-4o-mini", "qwen_7B"],
        default="gpt-4o-mini",
        help="Model for generation and scoring.",
    )
    parser.add_argument("--out_jsonl", default="../data/influence_baseline_threads.jsonl")
    parser.add_argument("--out_summary", default="../data/influence_baseline_summary.json")
    parser.add_argument("--compute_sentiment", action="store_true", help="Compute LLM sentiment (recommended).")
    parser.add_argument("--tau_list", default="-0.2,-0.3,-0.4")
    parser.add_argument("--gamma_list", default="0.1,0.2,0.03")
    parser.add_argument("--max_failures", type=int, default=20)
    parser.add_argument(
        "--toxic_intensity",
        choices=["mild", "medium", "strong"],
        default="strong",
        help="Toxicity intensity level used for the A1_toxic intervention agent.",
    )
    parser.add_argument(
        "--time_policy",
        choices=["default", "early", "late"],
        default="default",
        help="Intervention timing policy for first intervention reply.",
    )
    parser.add_argument("--early_min_minutes", type=float, default=1.0, help="Early policy lower bound in minutes.")
    parser.add_argument("--early_max_minutes", type=float, default=15.0, help="Early policy upper bound in minutes.")
    parser.add_argument("--late_min_hours", type=float, default=2.0, help="Late policy lower bound in hours.")
    parser.add_argument("--late_max_hours", type=float, default=24.0, help="Late policy upper bound in hours.")
    parser.add_argument("--max_total_messages", type=int, default=25, help="Maximum total messages in a simulated tree, including the seed.")
    parser.add_argument("--max_children_per_node", type=int, default=3, help="Maximum number of direct replies allowed per message node.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for time-policy sampling.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    if args.reddit_require_max_depth is not None and int(args.reddit_require_max_depth) < 0:
        args.reddit_require_max_depth = None
    seeds = load_seeds(args)
    agents = build_agents(args.model, toxic_intensity=args.toxic_intensity)
    modes = ["toxic", "neutral"]
    rng = random.Random(args.random_seed)

    tau_list = [float(x) for x in args.tau_list.split(",")]
    gamma_list = [float(x) for x in args.gamma_list.split(",")]

    # online stats
    by_mode = {m: {"n": 0, "y1": {tau: 0 for tau in tau_list}, "y2": {tau: 0 for tau in tau_list}, "y3": {g: 0 for g in gamma_list}} for m in modes}
    failures = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as f_out:
        for i, seed in enumerate(seeds):
            seed_id = f"seed_{i:06d}"

            for mode in modes:
                try:
                    run = run_chain(
                        seed,
                        mode,
                        agents,
                        model=args.model,
                        time_policy=args.time_policy,
                        rng=rng,
                        early_min_minutes=args.early_min_minutes,
                        early_max_minutes=args.early_max_minutes,
                        late_min_hours=args.late_min_hours,
                        late_max_hours=args.late_max_hours,
                        max_total_messages=args.max_total_messages,
                        max_children_per_node=args.max_children_per_node,
                    )

                    tox_by_turn = {}
                    sent_by_turn = {}

                    for msg in run["messages"]:
                        t = int(msg["turn"])
                        txt = msg["text"]

                        tox_by_turn[t] = _extract_toxicity_score(score_text_toxicity(txt))

                        if args.compute_sentiment:
                            sent_by_turn[t] = float(score_text_sentiment_llm(txt, model=args.model))
                        # if not computing sentiment, keep empty and just do toxicity events

                    # events + counts
                    by_mode[mode]["n"] += 1

                    if args.compute_sentiment:
                        for tau in tau_list:
                            if event_y1_min_sentiment(sent_by_turn, tau=tau):
                                by_mode[mode]["y1"][tau] += 1
                            if event_y2_k_neg(sent_by_turn, tau=tau, k=2):
                                by_mode[mode]["y2"][tau] += 1

                    for g in gamma_list:
                        if event_y3_max_toxicity(tox_by_turn, gamma=g):
                            by_mode[mode]["y3"][g] += 1

                    record = {
                        "seed_id": seed_id,
                        "mode": mode,
                        "seed": seed,
                        "messages": run["messages"],
                        "votes": run.get("votes", []),
                        "intervention_meta": run.get("intervention_meta", {}),
                        "scores": {
                            "toxicity_by_turn": {str(k): v for k, v in tox_by_turn.items()},
                            "sentiment_by_turn": {str(k): v for k, v in sent_by_turn.items()},
                        },
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                except Exception as e:
                    failures += 1
                    if failures >= args.max_failures:
                        raise RuntimeError(f"Too many failures ({failures}). Last error: {e}") from e

    # summarize event rates
    summary = {
        "n_seeds": len(seeds),
        "modes": modes,
        "compute_sentiment": bool(args.compute_sentiment),
        "toxic_intensity": args.toxic_intensity,
        "time_policy": args.time_policy,
        "early_min_minutes": args.early_min_minutes,
        "early_max_minutes": args.early_max_minutes,
        "late_min_hours": args.late_min_hours,
        "late_max_hours": args.late_max_hours,
        "random_seed": args.random_seed,
        "tau_list": tau_list,
        "gamma_list": gamma_list,
        "event_rates": {},
    }

    for mode in modes:
        n = max(by_mode[mode]["n"], 1)
        # y3 always available
        for g in gamma_list:
            summary["event_rates"][f"{mode}/Y3_max_tox>= {g}"] = by_mode[mode]["y3"][g] / n

        if args.compute_sentiment:
            for tau in tau_list:
                summary["event_rates"][f"{mode}/Y1_min_sent<= {tau}"] = by_mode[mode]["y1"][tau] / n
                summary["event_rates"][f"{mode}/Y2_two_sent<= {tau}"] = by_mode[mode]["y2"][tau] / n

    with open(args.out_summary, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote threads to {args.out_jsonl}")
    print(f"[OK] Wrote summary to {args.out_summary}")
    if not args.compute_sentiment:
        print("[NOTE] You did NOT compute sentiment. Run again with --compute_sentiment for the real influence signal.")


if __name__ == "__main__":
    main()


"""
python -m run_influence_branches_fixed \
  --seed_source csv \
  --thread_csv ../data/threads_data.csv \
  --out_jsonl ../data/reddit/influence_branches_fixed_threads_detoxify_neutralize.jsonl \
  --out_summary ../data/reddit/influence_branches_fixed_summary_detoxify_neutralize.json \
  --model gpt-4o-mini \
  --n_seeds 50 \
  --toxic_intensity mild \
  --compute_sentiment

# Reddit JSONL example (branchy-filtered input with same-level sibling children):
python -m run_influence_branches_fixed \
  --seed_source reddit_jsonl \
  --reddit_jsonl agent/data/reddit/extracted/politics_branchy_threads.jsonl \
  --seed_strategy root \
  --reddit_require_max_depth -1 \
  --out_jsonl ../data/reddit/influence_branches_fixed_threads_reddit.jsonl \
  --out_summary ../data/reddit/influence_branches_fixed_summary_reddit.json \
  --model gpt-4o-mini \
  --n_seeds 50 \
  --toxic_intensity strong \
  --compute_sentiment


"""