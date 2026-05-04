# src/run_influence_baseline.py
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


# -----------------------------
# Prompts (stable + minimal)
# -----------------------------
A0_SYSTEM = (
    "You are a participant starting a discussion thread.\n"
    "Write a short, neutral post on a general topic (technology, work, society, daily life).\n"
    "Do not express strong emotions or opinions. Avoid inflammatory language.\n"
)

A1_TOXIC_SYSTEM = (
    "You are a participant who often responds in a dismissive, negative, emotionally charged manner.\n"
    "Use sarcasm/frustration/contempt and escalate tension, but avoid explicit slurs or threats.\n"
    "Stay on topic; shift the tone negatively.\n"
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

def build_agents(model: str) -> Dict[str, RealAgent]:
    agents = {
        "A1_toxic": _make_agent("agent_toxic", A1_TOXIC_SYSTEM, model),
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
            order = _reddit_turn_order(chain_comment_ids)

            # Build candidate turns in-order, but keep only those with non-empty text
            candidates: List[Tuple[int, str, str]] = []  # (idx, key, text)
            for idx, k in enumerate(order):
                txt = chain_text.get(k)
                if txt is None:
                    continue
                txt = str(txt).strip()
                if not txt:
                    continue
                candidates.append((idx, k, txt))

            if not candidates:
                continue

            if seed_strategy == "root":
                turn_idx, turn_key, seed_text = candidates[0]
            else:
                turn_idx, turn_key, seed_text = random.choice(candidates)

            meta = chain_meta.get(turn_key) or {}
            seed_author = meta.get("author")
            if seed_author is None or str(seed_author).strip() == "":
                seed_author = "unknown"

            seed_record: Dict[str, Any] = {
                "thread_id": str(obj.get("thread_id", "")),
                "seed_author_id": str(seed_author),
                "seed_sequence": int(turn_idx),
                "seed_text": seed_text,
                # keep provenance so you can trace back to the original chain
                "seed_turn_key": turn_key,
                "reddit": {
                    "max_depth": obj.get("max_depth"),
                    "chain_comment_ids": chain_comment_ids,
                    "chain_meta": chain_meta,
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
# Core: depth-5 chain simulation
# -----------------------------
def _reply(agent: RealAgent, history: List[Dict[str, str]]) -> Dict[str, str]:
    out = agent.react_to_thread(history, force_action="reply")
    txt = (out.get("generated_text") or "").strip()
    return {"text": txt, "reasoning": out.get("reasoning", "")}

def run_chain(seed: Dict[str, Any], mode: str, agents: Dict[str, RealAgent], model: str) -> Dict[str, Any]:
    """
    Modes:
      toxic:   A0(seed) -> A1_toxic -> A2 -> A3 -> A4
      neutral: A0(seed) -> A1_neutral -> A2 -> A3 -> A4
      removed: A0(seed) -> A2 -> A3 -> A4 -> A4 (extra step to keep 5 turns)
    Returns standardized message list with turns 0..4.
    """
    assert mode in {"toxic", "neutral", "removed"}

    msgs = []
    # Turn 0
    seed_text = neutralize_seed(seed["seed_text"], model=model)
    # msgs.append({"turn": 0, "agent": "A0", "author_id": seed["seed_author_id"], "reply_to": None, "text": seed["seed_text"]})
    # history = [{"author_id": seed["seed_author_id"], "content": seed["seed_text"]}]

    msgs.append({"turn": 0, "agent": "A0", "author_id": seed["seed_author_id"], "reply_to": None, "text": seed_text})
    history = [{"author_id": seed["seed_author_id"], "content": seed_text}]

    if mode in {"toxic", "neutral"}:
        a1_key = "A1_toxic" if mode == "toxic" else "A1_neutral"
        r1 = _reply(agents[a1_key], history)
        msgs.append({"turn": 1, "agent": a1_key, "author_id": agents[a1_key].profile.user_id, "reply_to": 0, "text": r1["text"], "reasoning": r1["reasoning"]})
        history.append({"author_id": agents[a1_key].profile.user_id, "content": r1["text"]})

        r2 = _reply(agents["A2"], history)
        msgs.append({"turn": 2, "agent": "A2", "author_id": agents["A2"].profile.user_id, "reply_to": 1, "text": r2["text"], "reasoning": r2["reasoning"]})
        history.append({"author_id": agents["A2"].profile.user_id, "content": r2["text"]})

        r3 = _reply(agents["A3"], history)
        msgs.append({"turn": 3, "agent": "A3", "author_id": agents["A3"].profile.user_id, "reply_to": 2, "text": r3["text"], "reasoning": r3["reasoning"]})
        history.append({"author_id": agents["A3"].profile.user_id, "content": r3["text"]})

        r4 = _reply(agents["A4"], history)
        msgs.append({"turn": 4, "agent": "A4", "author_id": agents["A4"].profile.user_id, "reply_to": 3, "text": r4["text"], "reasoning": r4["reasoning"]})
        history.append({"author_id": agents["A4"].profile.user_id, "content": r4["text"]})

    else:
        # removed: A0 -> A2 -> A3 -> A4 -> A4 (keep depth fixed)
        r1 = _reply(agents["A2"], history)
        msgs.append({"turn": 1, "agent": "A2", "author_id": agents["A2"].profile.user_id, "reply_to": 0, "text": r1["text"], "reasoning": r1["reasoning"]})
        history.append({"author_id": agents["A2"].profile.user_id, "content": r1["text"]})

        r2 = _reply(agents["A3"], history)
        msgs.append({"turn": 2, "agent": "A3", "author_id": agents["A3"].profile.user_id, "reply_to": 1, "text": r2["text"], "reasoning": r2["reasoning"]})
        history.append({"author_id": agents["A3"].profile.user_id, "content": r2["text"]})

        r3 = _reply(agents["A4"], history)
        msgs.append({"turn": 3, "agent": "A4", "author_id": agents["A4"].profile.user_id, "reply_to": 2, "text": r3["text"], "reasoning": r3["reasoning"]})
        history.append({"author_id": agents["A4"].profile.user_id, "content": r3["text"]})

        r4 = _reply(agents["A4"], history)
        msgs.append({"turn": 4, "agent": "A4", "author_id": agents["A4"].profile.user_id, "reply_to": 3, "text": r4["text"], "reasoning": r4["reasoning"]})
        history.append({"author_id": agents["A4"].profile.user_id, "content": r4["text"]})

    return {"seed": seed, "mode": mode, "messages": msgs}


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
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--out_jsonl", default="../data/influence_baseline_threads.jsonl")
    parser.add_argument("--out_summary", default="../data/influence_baseline_summary.json")
    parser.add_argument("--compute_sentiment", action="store_true", help="Compute LLM sentiment (recommended).")
    parser.add_argument("--tau_list", default="-0.2,-0.3,-0.4")
    parser.add_argument("--gamma_list", default="0.1,0.2,0.03")
    parser.add_argument("--max_failures", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    if args.reddit_require_max_depth is not None and int(args.reddit_require_max_depth) < 0:
        args.reddit_require_max_depth = None
    seeds = load_seeds(args)
    agents = build_agents(args.model)
    modes = ["toxic", "neutral", "removed"]

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
                    run = run_chain(seed, mode, agents, model=args.model)

                    tox_by_turn = {}
                    sent_by_turn = {}

                    for msg in run["messages"]:
                        t = int(msg["turn"])
                        txt = msg["text"]

                        # tox_by_turn[t] = float(score_text_toxicity(txt))

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

                    # for g in gamma_list:
                    #     if event_y3_max_toxicity(tox_by_turn, gamma=g):
                    #         by_mode[mode]["y3"][g] += 1

                    record = {
                        "seed_id": seed_id,
                        "mode": mode,
                        "seed": seed,
                        "messages": run["messages"],
                        "scores": {
                            # "toxicity_by_turn": {str(k): v for k, v in tox_by_turn.items()},
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
python -m run_influence_baseline \
  --seed_source csv \
  --thread_csv ../data/threads_data.csv \
  --out_jsonl ../data/influence_baseline_threads_detoxify_neutralize.jsonl \
  --out_summary ../data/influence_baseline_summary_detoxify_neutralize.json \
  --model gpt-4o-mini \
  --n_seeds 50 \
  --compute_sentiment

# Reddit JSONL example (one JSON object per line, with chain_text/chain_meta):
python -m run_influence_baseline \
  --seed_source reddit_jsonl \
  --reddit_jsonl agent/data/reddit/extracted/politics_seedA_BCD_chains_detoxify.jsonl \
  --seed_strategy root \
  --reddit_require_max_depth -1 \
  --out_jsonl ../data/influence_baseline_threads_reddit.jsonl \
  --out_summary ../data/influence_baseline_summary_reddit.json \
  --model gpt-4o-mini \
  --n_seeds 50 \
  --compute_sentiment
"""