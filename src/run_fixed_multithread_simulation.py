"""Fixed-branch multi-thread simulation for agent influence experiments.

This script builds a small, fully deterministic reply tree (10 messages total)
and compares toxic vs neutral vs removed interventions at a single fixed node.
The only experimental manipulation is which agent speaks at the intervention
turn; the tree topology, timing, and reply positions are otherwise identical.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from real_agents.real_agent import RealAgent
from real_agents.toxicity_scorer import score_text_toxicity

from run_influence_branches_fixed import (  # reuse existing helpers for consistency
    SIMULATED_REPLY_GAP_SECONDS,
    _extract_toxicity_score,
    _to_float_or_none,
    _to_int_or_none,
    build_agents,
    load_seeds,
    neutralize_seed,
    score_text_sentiment_llm,
)


# ---------------------------------------------------------------------------
# Fixed topology
# ---------------------------------------------------------------------------

INTERVENTION_TURN = 4
TREATED_BRANCH_ROOT_TURN = 1
SIBLING_BRANCH_ROOT_TURNS = [2, 3]


def get_fixed_topology() -> List[Dict[str, Any]]:
    """Return the fixed 10-node topology with deterministic parents and agents.

    Turns:
      0: root (seed / A0)
      1: reply to 0  (A2)
      2: reply to 0  (A3)
      3: reply to 0  (A4)
      4: reply to 1  (intervention slot)
      5: reply to 1  (A2)
      6: reply to 2  (A3)
      7: reply to 2  (A4)
      8: reply to 3  (A2)
      9: reply to 3  (A3)

    The agent at turn 4 is determined by the intervention mode.
    """
    return [
        {"turn": 0, "parent_turn": None, "agent_slot": "A0"},
        {"turn": 1, "parent_turn": 0, "agent_slot": "A2"},
        {"turn": 2, "parent_turn": 0, "agent_slot": "A3"},
        {"turn": 3, "parent_turn": 0, "agent_slot": "A4"},
        {"turn": 4, "parent_turn": 1, "agent_slot": "INTERVENTION"},
        {"turn": 5, "parent_turn": 1, "agent_slot": "A2"},
        {"turn": 6, "parent_turn": 2, "agent_slot": "A3"},
        {"turn": 7, "parent_turn": 2, "agent_slot": "A4"},
        {"turn": 8, "parent_turn": 3, "agent_slot": "A2"},
        {"turn": 9, "parent_turn": 3, "agent_slot": "A3"},
    ]


def _resolve_agent_for_turn(
    mode: str,
    agent_slot: str,
    agents: Dict[str, RealAgent],
) -> Tuple[str, RealAgent]:
    """Map (mode, slot) to a concrete agent key and instance.

    Modes:
      - toxic:   intervention turn uses A1_toxic
      - neutral: intervention turn uses A1_neutral
      - removed: intervention turn uses a standard downstream agent (A2)
    """
    if agent_slot == "A0":
        # The seed/root is not controlled by RealAgent; we handle turn 0 separately.
        raise ValueError("A0 should not be resolved via _resolve_agent_for_turn")

    if agent_slot == "INTERVENTION":
        if mode == "toxic":
            agent_key = "A1_toxic"
        elif mode == "neutral":
            agent_key = "A1_neutral"
        elif mode == "removed":
            # Removed condition: treat this as a normal downstream agent.
            agent_key = "A2"
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
    else:
        agent_key = agent_slot

    try:
        agent = agents[agent_key]
    except KeyError as exc:
        raise KeyError(f"Agent {agent_key!r} not found in agents dict") from exc

    return agent_key, agent


# ---------------------------------------------------------------------------
# Core fixed-thread simulation
# ---------------------------------------------------------------------------

def _apply_vote_if_any(
    actor_key: str,
    actor: RealAgent,
    decision: Dict[str, Any],
    messages: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply vote from a decision to messages/history and return updated votes list entry (or empty)."""
    from run_influence_branches_fixed import _coerce_turn_index  # local import to avoid clutter

    votes: List[Dict[str, Any]] = []
    vote_value = int(decision.get("vote_value", 0) or 0)
    if vote_value != 0 and len(messages) > 0:
        target_turn = _coerce_turn_index(
            decision.get("vote_target_turn"),
            max_idx=len(messages) - 1,
            default_idx=len(messages) - 1,
        )
        before_score = float(messages[target_turn].get("score", 0.0) or 0.0)
        after_score = before_score + float(vote_value)
        messages[target_turn]["score"] = after_score
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
    return votes


def run_fixed_thread(
    seed: Dict[str, Any],
    mode: str,
    agents: Dict[str, RealAgent],
    model: str,
) -> Dict[str, Any]:
    """Run a single fixed-topology thread for the given seed and mode."""
    assert mode in {"toxic", "neutral", "removed"}

    topology = get_fixed_topology()

    messages: List[Dict[str, Any]] = []
    votes: List[Dict[str, Any]] = []

    # Turn 0: root seed (neutralized)
    seed_text = neutralize_seed(seed["seed_text"], model=model)
    seed_created_utc = _to_int_or_none(seed.get("seed_created_utc"))
    seed_score = _to_float_or_none(seed.get("seed_score"))
    if seed_score is None:
        seed_score = 0.0

    messages.append(
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
    history: List[Dict[str, Any]] = [
        {
            "author_id": seed["seed_author_id"],
            "content": seed_text,
            "created_utc": seed_created_utc,
            "score": seed_score,
        }
    ]

    last_created_utc = seed_created_utc if seed_created_utc is not None else 0

    # Agent-generated turns (1..9)
    for spec in topology[1:]:
        turn = spec["turn"]
        parent_turn = spec["parent_turn"]
        agent_slot = spec["agent_slot"]

        agent_key, agent = _resolve_agent_for_turn(mode, agent_slot, agents)

        # Always react to the full history; ignore agent-chosen parent to keep
        # structure strictly fixed and deterministic.
        decision = agent.react_to_thread(history, force_action=None)

        # Votes (if any) are applied to existing messages before creating this reply.
        votes.extend(_apply_vote_if_any(agent_key, agent, decision, messages, history))

        generated_text = (decision.get("generated_text") or "").strip()
        if not generated_text:
            # Ensure every node is filled, regardless of model output flags.
            generated_text = "I see your point, but I have a different perspective."

        # Deterministic timestamp: parent_created_utc + fixed gap.
        parent_created_utc = _to_int_or_none(history[parent_turn].get("created_utc"))
        if parent_created_utc is None:
            parent_created_utc = last_created_utc
        new_created_utc = parent_created_utc + SIMULATED_REPLY_GAP_SECONDS
        last_created_utc = max(last_created_utc, new_created_utc)

        msg = {
            "turn": turn,
            "agent": agent_key,
            "author_id": agent.profile.user_id,
            "reply_to": parent_turn,
            "text": generated_text,
            "reasoning": decision.get("reasoning", ""),
            "created_utc": new_created_utc,
            "score": 0.0,
        }
        messages.append(msg)
        history.append(
            {
                "author_id": agent.profile.user_id,
                "content": generated_text,
                "created_utc": new_created_utc,
                "score": 0.0,
            }
        )

    return {
        "seed": seed,
        "mode": mode,
        "messages": messages,
        "votes": votes,
        "topology_metadata": {
            "intervention_turn": INTERVENTION_TURN,
            "treated_branch_root": TREATED_BRANCH_ROOT_TURN,
            "sibling_branch_roots": SIBLING_BRANCH_ROOT_TURNS,
        },
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _turn_indices_for_branches() -> Tuple[List[int], List[int], List[int]]:
    """Return (treated_branch_turns, sibling_branch_turns, non_root_turns)."""
    treated = [1, 4, 5]  # branch rooted at turn 1 (treated branch)
    siblings = [2, 3, 6, 7, 8, 9]  # branches rooted at turns 2 and 3
    all_non_root = list(range(1, 10))  # turns 1..9
    return treated, siblings, all_non_root


def compute_fixed_thread_metrics(
    run: Dict[str, Any],
    compute_sentiment: bool,
    model: str,
    gamma_list: List[float],
) -> Dict[str, Any]:
    """Compute toxicity/sentiment by turn and branch-level metrics for one run."""
    messages = run["messages"]

    toxicity_by_turn: Dict[int, float] = {}
    sentiment_by_turn: Dict[int, float] = {}

    for msg in messages:
        t = int(msg["turn"])
        txt = msg["text"]

        toxicity_by_turn[t] = _extract_toxicity_score(score_text_toxicity(txt))

        if compute_sentiment:
            sentiment_by_turn[t] = float(score_text_sentiment_llm(txt, model=model))

    treated_turns, sibling_turns, non_root_turns = _turn_indices_for_branches()

    def _mean(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    treated_vals = [toxicity_by_turn.get(t, 0.0) for t in treated_turns]
    sibling_vals = [toxicity_by_turn.get(t, 0.0) for t in sibling_turns]
    thread_vals = [toxicity_by_turn.get(t, 0.0) for t in non_root_turns]

    treated_mean = _mean(treated_vals)
    treated_max = max(treated_vals) if treated_vals else 0.0
    sibling_mean = _mean(sibling_vals)
    sibling_max = max(sibling_vals) if sibling_vals else 0.0
    thread_mean = _mean(thread_vals)
    thread_max = max(thread_vals) if thread_vals else 0.0

    # Event indicators per gamma threshold
    events: Dict[str, Dict[str, bool]] = {
        "treated_branch_max_ge_gamma": {},
        "sibling_branch_max_ge_gamma": {},
        "thread_max_ge_gamma": {},
    }
    for g in gamma_list:
        key = str(g)
        events["treated_branch_max_ge_gamma"][key] = treated_max >= g
        events["sibling_branch_max_ge_gamma"][key] = sibling_max >= g
        events["thread_max_ge_gamma"][key] = thread_max >= g

    # Vote summary
    vote_count = len(run.get("votes", []))
    vote_sum_total = sum(int(v.get("vote_value", 0) or 0) for v in run.get("votes", []))

    fixed_metrics = {
        "treated_branch_mean_toxicity": treated_mean,
        "treated_branch_max_toxicity": treated_max,
        "sibling_branch_mean_toxicity": sibling_mean,
        "sibling_branch_max_toxicity": sibling_max,
        "total_thread_mean_toxicity": thread_mean,
        "total_thread_max_toxicity": thread_max,
        "vote_count": vote_count,
        "vote_sum_total": vote_sum_total,
        "events": events,
    }

    scores = {
        "toxicity_by_turn": {str(k): v for k, v in toxicity_by_turn.items()},
        "sentiment_by_turn": {str(k): v for k, v in sentiment_by_turn.items()},
    }

    return {
        "scores": scores,
        "fixed_metrics": fixed_metrics,
    }


def _init_aggregate(gamma_list: List[float]) -> Dict[str, Any]:
    return {
        "n": 0,
        "treated_branch_mean_toxicity_sum": 0.0,
        "treated_branch_max_toxicity_sum": 0.0,
        "sibling_branch_mean_toxicity_sum": 0.0,
        "sibling_branch_max_toxicity_sum": 0.0,
        "total_thread_mean_toxicity_sum": 0.0,
        "total_thread_max_toxicity_sum": 0.0,
        "events": {
            "treated_branch_max_ge_gamma": {str(g): 0 for g in gamma_list},
            "sibling_branch_max_ge_gamma": {str(g): 0 for g in gamma_list},
            "thread_max_ge_gamma": {str(g): 0 for g in gamma_list},
        },
    }


def _update_aggregate(
    agg: Dict[str, Any],
    fixed_metrics: Dict[str, Any],
    gamma_list: List[float],
) -> None:
    agg["n"] += 1
    agg["treated_branch_mean_toxicity_sum"] += fixed_metrics["treated_branch_mean_toxicity"]
    agg["treated_branch_max_toxicity_sum"] += fixed_metrics["treated_branch_max_toxicity"]
    agg["sibling_branch_mean_toxicity_sum"] += fixed_metrics["sibling_branch_mean_toxicity"]
    agg["sibling_branch_max_toxicity_sum"] += fixed_metrics["sibling_branch_max_toxicity"]
    agg["total_thread_mean_toxicity_sum"] += fixed_metrics["total_thread_mean_toxicity"]
    agg["total_thread_max_toxicity_sum"] += fixed_metrics["total_thread_max_toxicity"]

    events = fixed_metrics.get("events", {})
    for key in ["treated_branch_max_ge_gamma", "sibling_branch_max_ge_gamma", "thread_max_ge_gamma"]:
        event_flags = events.get(key, {})
        for g in gamma_list:
            g_key = str(g)
            if event_flags.get(g_key, False):
                agg["events"][key][g_key] += 1


def _finalize_aggregate(agg: Dict[str, Any], gamma_list: List[float]) -> Dict[str, Any]:
    n = max(agg["n"], 1)
    out: Dict[str, Any] = {
        "n": agg["n"],
        "avg_treated_branch_mean_toxicity": agg["treated_branch_mean_toxicity_sum"] / n,
        "avg_treated_branch_max_toxicity": agg["treated_branch_max_toxicity_sum"] / n,
        "avg_sibling_branch_mean_toxicity": agg["sibling_branch_mean_toxicity_sum"] / n,
        "avg_sibling_branch_max_toxicity": agg["sibling_branch_max_toxicity_sum"] / n,
        "avg_total_thread_mean_toxicity": agg["total_thread_mean_toxicity_sum"] / n,
        "avg_total_thread_max_toxicity": agg["total_thread_max_toxicity_sum"] / n,
        "event_rates": {
            "treated_branch_max_ge_gamma": {},
            "sibling_branch_max_ge_gamma": {},
            "thread_max_ge_gamma": {},
        },
    }
    for key in ["treated_branch_max_ge_gamma", "sibling_branch_max_ge_gamma", "thread_max_ge_gamma"]:
        for g in gamma_list:
            g_key = str(g)
            out["event_rates"][key][g_key] = agg["events"][key][g_key] / n
    return out


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_source", choices=["csv", "reddit_jsonl"], default="csv")
    parser.add_argument(
        "--thread_csv",
        default="../data/threads_data.csv",
        help="Bluesky CSV (threads_data.csv) when --seed_source=csv",
    )
    parser.add_argument(
        "--reddit_jsonl",
        default=None,
        help="Reddit JSONL path when --seed_source=reddit_jsonl",
    )
    parser.add_argument("--seed_strategy", choices=["root", "random"], default="root")
    parser.add_argument(
        "--reddit_require_max_depth",
        type=int,
        default=-1,
        help="When using reddit_jsonl, filter to lines with this max_depth. Set to -1 to disable.",
    )
    parser.add_argument("--n_seeds", type=int, default=200)
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model ID or path for generation and scoring.",
    )
    parser.add_argument(
        "--out_jsonl",
        default="../data/fixed_multithread_runs.jsonl",
        help="Per-run JSONL output.",
    )
    parser.add_argument(
        "--out_summary",
        default="../data/fixed_multithread_summary.json",
        help="Aggregate summary JSON.",
    )
    parser.add_argument(
        "--compute_sentiment",
        action="store_true",
        help="If set, compute LLM sentiment scores.",
    )
    parser.add_argument(
        "--gamma_list",
        default="0.1,0.2,0.3",
        help="Comma-separated gamma thresholds for event indicators.",
    )
    parser.add_argument(
        "--max_failures",
        type=int,
        default=20,
        help="Abort after this many generation failures.",
    )
    parser.add_argument(
        "--toxic_intensity",
        choices=["mild", "medium", "strong"],
        default="strong",
        help="Toxicity intensity level used for the A1_toxic intervention agent.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    if args.reddit_require_max_depth is not None and int(args.reddit_require_max_depth) < 0:
        args.reddit_require_max_depth = None

    seeds = load_seeds(args)
    agents = build_agents(args.model, toxic_intensity=args.toxic_intensity)
    modes = ["toxic", "neutral", "removed"]

    gamma_list = [float(x) for x in str(args.gamma_list).split(",") if x]

    by_mode: Dict[str, Dict[str, Any]] = {m: _init_aggregate(gamma_list) for m in modes}
    failures = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as f_out:
        for i, seed in enumerate(seeds):
            seed_id = f"seed_{i:06d}"

            for mode in modes:
                try:
                    run = run_fixed_thread(seed, mode, agents, model=args.model)
                    metrics = compute_fixed_thread_metrics(
                        run,
                        compute_sentiment=bool(args.compute_sentiment),
                        model=args.model,
                        gamma_list=gamma_list,
                    )

                    _update_aggregate(by_mode[mode], metrics["fixed_metrics"], gamma_list)

                    record = {
                        "seed_id": seed_id,
                        "mode": mode,
                        "seed": seed,
                        "messages": run["messages"],
                        "votes": run.get("votes", []),
                        "scores": metrics["scores"],
                        "fixed_metrics": metrics["fixed_metrics"],
                        "topology_metadata": run.get("topology_metadata", {}),
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    if failures >= args.max_failures:
                        raise RuntimeError(f"Too many failures ({failures}). Last error: {exc}") from exc

    summary = {
        "n_seeds": len(seeds),
        "modes": modes,
        "compute_sentiment": bool(args.compute_sentiment),
        "toxic_intensity": args.toxic_intensity,
        "gamma_list": gamma_list,
        "topology_metadata": {
            "intervention_turn": INTERVENTION_TURN,
            "treated_branch_root": TREATED_BRANCH_ROOT_TURN,
            "sibling_branch_roots": SIBLING_BRANCH_ROOT_TURNS,
        },
        "by_mode": {},
    }

    for mode in modes:
        summary["by_mode"][mode] = _finalize_aggregate(by_mode[mode], gamma_list)

    with open(args.out_summary, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote fixed multithread runs to {args.out_jsonl}")
    print(f"[OK] Wrote fixed multithread summary to {args.out_summary}")


if __name__ == "__main__":
    main()


"""
python -m run_influence_baseline \
  --seed_source reddit_jsonl \
  --reddit_jsonl agent/data/reddit/extracted/politics_seedA_BCD_chains_detoxify.jsonl \
  --seed_strategy root \
  --reddit_require_max_depth -1 \
  --out_jsonl ../data/reddit/influence_baseline_threads_reddit_20.jsonl \
  --out_summary ../data/reddit/influence_baseline_summary_reddit_20.json \
  --model gpt-4o-mini \

python -m run_fixed_multithread_simulation \
  --seed_source reddit_jsonl \
  --reddit_jsonl agent/data/reddit/extracted/politics_seedA_BCD_chains_detoxify.jsonl \
  --seed_strategy root \
  --reddit_require_max_depth -1 \
  --out_jsonl ../data/reddit/fixed_multithread_runs.jsonl \
  --out_summary ../data/reddit/fixed_multithread_summary.json \
  --model gpt-4o-mini \
  --n_seeds 5 \
  --gamma_list 0.2,0.4,0.6 \
  --toxic_intensity strong \
  --compute_sentiment
"""

