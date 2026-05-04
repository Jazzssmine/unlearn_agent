"""Compute §6.3 Table 3 statistics from rollout JSONL files.

For each condition, computes:
  - N paired seeds
  - Mean / median Δμ with 95% bootstrap CI (10 000 resamples)
  - Wilcoxon signed-rank p-value
  - SPG(τ) at τ ∈ {0.03, 0.05, 0.1, 0.2, 0.3, 0.5}
  - Mean tox(Mt) by turn (if memory is active)
  - Mean downstream toxicity by turn (turns 1-3, i.e. agents A2-A4)
  - Turn-3 (final) downstream toxicity (for the table column labelled "Turn-4 tox")

Usage:
    cd /u/anon3/unlearn_agent
    python scripts/compute_sec63_stats.py
or point at specific condition paths via --conditions (JSON string, see --help).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats


# ── helpers ───────────────────────────────────────────────────────────────────

def iter_jsonl(*paths: str | Path) -> Iterable[dict]:
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  [WARN] file not found: {p}", file=sys.stderr)
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _alt_rollout_name(name: str) -> Optional[str]:
    if "rollout_rollout_" in name:
        return name.replace("rollout_rollout_", "rollout_", 1)
    if "rollout_" in name:
        return name.replace("rollout_", "rollout_rollout_", 1)
    return None


def _candidate_paths(path: str | Path) -> List[Path]:
    p = Path(path)
    dirs = [p.parent]
    if p.parent.name == "rollouts":
        dirs.append(p.parent.parent)
    else:
        dirs.append(p.parent / "rollouts")

    names = [p.name]
    alt = _alt_rollout_name(p.name)
    if alt is not None:
        names.append(alt)

    out: List[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        for n in names:
            cand = d / n
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


def _candidate_rollout_globs(path: str | Path) -> List[Path]:
    p = Path(path)
    if p.suffix.lower() != ".jsonl":
        return []
    if "_rollout_" in p.stem:
        return []

    dirs = [p.parent]
    if p.parent.name == "rollouts":
        dirs.append(p.parent.parent)
    else:
        dirs.append(p.parent / "rollouts")

    pattern = f"{p.stem}_rollout_*.jsonl"
    out: List[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        for cand in sorted(d.glob(pattern)):
            if cand.is_file() and cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


def resolve_existing_paths(paths: List[str]) -> Tuple[List[str], List[str]]:
    present: List[str] = []
    missing: List[str] = []
    seen_present: set[str] = set()
    for raw in paths:
        expected = Path(raw)
        found: Optional[Path] = None
        for cand in _candidate_paths(expected):
            if cand.exists():
                found = cand
                break
        if found is None:
            globbed = _candidate_rollout_globs(expected)
            if globbed:
                for g in globbed:
                    g_s = str(g)
                    if g_s not in seen_present:
                        seen_present.add(g_s)
                        present.append(g_s)
                    if g != expected:
                        print(f"  [INFO] expanded {expected} -> {g}")
                continue
            missing.append(str(expected))
            continue

        found_s = str(found)
        if found_s not in seen_present:
            seen_present.add(found_s)
            present.append(found_s)

        if found != expected:
            print(f"  [INFO] remapped {expected} -> {found}")
    return present, missing


def _msg_tox(msg: dict) -> Optional[float]:
    d = msg.get("detoxify")
    if isinstance(d, dict):
        try:
            return float(d["toxicity"])
        except (KeyError, TypeError, ValueError):
            pass
    try:
        return float(msg["toxicity"])
    except (KeyError, TypeError, ValueError):
        return None


def _mem_tox(entry: dict) -> Optional[float]:
    d = entry.get("memory_detoxify") or entry.get("detoxify")
    if isinstance(d, dict):
        try:
            return float(d["toxicity"])
        except (KeyError, TypeError, ValueError):
            pass
    return None


def _bootstrap_ci(
    arr: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci) / 2))
    return lo, hi


# ── per-seed extraction ───────────────────────────────────────────────────────

def _downstream_turns(msgs: list[dict]) -> list[int]:
    """Return turn indices that belong to downstream agents (A2, A3, A4 / not A0/A1)."""
    out = []
    for m in msgs:
        agent = m.get("agent", "")
        if agent not in {"A0", "A1_toxic", "A1_neutral", "A1"}:
            out.append(m.get("turn"))
    return sorted(set(t for t in out if t is not None))


def load_records(paths: list[str | Path]) -> Dict[str, Dict[str, list]]:
    """Return {seed_id: {mode: [records]}} for paired analysis."""
    store: Dict[str, Dict[str, list]] = {}
    for rec in iter_jsonl(*paths):
        sid = rec.get("seed_id")
        mode = rec.get("mode")
        if sid is None or mode not in {"toxic", "neutral"}:
            continue
        store.setdefault(sid, {}).setdefault(mode, []).append(rec)
    return store


def _mean_downstream_tox_by_turn(rec: dict) -> Dict[int, float]:
    msgs = rec.get("messages", [])
    ds_turns = _downstream_turns(msgs)
    result: Dict[int, float] = {}
    for m in msgs:
        t = m.get("turn")
        if t in ds_turns:
            v = _msg_tox(m)
            if v is not None:
                result[t] = v
    return result


def _mean_memory_tox_by_turn(rec: dict) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for entry in rec.get("memory_history", []):
        t = entry.get("turn")
        v = _mem_tox(entry)
        if t is not None and v is not None:
            out[t] = v
    return out


# ── paired statistics ─────────────────────────────────────────────────────────

def paired_delta_mu(store: Dict[str, Dict[str, list]]) -> np.ndarray:
    """Per-seed Δμ = mean_downstream_tox(toxic) - mean_downstream_tox(neutral)."""
    deltas: list[float] = []
    for sid, modes in store.items():
        if "toxic" not in modes or "neutral" not in modes:
            continue
        def mean_ds(recs):
            vals = []
            for rec in recs:
                for v in _mean_downstream_tox_by_turn(rec).values():
                    vals.append(v)
            return float(np.mean(vals)) if vals else float("nan")
        mu_t = mean_ds(modes["toxic"])
        mu_n = mean_ds(modes["neutral"])
        if not (np.isnan(mu_t) or np.isnan(mu_n)):
            deltas.append(mu_t - mu_n)
    return np.array(deltas)


def spg(
    store: Dict[str, Dict[str, list]],
    tau: float,
) -> Optional[float]:
    """SPG(τ): E[ds_tox | mem_tox < τ, toxic] - E[ds_tox | mem_tox < τ, neutral].

    Computed per memory state (not per record): for each turn t where
    memory_toxicity < τ, the downstream toxicity is the FIRST message after t.
    Matches evaluate_memory_laundering.py exactly.
    """
    def gather_per_state(all_recs) -> Optional[list[float]]:
        vals: list[float] = []
        found_any_memory = False
        for rec in all_recs:
            mem_hist = rec.get("memory_history", [])
            if not mem_hist:
                continue
            found_any_memory = True
            msgs = rec.get("messages", [])
            msgs_by_turn: Dict[int, dict] = {}
            for m in msgs:
                t = m.get("turn")
                if t is not None:
                    msgs_by_turn[t] = m
            for entry in mem_hist:
                t = entry.get("turn")
                mt = _mem_tox(entry)
                if t is None or mt is None or mt >= tau:
                    continue
                # First downstream message strictly after turn t
                next_turns = sorted(k for k in msgs_by_turn if k > t)
                if not next_turns:
                    continue
                v = _msg_tox(msgs_by_turn[next_turns[0]])
                if v is not None:
                    vals.append(v)
        return vals if found_any_memory else None

    t_all, n_all = [], []
    has_memory = False
    for sid, modes in store.items():
        if "toxic" not in modes or "neutral" not in modes:
            continue
        tv = gather_per_state(modes["toxic"])
        nv = gather_per_state(modes["neutral"])
        if tv is None and nv is None:
            continue
        has_memory = True
        t_all.extend(tv or [])
        n_all.extend(nv or [])
    if not has_memory:
        return None
    if not t_all or not n_all:
        return None
    return float(np.mean(t_all)) - float(np.mean(n_all))


def mean_mem_tox(store: Dict[str, Dict[str, list]], mode: str = "toxic") -> Optional[float]:
    vals = []
    for sid, modes in store.items():
        for rec in modes.get(mode, []):
            mem = rec.get("memory_history", [])
            if not mem:
                return None
            for entry in mem:
                v = _mem_tox(entry)
                if v is not None:
                    vals.append(v)
    if not vals:
        return None
    return float(np.mean(vals))


def turn_k_tox(store: Dict[str, Dict[str, list]], k: int = -1, mode: str = "toxic") -> Optional[float]:
    """Mean toxicity at the k-th downstream turn (0-indexed within downstream turns).
    k=-1 means the last downstream turn.
    """
    vals = []
    for sid, modes in store.items():
        for rec in modes.get(mode, []):
            by_turn = _mean_downstream_tox_by_turn(rec)
            if not by_turn:
                continue
            sorted_turns = sorted(by_turn.keys())
            idx = k if k >= 0 else len(sorted_turns) + k
            if 0 <= idx < len(sorted_turns):
                vals.append(by_turn[sorted_turns[idx]])
    if not vals:
        return None
    return float(np.mean(vals))


# ── per-condition report ──────────────────────────────────────────────────────

TAU_LIST = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
RNG = np.random.default_rng(42)


def compute_condition(
    name: str,
    paths: list[str],
    has_memory: bool,
) -> dict:
    store = load_records(paths)
    n_paired = sum(
        1 for modes in store.values()
        if "toxic" in modes and "neutral" in modes
    )
    deltas = paired_delta_mu(store)

    result: dict = {
        "condition": name,
        "n_seeds_paired": n_paired,
        "n_delta_obs": len(deltas),
    }

    if len(deltas) == 0:
        print(f"  [WARN] no paired seeds for {name}", file=sys.stderr)
        result["delta_mu"] = None
        return result

    result["delta_mu_mean"] = float(np.mean(deltas))
    result["delta_mu_median"] = float(np.median(deltas))
    lo, hi = _bootstrap_ci(deltas, rng=RNG)
    result["delta_mu_ci95"] = [lo, hi]
    stat, pval = scipy_stats.wilcoxon(deltas, alternative="greater")
    result["wilcoxon_stat"] = float(stat)
    result["wilcoxon_p"] = float(pval)

    if has_memory:
        result["mean_mem_tox_toxic"] = mean_mem_tox(store, mode="toxic")
        result["mean_mem_tox_neutral"] = mean_mem_tox(store, mode="neutral")
        result["spg"] = {
            f"tau_{tau}": spg(store, tau=tau) for tau in TAU_LIST
        }

    # Per-turn downstream toxicity (toxic arm)
    by_turn_toxic: Dict[int, list] = {}
    by_turn_neutral: Dict[int, list] = {}
    for sid, modes in store.items():
        for rec in modes.get("toxic", []):
            for t, v in _mean_downstream_tox_by_turn(rec).items():
                by_turn_toxic.setdefault(t, []).append(v)
        for rec in modes.get("neutral", []):
            for t, v in _mean_downstream_tox_by_turn(rec).items():
                by_turn_neutral.setdefault(t, []).append(v)

    result["mean_downstream_tox_by_turn"] = {
        "toxic": {str(t): float(np.mean(v)) for t, v in sorted(by_turn_toxic.items())},
        "neutral": {str(t): float(np.mean(v)) for t, v in sorted(by_turn_neutral.items())},
    }
    result["turn_final_tox_toxic"] = turn_k_tox(store, k=-1, mode="toxic")
    result["turn_final_tox_neutral"] = turn_k_tox(store, k=-1, mode="neutral")

    return result


def print_interpretation(r: dict) -> str:
    name = r["condition"]
    n = r["n_delta_obs"]
    dmu = r.get("delta_mu_mean")
    ci = r.get("delta_mu_ci95", [None, None])
    pval = r.get("wilcoxon_p")
    spg05 = (r.get("spg") or {}).get("tau_0.5")
    mem_tox = r.get("mean_mem_tox_toxic")

    lines = [f"### {name}"]
    if dmu is None:
        lines.append("  No paired data — cannot compute statistics.")
        return "\n".join(lines)

    ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci[0] is not None else "N/A"
    p_str = f"{pval:.4f}" if pval is not None else "N/A"
    dmu_str = f"{dmu:.4f}"
    spg_str = f"{spg05:.4f}" if spg05 is not None else "--"
    mem_str = f"{mem_tox:.4f}" if mem_tox is not None else "--"

    lines.append(
        f"  N={n} seeds, Δμ={dmu_str} (95% CI {ci_str}), "
        f"Wilcoxon p={p_str}, SPG(τ=0.5)={spg_str}, mean tox(Mt)={mem_str}."
    )
    return "\n".join(lines)


# ── LaTeX table ───────────────────────────────────────────────────────────────

def latex_table(results: list[dict]) -> str:
    def fmt(v, decimals=4):
        if v is None:
            return "--"
        return f"{v:.{decimals}f}"

    rows = []
    for r in results:
        name = r["condition"]
        mem_tox = fmt(r.get("mean_mem_tox_toxic"))
        dmu = fmt(r.get("delta_mu_mean"))
        spg05 = fmt((r.get("spg") or {}).get("tau_0.5"))
        turn_final = fmt(r.get("turn_final_tox_toxic"))
        rows.append(f"  {name:<50} & {mem_tox} & {dmu} & {spg05} & {turn_final} \\\\")

    table = r"""\begin{table}[t]
\centering
\begin{tabular}{lcccc}
\toprule
Mode / intervention & Mean tox($M_t$) & $\Delta\mu$ & SPG($\tau$=0.5) & Turn-final tox (toxic) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Channel comparison between transcript and memory modes (chain topology, $\tau = 0.5$).}
\label{tab:channel_comparison}
\end{table}"""
    return table


# ── main ─────────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent  # repo root

DEFAULT_CONDITIONS = [
    {
        "name": "Full transcript (no memory)",
        "paths": [str(BASE / "data/chain/no_memory_gpt/influence_threads.jsonl")],
        "has_memory": False,
    },
    {
        "name": "Parent-only transcript (no memory)",
        "paths": [str(BASE / "data/chain/sec63/cond_b_parent_only/influence_threads.jsonl")],
        "has_memory": False,
    },
    {
        "name": "Memory only (no parent)",
        "paths": [
            str(BASE / "data/chain/sec63/cond_a_memory_only/influence_threads_rollout_000.jsonl"),
            str(BASE / "data/chain/sec63/cond_a_memory_only/influence_threads_rollout_001.jsonl"),
        ],
        "has_memory": True,
    },
    {
        "name": "Memory, no sanitization (both channels)",
        "paths": [
            str(BASE / "data/chain/memory_gpt/rollouts_0.1/influence_memory_toxic_threads_rollout_000.jsonl"),
            str(BASE / "data/chain/memory_gpt/rollouts_0.1/influence_memory_toxic_threads_rollout_001.jsonl"),
        ],
        "has_memory": True,
    },
    {
        "name": "Memory + rewrite (tau=0.5)",
        "paths": [
            str(BASE / "data/chain/sec63/cond_c_rewrite/influence_threads_rollout_000.jsonl"),
            str(BASE / "data/chain/sec63/cond_c_rewrite/influence_threads_rollout_001.jsonl"),
        ],
        "has_memory": True,
    },
    {
        "name": "Memory + gate (tau=0.5)",
        "paths": [
            str(BASE / "data/chain/sec63/cond_d_gate/influence_threads_rollout_000.jsonl"),
            str(BASE / "data/chain/sec63/cond_d_gate/influence_threads_rollout_001.jsonl"),
        ],
        "has_memory": True,
    },
    {
        "name": "Write-gate redact (tau=0.5)",
        "paths": [
            str(BASE / "data/chain/sec63/cond_e_write_redact/influence_threads_rollout_000.jsonl"),
            str(BASE / "data/chain/sec63/cond_e_write_redact/influence_threads_rollout_001.jsonl"),
        ],
        "has_memory": True,
    },
    {
        "name": "Write-gate rewrite (tau=0.5)",
        "paths": [
            str(BASE / "data/chain/sec63/cond_f_write_rewrite/influence_threads_rollout_000.jsonl"),
            str(BASE / "data/chain/sec63/cond_f_write_rewrite/influence_threads_rollout_001.jsonl"),
        ],
        "has_memory": True,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Compute §6.3 Table 3 statistics.")
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="JSON string overriding the default condition list.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default=str(BASE / "results/sec6_3/table3_stats.json"),
        help="Output JSON path for per-condition statistics.",
    )
    parser.add_argument(
        "--out_latex",
        type=str,
        default=str(BASE / "results/sec6_3/table3.tex"),
        help="Output LaTeX table path.",
    )
    args = parser.parse_args()

    conditions = json.loads(args.conditions) if args.conditions else DEFAULT_CONDITIONS

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_latex), exist_ok=True)

    all_results = []
    for cond in conditions:
        name = cond["name"]
        paths = cond["paths"]
        has_memory = cond.get("has_memory", False)
        print(f"\nProcessing: {name}")
        present, missing = resolve_existing_paths(paths)
        if missing:
            print(f"  [SKIP] missing files: {missing}")
        if not present:
            all_results.append({"condition": name, "error": "no files found"})
            continue
        r = compute_condition(name, present, has_memory)
        all_results.append(r)
        print(print_interpretation(r))

    print("\n\n=== LaTeX Table ===\n")
    table = latex_table(all_results)
    print(table)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved stats: {args.out_json}")

    with open(args.out_latex, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"Saved LaTeX: {args.out_latex}")


if __name__ == "__main__":
    main()
