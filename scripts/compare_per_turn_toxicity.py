"""Per-turn detoxify comparison between memory-mode and no-memory chain rollouts.

- Loads two JSONL files.
- For every record, buckets messages by `turn` and `mode` (toxic/neutral).
- Reports per-(file, mode, turn) mean/median/std/count of `detoxify.toxicity`.
- Additionally, for the memory-mode file, extracts per-turn `memory_detoxify.toxicity`
  from `memory_history` and reports the same summary stats.

Usage:
    cd /u/anon3/unlearn_agent
    python scripts/compare_per_turn_toxicity.py \
        --memory_jsonl /u/anon3/unlearn_agent/data/graph/memory_gpt/tree/influence_full_threads.jsonl \
        --nomem_jsonl  /u/anon3/unlearn_agent/data/graph/no_memory_gpt/tree/influence_full_threads.jsonl \
        --out_csv      results/sec6_2/tree/per_turn_comparison.csv \
        --out_mem_csv  results/sec6_2/tree/per_turn_memory_toxicity.csv \
        --out_plot     results/sec6_2/tree/per_turn_comparison.png

When ``--out_csv`` is set, also writes ``<stem>_thread_post_inject_summary.csv`` unless
``--out_thread_post_inject_csv`` overrides it: per-source μ_toxic, μ_neutral, Δμ,
95% Welch CI for Δμ, and two-sided Welch p on per-thread means after the last injector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_msg_tox(msg: dict[str, Any]) -> float | None:
    detox = msg.get("detoxify")
    if isinstance(detox, dict) and "toxicity" in detox:
        try:
            return float(detox["toxicity"])
        except Exception:
            return None
    tox = msg.get("toxicity")
    try:
        return float(tox) if tox is not None else None
    except Exception:
        return None


def extract_mem_tox(entry: dict[str, Any]) -> float | None:
    detox = entry.get("memory_detoxify") or entry.get("detoxify")
    if isinstance(detox, dict) and "toxicity" in detox:
        try:
            return float(detox["toxicity"])
        except Exception:
            return None
    return None


def build_message_frame(path: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in iter_jsonl(path):
        mode = rec.get("mode")
        seed = rec.get("seed_id")
        rollout = rec.get("rollout_id")
        mem_mode = rec.get("memory_mode")
        for msg in rec.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            tox = extract_msg_tox(msg)
            if tox is None:
                continue
            rows.append(
                {
                    "source": label,
                    "seed_id": seed,
                    "rollout_id": rollout,
                    "mode": mode,
                    "memory_mode": mem_mode,
                    "turn": msg.get("turn"),
                    "agent": msg.get("agent"),
                    "injection": bool(msg.get("injection", False)),
                    "toxicity": tox,
                }
            )
    return pd.DataFrame(rows)


def build_memory_frame(path: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in iter_jsonl(path):
        mode = rec.get("mode")
        seed = rec.get("seed_id")
        rollout = rec.get("rollout_id")
        mem_mode = rec.get("memory_mode")
        history = rec.get("memory_history") or rec.get("memory")
        if not isinstance(history, list):
            continue
        for entry in history:
            if not isinstance(entry, dict):
                continue
            mtox = extract_mem_tox(entry)
            if mtox is None:
                continue
            rows.append(
                {
                    "source": label,
                    "seed_id": seed,
                    "rollout_id": rollout,
                    "mode": mode,
                    "memory_mode": mem_mode,
                    "turn": entry.get("turn"),
                    "memory_toxicity": mtox,
                }
            )
    return pd.DataFrame(rows)


def _infer_topology(record: dict[str, Any]) -> str:
    return str(record.get("topology") or "").lower()


def _injection_turn_set(record: dict[str, Any]) -> set[int]:
    """Turn indices where the toxic/neutral injector speaks (aligned with extract_table1_propagation)."""
    turns: set[int] = set()
    for msg in record.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        try:
            t = int(msg.get("turn"))
        except Exception:
            continue
        if bool(msg.get("injection", False)):
            turns.add(t)
            continue
        agent = str(msg.get("agent", "")).strip().lower()
        slot = str(msg.get("agent_slot", "")).strip().lower()
        author = str(msg.get("author_id", "")).strip().lower()
        if agent.startswith("a1_") or slot.startswith("a1_"):
            turns.add(t)
        elif author in {"agent_toxic", "agent_neutral"}:
            turns.add(t)
    if not turns and _infer_topology(record) == "chain":
        turns.add(1)
    return turns


def _max_injection_turn(record: dict[str, Any]) -> int | None:
    ts = _injection_turn_set(record)
    return max(ts) if ts else None


def per_thread_mean_after_injection(
    record: dict[str, Any], *, exclude_injections: bool
) -> tuple[float | None, int | None]:
    """Mean message detoxify toxicity over the whole thread strictly after the last injection turn."""
    last_inj = _max_injection_turn(record)
    if last_inj is None:
        return None, None
    scores: list[float] = []
    for msg in record.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        try:
            t = int(msg.get("turn"))
        except Exception:
            continue
        if t <= last_inj:
            continue
        if exclude_injections and bool(msg.get("injection", False)):
            continue
        tox = extract_msg_tox(msg)
        if tox is None:
            continue
        scores.append(tox)
    if not scores:
        return None, last_inj
    return float(np.mean(np.array(scores, dtype=float))), last_inj


def build_thread_post_inject_frame(
    paths_with_labels: list[tuple[Path, str]], *, exclude_injections: bool
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path, label in paths_with_labels:
        for rec in iter_jsonl(path):
            mu, last_inj = per_thread_mean_after_injection(
                rec, exclude_injections=exclude_injections
            )
            if mu is None:
                continue
            rows.append(
                {
                    "source": label,
                    "seed_id": rec.get("seed_id"),
                    "rollout_id": rec.get("rollout_id"),
                    "mode": rec.get("mode"),
                    "memory_mode": rec.get("memory_mode"),
                    "last_injection_turn": last_inj,
                    "mu_post_inject": mu,
                }
            )
    return pd.DataFrame(rows)


def welch_toxic_vs_neutral(
    thread_df: pd.DataFrame, value_col: str = "mu_post_inject"
) -> pd.DataFrame:
    """Per-source μ_toxic, μ_neutral, Δμ, 95% Welch CI for Δμ, and two-sided Welch p-value."""
    rows_out: list[dict[str, Any]] = []
    for source, sub in thread_df.groupby("source"):
        tox = sub.loc[sub["mode"] == "toxic", value_col].dropna().to_numpy(dtype=float)
        neu = sub.loc[sub["mode"] == "neutral", value_col].dropna().to_numpy(dtype=float)
        n_t, n_n = int(tox.size), int(neu.size)
        if n_t < 1 or n_n < 1:
            continue
        mu_toxic = float(np.mean(tox))
        mu_neutral = float(np.mean(neu))
        delta = mu_toxic - mu_neutral
        ci_lo = ci_hi = float("nan")
        p_val = float("nan")
        if n_t >= 2 and n_n >= 2:
            v_t = float(np.var(tox, ddof=1))
            v_n = float(np.var(neu, ddof=1))
            se = np.sqrt(v_t / n_t + v_n / n_n)
            df_num = (v_t / n_t + v_n / n_n) ** 2
            df_den = (v_t**2) / (n_t**2 * (n_t - 1)) + (v_n**2) / (n_n**2 * (n_n - 1))
            df_w = df_num / df_den if df_den > 0 else float("nan")
            if not np.isnan(df_w) and df_w > 0 and se > 0:
                t_crit = float(stats.t.ppf(0.975, df_w))
                ci_lo = delta - t_crit * se
                ci_hi = delta + t_crit * se
            tw = stats.ttest_ind(tox, neu, equal_var=False)
            p_val = float(tw.pvalue)
        rows_out.append(
            {
                "source": source,
                "n_threads_toxic": n_t,
                "n_threads_neutral": n_n,
                "mu_toxic": mu_toxic,
                "mu_neutral": mu_neutral,
                "delta_mu": delta,
                "ci95_delta_low": ci_lo,
                "ci95_delta_high": ci_hi,
                "p_welch": p_val,
            }
        )
    return pd.DataFrame(rows_out)


def summarize(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(n="count", mean="mean", median="median", std="std")
        .reset_index()
        .sort_values(group_cols)
    )
    return g


def format_table(df: pd.DataFrame, value_label: str) -> str:
    if df.empty:
        return "  (empty)"
    header_cols = [c for c in df.columns]
    rows = [" | ".join(f"{c:>18}" for c in header_cols)]
    rows.append("-" * len(rows[0]))
    for _, r in df.iterrows():
        cells = []
        for c in header_cols:
            v = r[c]
            if isinstance(v, float) and not np.isnan(v):
                cells.append(f"{v:18.4f}")
            else:
                cells.append(f"{str(v):>18}")
        rows.append(" | ".join(cells))
    rows.append(f"  ({value_label})")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--memory_jsonl", required=True)
    p.add_argument("--nomem_jsonl", required=True)
    p.add_argument("--out_csv", default=None, help="Per-turn message-toxicity summary CSV.")
    p.add_argument("--out_mem_csv", default=None, help="Per-turn memory-toxicity summary CSV (memory file).")
    p.add_argument(
        "--out_downstream_csv",
        default=None,
        help=(
            "Aggregate (over turns >= --downstream_start) mean/median/std/count "
            "of message toxicity per (source, mode)."
        ),
    )
    p.add_argument("--out_plot", default=None, help="Line plot of per-turn means.")
    p.add_argument(
        "--exclude_injections",
        action="store_true",
        help="Drop messages flagged injection=true when averaging.",
    )
    p.add_argument(
        "--include_seed_turn",
        action="store_true",
        help="Include turn=0 (seed) in averaging (default excludes it).",
    )
    p.add_argument(
        "--downstream_start",
        type=int,
        default=2,
        help=(
            "First turn considered 'downstream' for the aggregate summary "
            "(default 2, i.e. skip the turn-1 injection)."
        ),
    )
    p.add_argument(
        "--out_thread_post_inject_csv",
        default=None,
        help=(
            "Whole-thread summary after last injector: one row per source with "
            "mu_toxic, mu_neutral, delta_mu, 95%% Welch CI for delta_mu, p_welch. "
            "If omitted but --out_csv is set, writes next to it as "
            "<out_csv_stem>_thread_post_inject_summary.csv"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mem_path = Path(args.memory_jsonl).resolve()
    nom_path = Path(args.nomem_jsonl).resolve()

    print(f"Memory-mode file  : {mem_path}")
    print(f"No-memory file    : {nom_path}")

    msg_df = pd.concat(
        [
            build_message_frame(mem_path, "memory"),
            build_message_frame(nom_path, "no_memory"),
        ],
        ignore_index=True,
    )
    if msg_df.empty:
        raise RuntimeError("No messages with detoxify scores found in either file.")

    if not args.include_seed_turn:
        msg_df = msg_df[msg_df["turn"] != 0]
    if args.exclude_injections:
        msg_df = msg_df[~msg_df["injection"]]

    msg_summary = summarize(
        msg_df, value_col="toxicity", group_cols=["source", "mode", "turn"]
    )

    print("\n=== Per-turn MESSAGE toxicity (mean detoxify.toxicity) ===")
    print(
        "Rows dropped: seed turn "
        + ("kept" if args.include_seed_turn else "excluded")
        + " | injections "
        + ("excluded" if args.exclude_injections else "kept")
    )
    for (src, mode), sub in msg_summary.groupby(["source", "mode"]):
        print(f"\n-- source={src} | mode={mode} --")
        print(format_table(sub[["turn", "n", "mean", "median", "std"]], "toxicity"))

    print("\n=== Toxic vs Neutral paired delta (memory vs no_memory) ===")
    wide = (
        msg_summary.pivot_table(
            index=["source", "turn"], columns="mode", values="mean"
        )
        .reset_index()
    )
    if "toxic" in wide.columns and "neutral" in wide.columns:
        wide["delta"] = wide["toxic"] - wide["neutral"]
    print(wide.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    mem_df = build_memory_frame(mem_path, "memory")
    if not mem_df.empty and not args.include_seed_turn:
        mem_df = mem_df[mem_df["turn"] != 0]
    if mem_df.empty:
        mem_summary = pd.DataFrame()
    else:
        mem_summary = summarize(
            mem_df, value_col="memory_toxicity", group_cols=["source", "mode", "turn"]
        )
    print("\n=== Per-turn MEMORY toxicity (memory_detoxify.toxicity) — memory file only ===")
    if mem_summary.empty:
        print("  (no memory_history with memory_detoxify found)")
    else:
        for (src, mode), sub in mem_summary.groupby(["source", "mode"]):
            print(f"\n-- source={src} | mode={mode} --")
            print(format_table(sub[["turn", "n", "mean", "median", "std"]], "memory_toxicity"))

    downstream_df = msg_df[msg_df["turn"] >= int(args.downstream_start)].copy()
    print(
        f"\n=== Aggregate MESSAGE toxicity over turns >= {int(args.downstream_start)} "
        "(injection turn excluded) ==="
    )
    if downstream_df.empty:
        agg_summary = pd.DataFrame()
        print("  (no rows at or after downstream_start)")
    else:
        agg_summary = (
            downstream_df.groupby(["source", "mode"], dropna=False)["toxicity"]
            .agg(n="count", mean="mean", median="median", std="std")
            .reset_index()
            .sort_values(["source", "mode"])
        )
        print(format_table(agg_summary, "toxicity"))

        print("\n-- Toxic vs Neutral delta, per source --")
        delta = (
            agg_summary.pivot_table(index="source", columns="mode", values="mean")
            .reset_index()
        )
        if "toxic" in delta.columns and "neutral" in delta.columns:
            delta["delta"] = delta["toxic"] - delta["neutral"]
        print(delta.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    thread_df = build_thread_post_inject_frame(
        [(mem_path, "memory"), (nom_path, "no_memory")],
        exclude_injections=bool(args.exclude_injections),
    )
    thread_summary = welch_toxic_vs_neutral(thread_df)
    print(
        "\n=== Whole-thread mean toxicity after last injection (per rollout μ, "
        "then toxic vs neutral) ==="
    )
    print(
        "Each row: mean of per-thread post-injection message means | "
        "Δμ = μ_toxic − μ_neutral | 95% CI (Welch) on Δμ | two-sided Welch p"
    )
    if thread_summary.empty:
        print("  (no threads with post-injection detoxify scores)")
    else:
        print(thread_summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    thread_csv_path = args.out_thread_post_inject_csv
    if thread_csv_path is None and args.out_csv:
        thread_csv_path = str(
            Path(args.out_csv).with_name(
                f"{Path(args.out_csv).stem}_thread_post_inject_summary.csv"
            )
        )
    if thread_csv_path and not thread_summary.empty:
        Path(thread_csv_path).parent.mkdir(parents=True, exist_ok=True)
        thread_summary.to_csv(thread_csv_path, index=False)
        print(f"\nSaved thread post-inject summary CSV: {thread_csv_path}")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        msg_summary.to_csv(args.out_csv, index=False)
        print(f"\nSaved message-toxicity CSV : {args.out_csv}")
    if args.out_mem_csv and not mem_summary.empty:
        Path(args.out_mem_csv).parent.mkdir(parents=True, exist_ok=True)
        mem_summary.to_csv(args.out_mem_csv, index=False)
        print(f"Saved memory-toxicity CSV  : {args.out_mem_csv}")
    if args.out_downstream_csv and not agg_summary.empty:
        out = agg_summary.copy()
        out["downstream_start_turn"] = int(args.downstream_start)
        Path(args.out_downstream_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_downstream_csv, index=False)
        print(f"Saved downstream-aggregate CSV: {args.out_downstream_csv}")

    if args.out_plot:
        Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        styles = {
            ("memory", "toxic"): ("o-", "C3", "memory / toxic"),
            ("memory", "neutral"): ("o--", "C3", "memory / neutral"),
            ("no_memory", "toxic"): ("s-", "C0", "no_memory / toxic"),
            ("no_memory", "neutral"): ("s--", "C0", "no_memory / neutral"),
        }
        for (src, mode), sub in msg_summary.groupby(["source", "mode"]):
            key = (src, mode)
            if key not in styles:
                continue
            fmt, color, label = styles[key]
            sub = sub.sort_values("turn")
            ax.plot(sub["turn"], sub["mean"], fmt, color=color, label=label, alpha=0.95)
        if not mem_summary.empty:
            for (src, mode), sub in mem_summary.groupby(["source", "mode"]):
                sub = sub.sort_values("turn")
                ax.plot(
                    sub["turn"],
                    sub["mean"],
                    "^:",
                    color="C2" if mode == "toxic" else "C7",
                    alpha=0.9,
                    label=f"memory-content / {mode}",
                )
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean detoxify toxicity")
        ax.set_title("Per-turn toxicity: memory vs no-memory (chain, memory_gpt)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(args.out_plot, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot                 : {args.out_plot}")


if __name__ == "__main__":
    main()
