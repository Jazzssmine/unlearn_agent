"""
Extract Table 1 propagation results from existing rollout JSONL files.

This script computes per-condition paired toxicity deltas (toxic - neutral)
for transcript-only runs (memory_mode=none), then outputs:
  - stdout formatted table
  - CSV summary
  - LaTeX table
  - bar plot of delta means with bootstrap 95% CI

By default, mean toxicity μ excludes seed turns and injection-slot messages
(toxic and neutral injectors), so μ reflects downstream propagation only.
Pass --include_injections to average over all non-seed messages including
injections.

Optional --summary_json paths load companion thread JSONL paths from each
file's output_files field (with robust resolution when paths are stale).
Use --jsonl for explicit threads paths. Use --summary_json_only to skip
directory scans and only use --summary_json plus --jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


CONDITION_ORDER = [
    "chain_inj1_full_visible",
    "tree_inj1_full_visible",
    "tree_inj1_parent_only",
    "tree_inj3_full_visible",
    "dag_inj1_full_visible",
    "dag_inj1_thread_local",
    "dag_inj3_full_visible",
    "dag_inj3_thread_local",
    "high_branch_inj1_full_visible",
    "high_branch_inj3_full_visible",
    "high_branch_inj2_parent_only",
]

CONDITION_DISPLAY = {
    "chain_inj1_full_visible": "Chain-single",
    "tree_inj1_full_visible": "Tree-single (full_visible)",
    "tree_inj1_parent_only": "Tree-single (parent_only)",
    "tree_inj3_full_visible": "Tree-multi (full_visible)",
    "dag_inj1_full_visible": "DAG-single (full_visible)",
    "dag_inj1_thread_local": "DAG-single (thread_local)",
    "dag_inj3_full_visible": "DAG-multi (full_visible)",
    "dag_inj3_thread_local": "DAG-multi (thread_local)",
    "high_branch_inj1_full_visible": "High-branch-single (full_visible)",
    "high_branch_inj3_full_visible": "High-branch-multi (full_visible)",
    "high_branch_inj2_parent_only": "High-branch (parent_only)",
}


@dataclass
class ParsedRecord:
    condition_label: str
    seed_id: str
    rollout_idx: int
    mode: str
    mu: float
    inj_tox_min: float
    source_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical_base",
        default="data/graph/canonical",
        help="Canonical graph rollout root.",
    )
    parser.add_argument(
        "--fallback_dirs",
        nargs="*",
        default=["data/graph/rollouts"],
        help="Fallback rollout directories to scan.",
    )
    parser.add_argument(
        "--chain_globs",
        nargs="*",
        default=[
            "data/reddit/influence_baseline_threads_reddit_rollouts_rollout_*.jsonl",
            "data/memory_gpt/rollouts/influence_memory_*_threads_rollout_*.jsonl",
            "data/memory_llama/rollouts/influence_memory_*_threads_rollout_*.jsonl",
        ],
        help="Glob patterns for possible chain rollouts.",
    )
    parser.add_argument(
        "--out_csv",
        default="results/tables/table1_propagation.csv",
    )
    parser.add_argument(
        "--out_tex",
        default="results/tables/table1_propagation.tex",
    )
    parser.add_argument(
        "--out_plot",
        default="results/figures/table1_delta_mu_barplot.png",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--include_injections",
        action="store_true",
        help=(
            "Include injection-slot messages in μ (default excludes them so the "
            "mean is over downstream agents only, excluding toxic/neutral injectors)."
        ),
    )
    parser.add_argument(
        "--downstream_only",
        action="store_true",
        help="Deprecated: excluding injections is now the default; use --include_injections to opt out.",
    )
    parser.add_argument(
        "--min_toxic_injection_toxicity",
        type=float,
        default=None,
        help=(
            "If provided, keep only rollout pairs whose toxic-mode injection message(s) "
            "have min toxicity >= this threshold."
        ),
    )
    parser.add_argument(
        "--summary_json",
        nargs="*",
        default=[],
        help=(
            "Run metadata JSON files (e.g. *influence*_summary.json) whose "
            "`output_files` list names the companion threads JSONL to score."
        ),
    )
    parser.add_argument(
        "--jsonl",
        nargs="*",
        default=[],
        help=(
            "Explicit paths to threads *.jsonl rollouts (e.g. tree single-inj "
            "influence_full_threads.jsonl). Merged with scanned paths unless "
            "--summary_json_only is set."
        ),
    )
    parser.add_argument(
        "--summary_json_only",
        action="store_true",
        help=(
            "If set, do not scan --canonical_base, --fallback_dirs, or --chain_globs; "
            "only load JSONL from --summary_json (output_files) and --jsonl."
        ),
    )
    parser.add_argument(
        "--auto_summary_root",
        default=None,
        help=(
            "Automatically include all *summary.json under this root and resolve "
            "their output_files to threads JSONL."
        ),
    )
    parser.add_argument(
        "--allow_memory_modes",
        nargs="*",
        default=["none"],
        help=(
            "memory_mode values to keep (default: 'none' for transcript-only "
            "Table 1). Pass 'memory' to include memory-mode rollouts; when "
            "'memory' is allowed, the memory_sanitize value is appended to the "
            "condition label so variants (gate/rewrite/none) appear as "
            "separate rows."
        ),
    )
    return parser.parse_args()


def normalize_context_mode(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower()
    if v in {"full", "full_visible"}:
        return "full_visible"
    if v in {"parent", "parent_only"}:
        return "parent_only"
    if v in {"thread_local", "local"}:
        return "thread_local"
    if v in {"path_to_root", "root_path"}:
        return "path_to_root"
    return v


def infer_topology_from_path(path: str) -> str | None:
    lowered = path.lower()
    for topo in ("high_branch", "tree", "dag"):
        if f"/{topo}/" in lowered or lowered.startswith(f"{topo}/"):
            return topo
    if "/chain/" in lowered or lowered.startswith("chain/"):
        return "chain"
    # Baseline reddit rollouts are chain-style threads.
    if "influence_baseline_threads" in lowered or "/reddit/" in lowered:
        return "chain"
    return None


def infer_context_from_path(path: str) -> str | None:
    lowered = path.lower()
    for ctx in ("full_visible", "parent_only", "thread_local", "path_to_root"):
        if f"ctx_{ctx}" in lowered or ctx in lowered:
            return ctx
    if "influence_baseline_threads" in lowered or "/reddit/" in lowered:
        return "full_visible"
    return None


def infer_inj_from_path(path: str) -> int | None:
    lowered = path.lower()
    m = re.search(r"inj(\d+)", lowered)
    if m:
        return int(m.group(1))
    if "single_injection" in lowered:
        return 1
    if "multi_injection" in lowered:
        return 3
    if "high_branch_injection" in lowered:
        return 2
    if "influence_baseline_threads" in lowered or "/reddit/" in lowered:
        return 1
    return None


def extract_condition(record: dict[str, Any], path: str) -> str | None:
    meta = record.get("intervention_meta") or {}
    graph_meta = record.get("graph_meta") or {}

    topology = (
        record.get("topology")
        or meta.get("topology")
        or graph_meta.get("topology")
        or infer_topology_from_path(path)
    )
    context_mode = normalize_context_mode(
        record.get("context_mode")
        or meta.get("context_mode")
        or graph_meta.get("context_mode")
        or infer_context_from_path(path)
    )
    n_inj = (
        record.get("n_toxic_injections")
        or meta.get("n_toxic_injections")
        or infer_inj_from_path(path)
        or 1
    )

    if topology is None or context_mode is None:
        return None
    return f"{topology}_inj{int(n_inj)}_{context_mode}"


def extract_memory_mode(record: dict[str, Any], path: str) -> str | None:
    meta = record.get("intervention_meta") or {}
    graph_meta = record.get("graph_meta") or {}
    memory_mode = record.get("memory_mode") or meta.get("memory_mode") or graph_meta.get("memory_mode")
    if memory_mode is None and "memsan_none" in path.lower():
        memory_mode = "none"
    return memory_mode


def extract_memory_sanitize(record: dict[str, Any]) -> str:
    meta = record.get("intervention_meta") or {}
    val = record.get("memory_sanitize") or meta.get("memory_sanitize") or "none"
    return str(val).strip().lower() or "none"


def keep_memory_mode(record: dict[str, Any], path: str, allowed: set[str]) -> bool:
    lowered = path.lower()
    # Preserve legacy guardrail: exclude memory-mode run directories when the
    # caller only asked for transcript-only (memory_mode=none) data.
    if allowed == {"none"} and (
        "/data/memory_gpt/" in lowered or "/data/memory_llama/" in lowered
    ):
        return False
    mm = extract_memory_mode(record, path) or "none"
    return mm in allowed


def is_memory_none(record: dict[str, Any], path: str) -> bool:
    return keep_memory_mode(record, path, {"none"})


def extract_mode(record: dict[str, Any]) -> str | None:
    mode = record.get("mode")
    if mode in {"toxic", "neutral"}:
        return mode
    return None


def extract_rollout_idx(record: dict[str, Any], path: str) -> int:
    for key in ("rollout_idx", "rollout_id"):
        if key in record:
            try:
                return int(record[key])
            except Exception:
                pass
    m = re.search(r"rollout_(\d+)", path)
    if m:
        return int(m.group(1))
    return -1


def is_seed_message(msg: dict[str, Any]) -> bool:
    role = str(msg.get("role", "")).strip().lower()
    agent = str(msg.get("agent", "")).strip().lower()
    author = str(msg.get("author_id", "")).strip().lower()
    turn = msg.get("turn")
    if role == "seed":
        return True
    if agent in {"a0", "seed"}:
        return True
    if author in {"seed", "human_seed"}:
        return True
    if turn == 0:
        return True
    return False


def is_injection_message(msg: dict[str, Any], record: dict[str, Any]) -> bool:
    # Preferred explicit marker in graph rollouts.
    if bool(msg.get("injection", False)):
        return True

    # Legacy / chain-style fallback markers.
    agent = str(msg.get("agent", "")).strip().lower()
    slot = str(msg.get("agent_slot", "")).strip().lower()
    author = str(msg.get("author_id", "")).strip().lower()
    turn = msg.get("turn")

    if agent.startswith("a1_") or slot.startswith("a1_"):
        return True
    if author in {"agent_toxic", "agent_neutral"}:
        return True

    # Final fallback: in chain baseline, intervention is at turn 1.
    topo = str(record.get("topology") or infer_topology_from_path(str(record.get("_source_path", ""))) or "").lower()
    if topo == "chain":
        try:
            return int(turn) == 1
        except Exception:
            return False

    return False


def _extract_injection_toxicities(record: dict[str, Any]) -> list[float]:
    messages = record.get("messages") or []
    inj_scores: list[float] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if not is_injection_message(msg, record):
            continue
        detox = msg.get("detoxify")
        if isinstance(detox, dict) and "toxicity" in detox:
            try:
                inj_scores.append(float(detox["toxicity"]))
            except Exception:
                continue
    return inj_scores


def compute_mu(record: dict[str, Any], downstream_only: bool = False) -> float:
    messages = record.get("messages") or []
    tox_scores: list[float] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if is_seed_message(msg):
            continue
        if downstream_only and is_injection_message(msg, record):
            continue
        detox = msg.get("detoxify")
        if isinstance(detox, dict) and "toxicity" in detox:
            try:
                tox_scores.append(float(detox["toxicity"]))
            except Exception:
                continue

    # Fallback for legacy records lacking per-message detoxify.
    if not tox_scores:
        scores = record.get("scores") or {}
        d_by_turn = scores.get("detoxify_by_turn") or {}
        if isinstance(d_by_turn, dict):
            for turn_str, detox in d_by_turn.items():
                try:
                    turn = int(turn_str)
                except Exception:
                    continue
                if turn <= 0:
                    continue
                if isinstance(detox, dict) and "toxicity" in detox:
                    try:
                        tox_scores.append(float(detox["toxicity"]))
                    except Exception:
                        continue

    if not tox_scores:
        return float("nan")
    return float(np.mean(np.array(tox_scores, dtype=float)))


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _resolve_summary_output_jsonl(summary_path: str, out_rel: str) -> Path | None:
    """Map a summary JSON `output_files` entry to an existing threads JSONL path."""
    summary = Path(summary_path).resolve()
    rel = Path(out_rel)
    candidates: list[Path] = []
    if rel.is_absolute():
        candidates.append(rel)
    else:
        candidates.append((Path.cwd() / rel).resolve())
        candidates.append((summary.parent / rel).resolve())
        candidates.append((summary.parent / rel.name).resolve())
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            return cand
    return None


def jsonl_paths_from_rollout_summary(summary_path: str) -> list[str]:
    with open(summary_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    outs = meta.get("output_files") or []
    if not isinstance(outs, list):
        return []
    resolved: list[str] = []
    for item in outs:
        if not isinstance(item, str) or not item.strip():
            continue
        p = _resolve_summary_output_jsonl(summary_path, item.strip())
        if p is None:
            print(
                f"WARNING: summary {summary_path!r} output_files entry {item!r} "
                "did not resolve to an existing JSONL (try same-dir basename)."
            )
            continue
        if p.suffix.lower() != ".jsonl":
            continue
        resolved.append(str(p.resolve()))
    return resolved


def collect_files(args: argparse.Namespace) -> list[str]:
    discovered: set[str] = set()
    summary_paths: set[str] = set()

    if not getattr(args, "summary_json_only", False):
        canonical = Path(args.canonical_base)
        if canonical.exists():
            for p in canonical.rglob("*.jsonl"):
                discovered.add(str(p.resolve()))

        for fallback in args.fallback_dirs:
            base = Path(fallback)
            if not base.exists():
                continue
            for p in base.rglob("*.jsonl"):
                discovered.add(str(p.resolve()))

        for pattern in args.chain_globs:
            for p in Path(".").glob(pattern):
                discovered.add(str(p.resolve()))

    for summary_path in getattr(args, "summary_json", []) or []:
        summary_paths.add(str(Path(summary_path).expanduser()))

    auto_root = getattr(args, "auto_summary_root", None)
    if auto_root:
        root = Path(auto_root).expanduser()
        if not root.exists():
            print(f"WARNING: --auto_summary_root does not exist: {str(root)!r}")
        else:
            for p in root.rglob("*summary.json"):
                summary_paths.add(str(p))

    for summary_path in sorted(summary_paths):
        sp = str(Path(summary_path).expanduser())
        if not Path(sp).is_file():
            print(f"WARNING: summary_json path missing or not a file: {sp!r}")
            continue
        for jp in jsonl_paths_from_rollout_summary(sp):
            discovered.add(jp)

    for raw in getattr(args, "jsonl", []) or []:
        p = Path(raw).expanduser().resolve()
        if not p.is_file():
            print(f"WARNING: --jsonl path missing or not a file: {raw!r}")
            continue
        if p.suffix.lower() != ".jsonl":
            print(f"WARNING: --jsonl expected .jsonl file, skipping: {raw!r}")
            continue
        discovered.add(str(p))

    return sorted(discovered)


def bootstrap_ci(deltas: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    n = len(deltas)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(deltas, size=n, replace=True)
        draws[i] = float(np.mean(sample))
    return tuple(np.percentile(draws, [2.5, 97.5]).tolist())


def p_value_wilcoxon_greater(deltas: np.ndarray) -> float:
    if len(deltas) < 10:
        return float("nan")
    if np.allclose(deltas, 0.0):
        return 1.0
    try:
        _stat, p = wilcoxon(deltas, alternative="greater")
        return float(p)
    except Exception:
        return float("nan")


def make_summary_table(
    paired: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        sub = paired[paired["condition_label"] == condition].copy()
        if sub.empty:
            print(f"WARNING: No paired data for condition={condition}; skipping.")
            continue
        deltas = sub["delta_mu"].to_numpy(dtype=float)
        n = len(deltas)
        if n < 20:
            print(f"WARNING: condition={condition} has only {n} paired seeds (<20).")

        ci_lo, ci_hi = bootstrap_ci(deltas, bootstrap_samples, bootstrap_seed)
        p_val = p_value_wilcoxon_greater(deltas)
        rows.append(
            {
                "condition_label": condition,
                "condition": CONDITION_DISPLAY.get(condition, condition),
                "n_seeds": n,
                "mu_toxic": float(sub["toxic"].mean()),
                "mu_neutral": float(sub["neutral"].mean()),
                "delta_mu_mean": float(np.mean(deltas)),
                "delta_mu_median": float(np.median(deltas)),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "p_value": p_val,
            }
        )
    return pd.DataFrame(rows)


def fmt_float(v: float, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "NA"
    return f"{v:.{digits}f}"


def fmt_p(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "NA"
    if v < 0.001:
        return "<0.001"
    return f"{v:.3f}"


def print_table(summary: pd.DataFrame) -> None:
    print("\nTable 1: Main propagation results (§6.1)")
    print("=" * 110)
    header = (
        f"{'Condition':<30} {'N':>5} {'mu_toxic':>10} {'mu_neutral':>11} "
        f"{'Delta(mean)':>12} {'Delta(med)':>11} {'95% CI':>19} {'p-value':>9}"
    )
    print(header)
    print("-" * 110)
    for _, r in summary.iterrows():
        ci = f"[{fmt_float(float(r['ci_lo']), 4)}, {fmt_float(float(r['ci_hi']), 4)}]"
        line = (
            f"{r['condition']:<30} {int(r['n_seeds']):>5d} "
            f"{fmt_float(float(r['mu_toxic']), 4):>10} "
            f"{fmt_float(float(r['mu_neutral']), 4):>11} "
            f"{fmt_float(float(r['delta_mu_mean']), 4):>12} "
            f"{fmt_float(float(r['delta_mu_median']), 4):>11} "
            f"{ci:>19} {fmt_p(float(r['p_value'])):>9}"
        )
        print(line)


def save_latex(summary: pd.DataFrame, out_path: str) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main propagation results across rollout conditions.}",
        r"\label{tab:propagation}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Condition & $N$ & $\bar{\mu}_{\text{toxic}}$ & $\bar{\mu}_{\text{neutral}}$ & $\Delta\mu$ & 95\% CI & $p$ \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        ci = f"[{fmt_float(float(r['ci_lo']), 4)}, {fmt_float(float(r['ci_hi']), 4)}]"
        row = (
            f"{r['condition']} & {int(r['n_seeds'])} & "
            f"{fmt_float(float(r['mu_toxic']), 4)} & {fmt_float(float(r['mu_neutral']), 4)} & "
            f"{fmt_float(float(r['delta_mu_mean']), 4)} & {ci} & {fmt_p(float(r['p_value']))} \\\\"
        )
        lines.append(row)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_plot(summary: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df = summary.copy()
    df["yerr_lo"] = df["delta_mu_mean"] - df["ci_lo"]
    df["yerr_hi"] = df["ci_hi"] - df["delta_mu_mean"]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(df))
    y = df["delta_mu_mean"].to_numpy(dtype=float)
    yerr = np.vstack([df["yerr_lo"].to_numpy(dtype=float), df["yerr_hi"].to_numpy(dtype=float)])

    bars = ax.bar(x, y, color=sns.color_palette("deep", len(df)), alpha=0.9, width=0.72)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=4, linewidth=1.2)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"], rotation=30, ha="right")
    ax.set_ylabel("Mean paired Δμ (toxic - neutral)")
    ax.set_xlabel("Condition")
    ax.set_title("Table 1 paired effect sizes with 95% bootstrap CI")

    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.03
    for i, (_bar, pval, top) in enumerate(zip(bars, df["p_value"], df["ci_hi"])):
        ax.text(i, top + pad, f"p={fmt_p(float(pval))}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if getattr(args, "downstream_only", False):
        print(
            "NOTE: --downstream_only is deprecated (injections are excluded by default). "
            "Use --include_injections only if you want injector text in μ."
        )
    downstream_only = not bool(getattr(args, "include_injections", False))
    allowed_memory_modes = {
        str(v).strip().lower() for v in (args.allow_memory_modes or ["none"]) if str(v).strip()
    } or {"none"}
    files = collect_files(args)
    print(f"Discovered {len(files)} candidate JSONL files.")
    print(
        "μ aggregation: "
        + (
            "downstream only (seed + injection messages excluded)"
            if downstream_only
            else "non-seed transcript (injection messages included)"
        )
    )

    parsed: list[ParsedRecord] = []
    skipped_memory_mode = 0
    skipped_mode = 0
    skipped_condition = 0

    for file_path in files:
        for record in iter_jsonl(file_path):
            record["_source_path"] = file_path
            if not keep_memory_mode(record, file_path, allowed_memory_modes):
                skipped_memory_mode += 1
                continue
            mode = extract_mode(record)
            if mode is None:
                skipped_mode += 1
                continue
            condition = extract_condition(record, file_path)
            if condition is None:
                skipped_condition += 1
                continue

            # For memory-mode rollouts, split variants by memory_sanitize so
            # (gate / rewrite / none) do not collapse into the same condition.
            mem_mode = (extract_memory_mode(record, file_path) or "none").lower()
            if mem_mode != "none":
                sanitize = extract_memory_sanitize(record)
                condition = f"{condition}__mem_{sanitize}"
                if condition not in CONDITION_ORDER:
                    CONDITION_ORDER.append(condition)
                if condition not in CONDITION_DISPLAY:
                    base = condition.split("__mem_")[0]
                    base_display = CONDITION_DISPLAY.get(base, base)
                    CONDITION_DISPLAY[condition] = f"{base_display} [memory:{sanitize}]"

            if condition not in CONDITION_ORDER:
                continue

            seed_id = str(record.get("seed_id") or "unknown_seed")
            rollout_idx = extract_rollout_idx(record, file_path)
            mu = compute_mu(record, downstream_only=downstream_only)
            if math.isnan(mu):
                continue
            inj_scores = _extract_injection_toxicities(record)
            inj_tox_min = float(np.min(inj_scores)) if inj_scores else float("nan")

            parsed.append(
                ParsedRecord(
                    condition_label=condition,
                    seed_id=seed_id,
                    rollout_idx=rollout_idx,
                    mode=mode,
                    mu=mu,
                    inj_tox_min=inj_tox_min,
                    source_path=file_path,
                )
            )

    print(
        "Parsed records:",
        f"kept={len(parsed)}",
        f"skipped_memory_mode={skipped_memory_mode}",
        f"skipped_mode={skipped_mode}",
        f"skipped_condition={skipped_condition}",
    )

    if not parsed:
        raise RuntimeError("No usable transcript-only records found for Table 1.")

    df = pd.DataFrame([r.__dict__ for r in parsed])
    # Deduplicate overlap between canonical and fallback by semantic key.
    df = df.drop_duplicates(
        subset=["condition_label", "seed_id", "rollout_idx", "mode"],
        keep="first",
    )

    if args.min_toxic_injection_toxicity is not None:
        thr = float(args.min_toxic_injection_toxicity)
        toxic = df[df["mode"] == "toxic"].copy()
        pass_keys = toxic[
            toxic["inj_tox_min"].notna() & (toxic["inj_tox_min"] >= thr)
        ][["condition_label", "seed_id", "rollout_idx"]].drop_duplicates()
        pass_keys["_keep"] = 1
        before = len(df)
        df = df.merge(
            pass_keys,
            on=["condition_label", "seed_id", "rollout_idx"],
            how="inner",
        ).drop(columns=["_keep"])
        after = len(df)
        print(
            f"Applied toxic-injection filter (min inj toxicity >= {thr:.3f}): "
            f"rows {before} -> {after}"
        )

    seed_avg = (
        df.groupby(["condition_label", "seed_id", "mode"], as_index=False)["mu"]
        .mean()
        .rename(columns={"mu": "mu_mode"})
    )
    pivot = seed_avg.pivot_table(
        index=["condition_label", "seed_id"],
        columns="mode",
        values="mu_mode",
        aggfunc="mean",
    ).reset_index()

    if "toxic" not in pivot.columns or "neutral" not in pivot.columns:
        raise RuntimeError("Could not build paired toxic/neutral data (missing one mode).")

    paired = pivot.dropna(subset=["toxic", "neutral"]).copy()
    paired["delta_mu"] = paired["toxic"] - paired["neutral"]

    summary = make_summary_table(
        paired=paired,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    if summary.empty:
        raise RuntimeError("All requested conditions were empty after pairing.")

    summary = summary.set_index("condition_label").loc[
        [c for c in CONDITION_ORDER if c in summary["condition_label"].values]
    ].reset_index()

    print_table(summary)

    out_csv = args.out_csv
    out_tex = args.out_tex
    out_plot = args.out_plot
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    save_latex(summary, out_tex)
    save_plot(summary, out_plot)

    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved LaTeX: {out_tex}")
    print(f"Saved plot: {out_plot}")

    chain_rows = summary[summary["condition_label"] == "chain_inj1_full_visible"]
    if chain_rows.empty:
        print(
            "NOTE: Chain-single condition unavailable for memory_mode=none; "
            "possibly only memory-mode chain files are present."
        )


if __name__ == "__main__":
    main()

"""
cd /u/anon3/unlearn_agent && python scripts/extract_table1_propagation.py \
  --summary_json_only \
  --auto_summary_root /u/anon3/unlearn_agent/data/chain/memory_gpt \
  --fallback_dirs --chain_globs \
  --out_csv results/sec6_1/tables/table1_propagation_memory_gpt_auto.csv \
  --out_tex results/sec6_1/tables/table1_propagation_memory_gpt_auto.tex \
  --out_plot results/sec6_1/figures/table1_propagation_memory_gpt_auto.png

cd /u/anon3/unlearn_agent && python scripts/extract_table1_propagation.py \
  --summary_json_only \
  --jsonl /u/anon3/unlearn_agent/data/graph/memory_gpt/high_branch/influence_full_threads.jsonl \
  --fallback_dirs --chain_globs \
  --allow_memory_modes memory \
  --out_csv  results/sec6_1/tables/table1_memory_gpt_high_branch.csv \
  --out_tex  results/sec6_1/tables/table1_memory_gpt_high_branch.tex \
  --out_plot results/sec6_1/figures/table1_memory_gpt_high_branch.png
"""