#!/usr/bin/env python3
"""Section 6.5 analysis pipeline for Table 5 and Figure 4."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

import scripts.compute_sec63_stats as sec63

BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PLACEHOLDER_HINTS = (
    "<reply text if action is reply, else empty>",
    "filtered due to content policy",
    "response filtered",
)

CONDITIONS: list[dict[str, Any]] = [
    {
        "key": "01_no_intervention",
        "label": "No intervention",
        "controls": {"transcript": False, "memory": False, "dpo": False},
        "group": "Baselines",
        "cell": 1,
    },
    {
        "key": "02_output_filter",
        "label": "Output filter",
        "controls": {"transcript": False, "memory": False, "dpo": False},
        "group": "Baselines",
        "cell": 2,
    },
    {
        "key": "06_dpo_only",
        "label": "DPO only",
        "controls": {"transcript": False, "memory": False, "dpo": True},
        "group": "Baselines",
        "cell": 6,
    },
    {
        "key": "03_transcript_only",
        "label": "Transcript only",
        "controls": {"transcript": True, "memory": False, "dpo": False},
        "group": "Single-channel interventions",
        "cell": 3,
    },
    {
        "key": "04_memory_only",
        "label": "Memory only",
        "controls": {"transcript": False, "memory": True, "dpo": False},
        "group": "Single-channel interventions",
        "cell": 4,
    },
    {
        "key": "05_transcript_memory",
        "label": "Transcript + Memory",
        "controls": {"transcript": True, "memory": True, "dpo": False},
        "group": "Combinations",
        "cell": 5,
    },
    {
        "key": "07_transcript_dpo",
        "label": "Transcript + DPO",
        "controls": {"transcript": True, "memory": False, "dpo": True},
        "group": "Combinations",
        "cell": 7,
    },
    {
        "key": "08_memory_dpo",
        "label": "Memory + DPO",
        "controls": {"transcript": False, "memory": True, "dpo": True},
        "group": "Combinations",
        "cell": 8,
    },
    {
        "key": "09_full_system",
        "label": "Full system",
        "controls": {"transcript": True, "memory": True, "dpo": True},
        "group": "Combinations",
        "cell": 9,
    },
]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _bootstrap_ci(values: list[float], n_boot: int = 10_000, seed: int = 42) -> list[float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot = arr[idx].mean(axis=1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def _wilcoxon_greater(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=float)
    if np.allclose(arr, 0.0):
        return None
    try:
        return float(wilcoxon(arr, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        return None


def _msg_toxicity(message: dict[str, Any], record: dict[str, Any]) -> float | None:
    detox = message.get("detoxify")
    if isinstance(detox, dict):
        val = _safe_float(detox.get("toxicity"))
        if val is not None:
            return val
    val = _safe_float(message.get("toxicity"))
    if val is not None:
        return val

    turn_key = str(message.get("turn"))
    scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
    tox_by_turn = scores.get("toxicity_by_turn") if isinstance(scores.get("toxicity_by_turn"), dict) else {}
    detox_by_turn = scores.get("detoxify_by_turn") if isinstance(scores.get("detoxify_by_turn"), dict) else {}
    val = _safe_float(tox_by_turn.get(turn_key))
    if val is not None:
        return val
    if isinstance(detox_by_turn.get(turn_key), dict):
        return _safe_float((detox_by_turn.get(turn_key) or {}).get("toxicity"))
    return None


def _memory_toxicity(entry: dict[str, Any], record: dict[str, Any]) -> float | None:
    detox = entry.get("memory_detoxify") or entry.get("detoxify")
    if isinstance(detox, dict):
        val = _safe_float(detox.get("toxicity"))
        if val is not None:
            return val
    for key in ("memory_tox", "mem_tox", "toxicity"):
        val = _safe_float(entry.get(key))
        if val is not None:
            return val

    turn = entry.get("turn")
    if turn is not None:
        turn_int = int(turn)
        for item in record.get("memory_interventions") or []:
            if not isinstance(item, dict):
                continue
            if int(item.get("turn", -9999)) != turn_int:
                continue
            val = _safe_float(item.get("memory_toxicity"))
            if val is not None:
                return val
    return None


def _downstream_toxicity_by_turn(record: dict[str, Any]) -> dict[int, float]:
    downstream: dict[int, float] = {}
    for msg in record.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        turn = msg.get("turn")
        if turn is None:
            continue
        try:
            turn_i = int(turn)
        except (TypeError, ValueError):
            continue
        agent = str(msg.get("agent") or "").strip()
        if agent in {"A0", "A1", "A1_toxic", "A1_neutral"}:
            continue
        tox = _msg_toxicity(msg, record)
        if tox is None:
            continue
        downstream[turn_i] = tox
    return downstream


def _downstream_top3_by_index(record: dict[str, Any]) -> dict[int, float]:
    by_turn = _downstream_toxicity_by_turn(record)
    out: dict[int, float] = {}
    for idx, turn in enumerate(sorted(by_turn.keys())[:3], start=1):
        out[idx] = by_turn[turn]
    return out


def _memory_by_turn(record: dict[str, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for entry in record.get("memory_history") or []:
        if not isinstance(entry, dict):
            continue
        turn = entry.get("turn")
        try:
            turn_i = int(turn)
        except (TypeError, ValueError):
            continue
        val = _memory_toxicity(entry, record)
        if val is not None:
            out[turn_i] = val
    return out


def _first_downstream_after(record: dict[str, Any], turn: int) -> float | None:
    by_turn = _downstream_toxicity_by_turn(record)
    next_turns = [t for t in sorted(by_turn.keys()) if t > turn]
    if not next_turns:
        return None
    return by_turn[next_turns[0]]


def _records_by_rollout(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        rid = rec.get("rollout_id")
        try:
            rid_i = int(rid)
        except (TypeError, ValueError):
            continue
        out.setdefault(rid_i, []).append(rec)
    return out


def _extract_model_paths(summary: Any, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(summary, dict):
        for key, value in summary.items():
            sub_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, str) and "model" in str(key).lower():
                out.append(value)
            out.extend(_extract_model_paths(value, prefix=sub_prefix))
    elif isinstance(summary, list):
        for item in summary:
            out.extend(_extract_model_paths(item, prefix=prefix))
    return out


def _check_dpo_summary(condition_key: str, cell_n: int, summary_path: Path) -> None:
    if not summary_path.exists():
        print(f"WARNING: {condition_key} missing summary.json for DPO verification")
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"WARNING: {condition_key} has invalid summary.json (cannot verify DPO model path)")
        return
    model_paths = [p for p in _extract_model_paths(summary) if p]
    if not model_paths:
        print(f"WARNING: {condition_key} summary.json has no model path field for DPO verification")
        return
    if any(p.strip() == BASE_MODEL_ID for p in model_paths):
        print(f"WARNING: cell {cell_n} appears to use base model, not DPO adapter")


def _spotcheck_placeholder_rate(store: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    total = 0
    hits = 0
    for modes in store.values():
        for rec in modes.get("toxic", []):
            for msg in rec.get("messages") or []:
                if not isinstance(msg, dict):
                    continue
                agent = str(msg.get("agent") or "")
                if not agent.startswith("A1"):
                    continue
                total += 1
                text = str(msg.get("text") or "").lower()
                if any(hint in text for hint in PLACEHOLDER_HINTS):
                    hits += 1
    rate = (hits / total) if total else 0.0
    print(f"[02_output_filter] A1 placeholder rate: {rate:.3f} ({hits}/{total})")


def _render_tex(stats: dict[str, dict[str, Any]], out_path: Path) -> None:
    turn4_vals = [r["turn4_tox_toxic"] for r in stats.values() if r.get("turn4_tox_toxic") is not None]
    delta_abs_vals = [abs(r["delta_mu"]) for r in stats.values() if r.get("delta_mu") is not None]
    spg_vals = [r["spg_tau05"] for r in stats.values() if r.get("spg_tau05") is not None]

    best_turn4 = min(turn4_vals) if turn4_vals else None
    best_delta_abs = min(delta_abs_vals) if delta_abs_vals else None
    best_spg = min(spg_vals) if spg_vals else None

    def fmt_num(value: float | None) -> str:
        if value is None:
            return "--"
        return f"{value:.3f}"

    def fmt_mean_ci(value: float | None, ci: list[float] | None) -> str:
        if value is None or ci is None or len(ci) != 2:
            return "--"
        return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"

    def maybe_bold(text: str, flag: bool) -> str:
        return f"\\textbf{{{text}}}" if flag and text != "--" else text

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Unified comparison of baselines, single-channel interventions, and full-system combinations on Llama-3.1-8B-Instruct chain memory rollouts. Transcript and Memory controls use rewrite sanitization at $\\tau=0.5$. $\\Delta\\mu$ and SPG report mean [95\\% bootstrap CI] with one-sided paired Wilcoxon $p$-values for the alternative $> 0$.}",
        "\\label{tab:defense_ablation}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Condition & Transcript & Memory & DPO & Turn-4 tox & $\\Delta\\mu$ & SPG($\\tau{=}0.5$) \\\\",
        "\\midrule",
    ]

    current_group = None
    for cond in CONDITIONS:
        key = cond["key"]
        row = stats[key]
        group = cond["group"]
        if group != current_group:
            if current_group is not None:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{group}}}}} \\\\")
            current_group = group

        if row.get("status") == "missing":
            lines.append(f"% MISSING: {key}")

        transcript = "\\checkmark" if cond["controls"]["transcript"] else "--"
        memory = "\\checkmark" if cond["controls"]["memory"] else "--"
        dpo = "\\checkmark" if cond["controls"]["dpo"] else "--"

        turn4_text = fmt_num(row.get("turn4_tox_toxic"))
        delta_text = fmt_mean_ci(row.get("delta_mu"), row.get("delta_mu_ci"))
        spg_text = fmt_mean_ci(row.get("spg_tau05"), row.get("spg_tau05_ci"))

        turn4_is_best = (
            best_turn4 is not None
            and row.get("turn4_tox_toxic") is not None
            and abs(row["turn4_tox_toxic"] - best_turn4) < 1e-12
        )
        delta_is_best = (
            best_delta_abs is not None
            and row.get("delta_mu") is not None
            and abs(abs(row["delta_mu"]) - best_delta_abs) < 1e-12
        )
        spg_is_best = (
            best_spg is not None
            and row.get("spg_tau05") is not None
            and abs(row["spg_tau05"] - best_spg) < 1e-12
        )

        turn4_text = maybe_bold(turn4_text, turn4_is_best)
        delta_text = maybe_bold(delta_text, delta_is_best)
        spg_text = maybe_bold(spg_text, spg_is_best)

        lines.append(
            f"{cond['label']} & {transcript} & {memory} & {dpo} & {turn4_text} & {delta_text} & {spg_text} \\\\"
        )

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_figure(stats: dict[str, dict[str, Any]], out_path: Path) -> None:
    labels = [c["label"] for c in CONDITIONS]
    values = [stats[c["key"]].get("delta_mu") for c in CONDITIONS]
    ci_low = [stats[c["key"]].get("delta_mu_ci")[0] if stats[c["key"]].get("delta_mu_ci") else None for c in CONDITIONS]
    ci_high = [stats[c["key"]].get("delta_mu_ci")[1] if stats[c["key"]].get("delta_mu_ci") else None for c in CONDITIONS]

    group_color = {
        "Baselines": "#9aa4b2",
        "Single-channel interventions": "#7fb069",
        "Combinations": "#e0a458",
    }
    colors = [group_color[c["group"]] for c in CONDITIONS]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    draw_vals = [0.0 if v is None else float(v) for v in values]
    alpha = [0.3 if v is None else 0.9 for v in values]
    for yi, val, col, a in zip(y, draw_vals, colors, alpha):
        ax.barh(yi, val, color=col, alpha=a, height=0.7)

    for yi, mu, lo, hi in zip(y, values, ci_low, ci_high):
        if mu is None or lo is None or hi is None:
            continue
        xerr = np.array([[mu - lo], [hi - mu]])
        ax.errorbar(mu, yi, xerr=xerr, fmt="none", color="black", capsize=3, lw=1.1)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"$\Delta\mu$ (toxic - neutral)")
    ax.set_title("Defense ablation on Llama chain memory")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _analyze_condition(cond: dict[str, Any], data_root: Path) -> dict[str, Any]:
    key = cond["key"]
    cond_dir = data_root / key
    stats_row: dict[str, Any] = {
        "label": cond["label"],
        "controls": cond["controls"],
        "n_paired": None,
        "delta_mu": None,
        "delta_mu_ci": None,
        "delta_mu_pvalue": None,
        "spg_tau05": None,
        "spg_tau05_ci": None,
        "spg_tau05_pvalue": None,
        "mean_mem_tox": None,
        "turn4_tox_toxic": None,
        "turn4_tox_toxic_ci": None,
        "status": "missing",
    }

    expected_paths = [
        str(cond_dir / "rollouts" / "influence_threads_rollout_000.jsonl"),
        str(cond_dir / "rollouts" / "influence_threads_rollout_001.jsonl"),
    ]
    present, missing = sec63.resolve_existing_paths(expected_paths)
    if missing:
        print(f"[{key}] WARNING missing expected rollout files: {missing}")
    if not present:
        return stats_row

    store_raw = sec63.load_records(present)
    toxic_seeds = {sid for sid, modes in store_raw.items() if modes.get("toxic")}
    neutral_seeds = {sid for sid, modes in store_raw.items() if modes.get("neutral")}
    paired = sorted(toxic_seeds & neutral_seeds)
    dropped = len(toxic_seeds ^ neutral_seeds)
    if dropped:
        print(f"[{key}] dropped unmatched seeds: {dropped}")
    store = {sid: {"toxic": store_raw[sid]["toxic"], "neutral": store_raw[sid]["neutral"]} for sid in paired}

    if key == "02_output_filter":
        _spotcheck_placeholder_rate(store)

    if key in {"06_dpo_only", "07_transcript_dpo", "08_memory_dpo", "09_full_system"}:
        _check_dpo_summary(key, int(cond["cell"]), cond_dir / "summary.json")

    delta_per_seed: list[float] = []
    spg_per_seed: list[float] = []
    mem_tox_all: list[float] = []
    turn4_per_seed: list[float] = []

    for sid in paired:
        seed_data = store[sid]
        tox_by_rollout = _records_by_rollout(seed_data["toxic"])
        neu_by_rollout = _records_by_rollout(seed_data["neutral"])
        common_rollouts = sorted(set(tox_by_rollout) & set(neu_by_rollout))

        seed_deltas: list[float] = []
        seed_spg_deltas: list[float] = []
        seed_turn4: list[float] = []

        for rid in common_rollouts:
            t_recs = tox_by_rollout[rid]
            n_recs = neu_by_rollout[rid]

            for rec in t_recs:
                for mt in _memory_by_turn(rec).values():
                    mem_tox_all.append(mt)

            t_top3_all = [_downstream_top3_by_index(rec) for rec in t_recs]
            n_top3_all = [_downstream_top3_by_index(rec) for rec in n_recs]
            for idx in (1, 2, 3):
                t_vals = [m[idx] for m in t_top3_all if idx in m]
                n_vals = [m[idx] for m in n_top3_all if idx in m]
                if t_vals and n_vals:
                    seed_deltas.append(float(np.mean(t_vals) - np.mean(n_vals)))
                if idx == 3 and t_vals:
                    seed_turn4.append(float(np.mean(t_vals)))

            for t_rec in t_recs:
                t_mem = _memory_by_turn(t_rec)
                for turn, mt in t_mem.items():
                    if mt > 0.5:
                        continue
                    t_next = _first_downstream_after(t_rec, turn)
                    if t_next is None:
                        continue
                    n_next_vals = []
                    for n_rec in n_recs:
                        n_next = _first_downstream_after(n_rec, turn)
                        if n_next is not None:
                            n_next_vals.append(n_next)
                    if n_next_vals:
                        seed_spg_deltas.append(float(t_next - np.mean(n_next_vals)))

        if seed_deltas:
            delta_per_seed.append(float(np.mean(seed_deltas)))
        if seed_spg_deltas:
            spg_per_seed.append(float(np.mean(seed_spg_deltas)))
        if seed_turn4:
            turn4_per_seed.append(float(np.mean(seed_turn4)))

    stats_row["n_paired"] = len(paired)
    stats_row["delta_mu"] = _mean(delta_per_seed)
    stats_row["delta_mu_ci"] = _bootstrap_ci(delta_per_seed, n_boot=10_000, seed=42)
    stats_row["delta_mu_pvalue"] = _wilcoxon_greater(delta_per_seed)
    stats_row["spg_tau05"] = _mean(spg_per_seed)
    stats_row["spg_tau05_ci"] = _bootstrap_ci(spg_per_seed, n_boot=10_000, seed=42)
    stats_row["spg_tau05_pvalue"] = _wilcoxon_greater(spg_per_seed)
    stats_row["mean_mem_tox"] = _mean(mem_tox_all)
    stats_row["turn4_tox_toxic"] = _mean(turn4_per_seed)
    stats_row["turn4_tox_toxic_ci"] = _bootstrap_ci(turn4_per_seed, n_boot=10_000, seed=42)
    stats_row["status"] = "ok"

    return stats_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("data/llama/sec65"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/sec6_5"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, dict[str, Any]] = {}
    missing_conditions: list[str] = []
    for cond in CONDITIONS:
        key = cond["key"]
        row = _analyze_condition(cond, args.data_root)
        all_stats[key] = row
        if row.get("status") != "ok":
            missing_conditions.append(key)
        dmu = row.get("delta_mu")
        ci = row.get("delta_mu_ci") or [None, None]
        p = row.get("delta_mu_pvalue")
        spg = row.get("spg_tau05")
        t4 = row.get("turn4_tox_toxic")
        n = row.get("n_paired")
        dmu_s = f"{dmu:.3f}" if dmu is not None else "--"
        lo_s = f"{ci[0]:.3f}" if ci[0] is not None else "--"
        hi_s = f"{ci[1]:.3f}" if ci[1] is not None else "--"
        p_s = f"{p:.3g}" if p is not None else "--"
        spg_s = f"{spg:.3f}" if spg is not None else "--"
        t4_s = f"{t4:.3f}" if t4 is not None else "--"
        n_s = str(n) if n is not None else "--"
        print(f"[{key}] n={n_s}  Δμ={dmu_s} [{lo_s}, {hi_s}]  p={p_s}  SPG={spg_s}  T4={t4_s}  status={row['status']}")

    stats_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conditions": all_stats,
    }

    table5_stats_path = args.out_dir / "table5_stats.json"
    table5_tex_path = args.out_dir / "table5.tex"
    fig4_path = args.out_dir / "figure4_ablation.pdf"

    table5_stats_path.write_text(json.dumps(stats_json, indent=2), encoding="utf-8")
    _render_tex(all_stats, table5_tex_path)
    _render_figure(all_stats, fig4_path)

    print(
        f"Completed sec6.5 analysis. missing={missing_conditions if missing_conditions else 'none'} "
        f"outputs={args.out_dir}"
    )


if __name__ == "__main__":
    main()
