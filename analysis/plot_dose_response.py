import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ALPHAS: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DOWNSTREAM_TURNS = [2, 3, 4]
DETOXIFY_KEYS = [
    "identity_attack",
    "insult",
    "obscene",
    "severe_toxicity",
    "threat",
    "toxicity",
]


def alpha_to_tag(alpha: float) -> str:
    return f"{alpha:.1f}"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def per_seed_turn_means(records: List[Dict[str, Any]], key: str) -> np.ndarray:
    vals: List[float] = []
    for rec in records:
        by_turn = ((rec.get("scores") or {}).get("detoxify_by_turn") or {})
        turn_vals: List[float] = []
        for t in DOWNSTREAM_TURNS:
            turn_data = by_turn.get(str(t), {})
            if not isinstance(turn_data, dict):
                continue
            try:
                turn_vals.append(float(turn_data.get(key, 0.0)))
            except (TypeError, ValueError):
                continue
        if turn_vals:
            vals.append(float(np.mean(turn_vals)))
    return np.asarray(vals, dtype=float)


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 95.0,
    rng_seed: int = 123,
) -> Tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(rng_seed)
    n = values.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[i] = float(np.mean(sample))
    mean = float(np.mean(values))
    lo = float(np.percentile(boots, (100.0 - ci) / 2.0))
    hi = float(np.percentile(boots, 100.0 - (100.0 - ci) / 2.0))
    return mean, lo, hi


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/dose_response")
    parser.add_argument("--out_pdf", default="figures/dose_response.pdf")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--rng_seed", type=int, default=123)
    args = parser.parse_args()

    key_stats: Dict[str, Dict[float, Tuple[float, float, float]]] = {
        k: {} for k in DETOXIFY_KEYS
    }

    for alpha in ALPHAS:
        alpha_tag = alpha_to_tag(alpha)
        path = os.path.join(args.data_dir, f"threads_alpha_{alpha_tag}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dose-response file: {path}")
        records = load_jsonl(path)
        for key in DETOXIFY_KEYS:
            vals = per_seed_turn_means(records, key)
            key_stats[key][alpha] = bootstrap_mean_ci(
                vals, n_boot=args.n_boot, rng_seed=args.rng_seed
            )

    x = np.asarray(ALPHAS, dtype=float)

    # Page 1: main toxicity curve with CI
    fig_main, ax_main = plt.subplots(figsize=(6.5, 4.0))
    y = np.asarray([key_stats["toxicity"][a][0] for a in ALPHAS], dtype=float)
    lo = np.asarray([key_stats["toxicity"][a][1] for a in ALPHAS], dtype=float)
    hi = np.asarray([key_stats["toxicity"][a][2] for a in ALPHAS], dtype=float)
    ax_main.plot(x, y, marker="o", color="#1f77b4", linewidth=2)
    ax_main.fill_between(x, lo, hi, color="#1f77b4", alpha=0.2)
    ax_main.set_xlabel("toxicity alpha")
    ax_main.set_ylabel("mean detoxify toxicity (turns 2-4)")
    ax_main.set_title("Dose-response curve (95% bootstrap CI)")
    ax_main.grid(alpha=0.3)
    fig_main.tight_layout()

    # Page 2: 6-panel category curves with CI
    fig_grid, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True)
    for idx, key in enumerate(DETOXIFY_KEYS):
        ax = axes[idx // 3, idx % 3]
        yk = np.asarray([key_stats[key][a][0] for a in ALPHAS], dtype=float)
        lok = np.asarray([key_stats[key][a][1] for a in ALPHAS], dtype=float)
        hik = np.asarray([key_stats[key][a][2] for a in ALPHAS], dtype=float)
        ax.plot(x, yk, marker="o", linewidth=1.8)
        ax.fill_between(x, lok, hik, alpha=0.2)
        ax.set_title(key)
        ax.grid(alpha=0.25)
        ax.set_xlim(min(ALPHAS), max(ALPHAS))
    for ax in axes[1]:
        ax.set_xlabel("toxicity alpha")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean score (turns 2-4)")
    fig_grid.tight_layout()

    out_pdf = args.out_pdf
    if not os.path.isabs(out_pdf):
        out_pdf = os.path.join(os.path.dirname(__file__), out_pdf)
    ensure_parent(out_pdf)

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig_main)
        pdf.savefig(fig_grid)

    plt.close(fig_main)
    plt.close(fig_grid)
    print(f"[OK] wrote {out_pdf}")


if __name__ == "__main__":
    main()
