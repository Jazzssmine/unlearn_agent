#!/usr/bin/env bash
# §6.3 Channel-disentanglement experiments.
# Runs Conditions A–D for Table 3.  Does NOT re-run the two reference conditions
# (full-transcript and memory-no-sanitization) that already exist.
#
# Usage:
#   cd /u/anon3/unlearn_agent
#   bash scripts/run_sec63.sh          # run all four conditions
#   bash scripts/run_sec63.sh cond_a   # run only Condition A
#   bash scripts/run_sec63.sh cond_b cond_c   # run B and C
#
# Environment variables (all optional — defaults shown):
#   REDDIT_JSONL  path to seed file
#   N_SEEDS       number of seeds (default: 50)
#   N_ROLLOUTS    rollouts per seed/mode (default: 2)
#   MODEL         LLM model ID (default: gpt-4o-mini)
#   BASE_SEED     base random seed (default: 12345)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON=/u/anon3/.conda/envs/py311/bin/python

: "${REDDIT_JSONL:=agent/data/reddit/extracted/politics_seedA_BCD_chains_detoxify.jsonl}"
: "${N_SEEDS:=50}"
: "${N_ROLLOUTS:=2}"
: "${MODEL:=gpt-4o-mini}"
: "${BASE_SEED:=12345}"

if [[ ! -f "$REDDIT_JSONL" ]]; then
  echo "ERROR: reddit seed file not found: $REDDIT_JSONL" >&2
  echo "Set REDDIT_JSONL to your local path." >&2
  exit 1
fi

# Shared flags for all conditions
SHARED=(
  --seed_source reddit_jsonl
  --reddit_jsonl "$REDDIT_JSONL"
  --n_seeds "$N_SEEDS"
  --n_rollouts "$N_ROLLOUTS"
  --model "$MODEL"
  --base_random_seed "$BASE_SEED"
  --compute_toxicity
  --toxic_intensity strong
  --intervention_position pos1
  --rollout_output_mode per_rollout
)

# Which conditions to run (default: all)
RUN_ALL=true
declare -A RUN_COND
if [[ $# -gt 0 ]]; then
  RUN_ALL=false
  for arg in "$@"; do
    RUN_COND[$arg]=1
  done
fi

should_run() {
  if $RUN_ALL; then return 0; fi
  [[ -n "${RUN_COND[$1]+x}" ]] && return 0 || return 1
}

echo "=== §6.3 runner ==="
echo "REDDIT_JSONL=$REDDIT_JSONL  N_SEEDS=$N_SEEDS  N_ROLLOUTS=$N_ROLLOUTS"
echo "MODEL=$MODEL  BASE_SEED=$BASE_SEED"
echo ""

# ── Condition A: memory-only (no parent message) ──────────────────────────────
if should_run cond_a; then
  echo "[A] Memory-only (no parent message in context)"
  OUT_DIR="$REPO_ROOT/data/chain/sec63/cond_a_memory_only"
  mkdir -p "$OUT_DIR"
  cd "$REPO_ROOT/src"
  $PYTHON -m run_influence_sec63_cond_a \
    "${SHARED[@]}" \
    --memory_mode memory \
    --memory_sanitize none \
    --sanitize_threshold 0.5 \
    --context_mode full \
    --out_jsonl "$OUT_DIR/influence_threads_rollout.jsonl" \
    --out_summary "$OUT_DIR/summary.json"
  echo "[A] Done."
fi

# ── Condition B: parent-only transcript (no memory) ───────────────────────────
if should_run cond_b; then
  echo "[B] Parent-only transcript (no memory)"
  OUT_DIR="$REPO_ROOT/data/chain/sec63/cond_b_parent_only"
  mkdir -p "$OUT_DIR"
  cd "$REPO_ROOT/src"
  $PYTHON -m run_influence_baseline \
    "${SHARED[@]}" \
    --context_mode parent_only \
    --memory_mode none \
    --memory_sanitize none \
    --sanitize_threshold 0.5 \
    --out_jsonl "$OUT_DIR/influence_threads.jsonl" \
    --out_summary "$OUT_DIR/summary.json"
  echo "[B] Done."
fi

# ── Condition C: memory + rewrite (tau=0.5) ───────────────────────────────────
if should_run cond_c; then
  echo "[C] Memory + rewrite (tau=0.5)"
  OUT_DIR="$REPO_ROOT/data/chain/sec63/cond_c_rewrite"
  mkdir -p "$OUT_DIR"
  cd "$REPO_ROOT/src"
  $PYTHON -m run_influence_baseline \
    "${SHARED[@]}" \
    --context_mode full \
    --memory_mode memory \
    --memory_sanitize rewrite \
    --sanitize_threshold 0.5 \
    --out_jsonl "$OUT_DIR/influence_threads.jsonl" \
    --out_summary "$OUT_DIR/summary.json"
  echo "[C] Done."
fi

# ── Condition D: memory + gate (tau=0.5) ──────────────────────────────────────
if should_run cond_d; then
  echo "[D] Memory + gate (tau=0.5)"
  OUT_DIR="$REPO_ROOT/data/chain/sec63/cond_d_gate"
  mkdir -p "$OUT_DIR"
  cd "$REPO_ROOT/src"
  $PYTHON -m run_influence_baseline \
    "${SHARED[@]}" \
    --context_mode full \
    --memory_mode memory \
    --memory_sanitize gate \
    --sanitize_threshold 0.5 \
    --out_jsonl "$OUT_DIR/influence_threads.jsonl" \
    --out_summary "$OUT_DIR/summary.json"
  echo "[D] Done."
fi

echo ""
echo "=== All requested conditions finished ==="
echo "Run the analysis:"
echo "  cd $REPO_ROOT && python scripts/compute_sec63_stats.py"
