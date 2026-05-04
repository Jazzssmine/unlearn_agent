#!/usr/bin/env bash
# Memory contamination experiment - chain topology (run_influence_baseline.py)
# Runs toxic/neutral memory conditions used by contamination analysis.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/src"

: "${REDDIT_JSONL:=../data/reddit_posts.jsonl}"
if [[ ! -f "$REDDIT_JSONL" ]]; then
  ALT_REDDIT_JSONL="agent/data/reddit/extracted/politics_seedA_BCD_chains_detoxify.jsonl"
  if [[ -f "$ALT_REDDIT_JSONL" ]]; then
    REDDIT_JSONL="$ALT_REDDIT_JSONL"
  else
    echo "ERROR: Reddit seed file not found: $REDDIT_JSONL" >&2
    echo "Set REDDIT_JSONL to your local reddit jsonl path." >&2
    exit 1
  fi
fi

: "${N_SEEDS:=100}"
: "${N_ROLLOUTS:=3}"
: "${MODEL:=gpt-4o-mini}"
: "${BASE_SEED:=42}"

mkdir -p ../data/memory

echo "Running memory contamination experiments..."
echo "REDDIT_JSONL=$REDDIT_JSONL"
echo "MODEL=$MODEL N_SEEDS=$N_SEEDS N_ROLLOUTS=$N_ROLLOUTS BASE_SEED=$BASE_SEED"

# Condition 1: toxic + memory + no sanitization
python -m run_influence_baseline \
  --seed_source reddit_jsonl \
  --reddit_jsonl "$REDDIT_JSONL" \
  --n_seeds "$N_SEEDS" \
  --n_rollouts "$N_ROLLOUTS" \
  --model "$MODEL" \
  --base_random_seed "$BASE_SEED" \
  --modes toxic \
  --memory_mode memory \
  --memory_sanitize none \
  --compute_toxicity \
  --out_jsonl ../data/memory/chain_toxic_memory_none.jsonl \
  --out_summary ../data/memory/chain_toxic_memory_none_summary.json \
  --rollout_output_mode combined

# Condition 2: neutral + memory + no sanitization
python -m run_influence_baseline \
  --seed_source reddit_jsonl \
  --reddit_jsonl "$REDDIT_JSONL" \
  --n_seeds "$N_SEEDS" \
  --n_rollouts "$N_ROLLOUTS" \
  --model "$MODEL" \
  --base_random_seed "$BASE_SEED" \
  --modes neutral \
  --memory_mode memory \
  --memory_sanitize none \
  --compute_toxicity \
  --out_jsonl ../data/memory/chain_neutral_memory_none.jsonl \
  --out_summary ../data/memory/chain_neutral_memory_none_summary.json \
  --rollout_output_mode combined

# Condition 3: toxic + memory + rewrite sanitization
python -m run_influence_baseline \
  --seed_source reddit_jsonl \
  --reddit_jsonl "$REDDIT_JSONL" \
  --n_seeds "$N_SEEDS" \
  --n_rollouts "$N_ROLLOUTS" \
  --model "$MODEL" \
  --base_random_seed "$BASE_SEED" \
  --modes toxic \
  --memory_mode memory \
  --memory_sanitize rewrite \
  --sanitize_threshold 0.05 \
  --compute_toxicity \
  --out_jsonl ../data/memory/chain_toxic_memory_rewrite.jsonl \
  --out_summary ../data/memory/chain_toxic_memory_rewrite_summary.json \
  --rollout_output_mode combined

# Condition 4: toxic + read_sanitize redact only, memory unsanitized
python -m run_influence_baseline \
  --seed_source reddit_jsonl \
  --reddit_jsonl "$REDDIT_JSONL" \
  --n_seeds "$N_SEEDS" \
  --n_rollouts "$N_ROLLOUTS" \
  --model "$MODEL" \
  --base_random_seed "$BASE_SEED" \
  --modes toxic \
  --memory_mode memory \
  --memory_sanitize none \
  --read_sanitize redact \
  --sanitize_threshold 0.05 \
  --compute_toxicity \
  --out_jsonl ../data/memory/chain_toxic_memory_redact_read_only.jsonl \
  --out_summary ../data/memory/chain_toxic_memory_redact_read_only_summary.json \
  --rollout_output_mode combined

echo "All memory contamination runs complete."
