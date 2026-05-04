# Unlearn Agent

Unified README for the cleaned `unlearn_agent` release.

This repository studies toxicity propagation and memory-mediated behavior in multi-agent conversation simulations, with additional evaluations for memory laundering, channel attribution, cross-topology/model checks, and defense ablations.

## Repository Layout

- `src/`: simulation and evaluation entrypoints
- `data/`: extracted seeds/chains and experiment outputs
- `analysis/`: postprocessing and plotting scripts
- `experiments/`: higher-level runners (dose-response, sanitization, DPO)
- `scripts/`: utility workflows (including smoking-gun pipeline)

## Environment

From repo root:

```bash
pip install pandas networkx torch detoxify tqdm matplotlib numpy scipy jupyter
```

## Core Workflow

### 1) Build extraction data

```bash
python data/build_reddit.py
python data/build_threads.py
```

Expected key files:
- `data/extracted/politics_depth_ge5.jsonl`
- `data/extracted/politics_seedA_BCD_chains_detoxify.jsonl`

### 2) Run baseline chain simulation

```bash
cd src
python -m run_influence_baseline \
  --seed_source reddit_jsonl \
  --reddit_jsonl agent/data/reddit/extracted/politics_seedA_BCD_chains_detoxify.jsonl \
  --seed_strategy root \
  --reddit_require_max_depth -1 \
  --out_jsonl ../data/reddit/influence_baseline_threads_reddit_rollouts.jsonl \
  --out_summary ../data/reddit/influence_baseline_summary_reddit_rollouts.json \
  --model gpt-4o-mini \
  --toxic_intensity strong \
  --n_seeds 200 \
  --n_rollouts 2 \
  --base_random_seed 12345 \
  --rollout_output_mode per_rollout \
  --intervention_position pos1 \
  --compute_toxicity
```

Common flags:
- `--context_mode {full,parent_only,seed_only,memory_only}`
- `--memory_mode {none,memory}`
- `--memory_sanitize {none,rewrite,gate}`
- `--write_gate {none,redact,rewrite}`
- `--sanitize_threshold 0.5`

### 3) Evaluate / summarize

```bash
python -m evaluate_single_thread_influence \
  --mild_path ../data/reddit/influence_baseline_threads_detoxify_mild.jsonl \
  --medium_path ../data/reddit/influence_baseline_threads_detoxify_medium.jsonl \
  --strong_path ../data/reddit/influence_baseline_threads_detoxify_strong.jsonl \
  --analysis_dir ../analysis
```

```bash
cd ..
python analysis/compute_summary_stats.py \
  --input_jsonl data/reddit/influence_baseline_threads_reddit_rollouts.jsonl \
  --out_csv analysis/summary_stats.csv \
  --out_pdf analysis/figures/effect_size_histogram.pdf
```

## Main Experiment Tracks

### Section 6.2: Memory laundering
- Script: `src/eval/evaluate_memory_laundering.py`
- Outputs: laundering records, SPG/table metrics, paired stats, summary JSON, top examples

### Section 6.3: Transcript vs memory channels
- Runner: `src/run_influence_baseline.py` (`memory_only`, `parent_only`, rewrite/gate toggles)
- Stats: `scripts/compute_sec63_stats.py`

### Section 6.4: Cross-topology / cross-model validation
- Tree topology runner: `src/run_influence_graph.py` (`--topology tree`)
- Qwen cross-model path documented via `README_qwen.md` workflow details

### Section 6.5: Defense comparison and full ablation
- Baseline + state-control + DPO-integrated comparisons
- DPO pipeline: `experiments/build_dpo_pairs.py`, `experiments/train_dpo.py`, `experiments/evaluate_dpo.py`

### Smoking-gun causal isolation
- `scripts/extract_memory_states.py`
- `scripts/run_smoking_gun.py`
- `scripts/analyze_smoking_gun.py`

This isolates memory-only causal effects by fixing transcript context and varying only injected memory summaries.

## Additional Runners

Dose-response:

```bash
python experiments/run_dose_response.py \
  --seed_source reddit_jsonl \
  --reddit_jsonl agent/data/reddit/extracted/politics_seedA_BCD_chains_detoxify.jsonl \
  --model gpt-4o-mini \
  --toxic_intensity strong \
  --n_seeds 200 \
  --data_dir data/dose_response \
  --summary_dir data/dose_response
```

Sanitization ablation:

```bash
python experiments/run_sanitization_ablation.py \
  --seed_source reddit_jsonl \
  --reddit_jsonl agent/data/reddit/extracted/politics_seedA_BCD_chains_detoxify.jsonl \
  --model gpt-4o-mini \
  --toxic_intensity strong \
  --n_seeds 200 \
  --n_rollouts 3 \
  --sanitize_threshold 0.5 \
  --data_dir data/sanitization_ablation \
  --summary_dir data/sanitization_ablation
```

