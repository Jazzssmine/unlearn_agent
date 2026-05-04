#!/usr/bin/env python3
import argparse
import csv
import os

import pandas as pd
from detoxify import Detoxify
import torch
from tqdm import tqdm


def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Detoxify(model_name, device=device)


def score_texts(model, texts, batch_size):
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        outputs = model.predict(batch)
        scores.extend(outputs.get("toxicity", []))
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Score toxicity for agent_reply and reaction_text in simulation results."
    )
    parser.add_argument(
        "--input_csv",
        default="/u/anon3/toxic_agent/data/simulation_results_intervention.csv",
        help="Path to simulation results CSV.",
    )
    parser.add_argument(
        "--output_csv",
        default="/u/anon3/toxic_agent/data/simulation_results_intervention_scored.csv",
        help="Path to output CSV with toxicity scores.",
    )
    parser.add_argument(
        "--model",
        default="original",
        help="Detoxify model: original | unbiased | multilingual",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for Detoxify scoring",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    model = load_model(args.model)

    # Score agent_reply
    agent_texts = df["agent_reply"].fillna("").astype(str).tolist()
    agent_scores = [None] * len(agent_texts)
    agent_idxs = [i for i, t in enumerate(agent_texts) if t.strip()]
    if agent_idxs:
        agent_to_score = [agent_texts[i] for i in agent_idxs]
        scored = score_texts(model, agent_to_score, args.batch_size)
        for idx, score in zip(agent_idxs, scored):
            agent_scores[idx] = score
    df["agent_reply_toxicity"] = agent_scores

    # Score reaction_text
    reaction_texts = df["reaction_text"].fillna("").astype(str).tolist()
    reaction_scores = [None] * len(reaction_texts)
    reaction_idxs = [i for i, t in enumerate(reaction_texts) if t.strip()]
    if reaction_idxs:
        reaction_to_score = [reaction_texts[i] for i in reaction_idxs]
        scored = score_texts(model, reaction_to_score, args.batch_size)
        for idx, score in zip(reaction_idxs, scored):
            reaction_scores[idx] = score
    df["reaction_text_toxicity"] = reaction_scores

    df.to_csv(args.output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved scored results to {args.output_csv}")


if __name__ == "__main__":
    main()


"""
python ./data/score_simulation_results.py \
  --input_csv ./data/simulation_results_intervention_random.csv \
  --output_csv ./data/simulation_results_intervention_random_scored.csv \
  --model original \
  --batch_size 64
"""