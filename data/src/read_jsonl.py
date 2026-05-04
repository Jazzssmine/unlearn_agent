import json

import pandas as pd
from detoxify import Detoxify
import torch
from tqdm import tqdm


# Score each message with Detoxify and write back to JSONL, then save per-seed/turn CSV

model_name = "original"  # or "unbiased" / "multilingual"
model = Detoxify(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

# JSONL path (relative to this data directory)
path = "reddit/influence_baseline_threads_reddit_strong_llama3_8b.jsonl"

with open(path, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(records)} records")

# Score messages in batches and attach metrics per message
batch_size = 64
all_scores_by_mode = {}

for rec in records:
    mode = rec.get("mode", "unknown")
    msgs = rec.get("messages", [])
    texts = [m.get("text", "") for m in msgs]

    scored = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        outputs = model.predict(batch)
        for j in range(len(batch)):
            msg_scores = {k: float(outputs[k][j]) for k in outputs.keys()}
            scored.append(msg_scores)

    for m, s in zip(msgs, scored):
        m["detoxify"] = s

    all_scores_by_mode.setdefault(mode, []).extend(scored)

# Write back to the same JSONL file (overwrite)
with open(path, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

# Flatten per-seed / per-turn toxicity scores (both neutral and toxic modes) plus text to CSV
rows = []
for rec in records:
    seed_id = rec.get("seed_id")
    mode = rec.get("mode")
    for msg in rec.get("messages", []):
        turn = msg.get("turn")
        det = msg.get("detoxify") or {}
        rows.append(
            {
                "seed_id": seed_id,
                "mode": mode,
                "turn": turn,
                "text": msg.get("text", ""),
                "toxicity": det.get("toxicity"),
                "severe_toxicity": det.get("severe_toxicity"),
                "obscene": det.get("obscene"),
                "threat": det.get("threat"),
                "insult": det.get("insult"),
                "identity_attack": det.get("identity_attack"),
            }
        )

df = pd.DataFrame(rows)
out_csv = "reddit/influence_baseline_threads_reddit_strong_llama3_8b_flat.csv"
df.to_csv(out_csv, index=False)
print(f"Wrote per-seed/turn Detoxify scores to {out_csv}")
