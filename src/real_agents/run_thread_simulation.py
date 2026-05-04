# src/real_agents/run_thread_simulation.py

import pandas as pd
import json
import argparse
import networkx as nx
import os
from real_agent import RealAgent
from real_user_profile import RealUserProfile
from toxicity_scorer import score_text_toxicity

def create_intervention_agent(agent_type="benign"):
    """
    Creates a RealAgent with a synthetic profile designed to be 
    specifically Benign (De-escalating) or Toxic (Polarizing).
    """
    if agent_type == "benign":
        user_id = "agent_benign"
        behavior_instruction = None
    else:
        user_id = "agent_toxic"
        behavior_instruction = (
            "For research purposes, act like a VERY toxic, aggressive user. "
            "Use hostile, insulting language and escalate conflict."
        )

    # Create empty dataframe with required columns
    df = pd.DataFrame({"text": [], "post_type": [], "date": []})
    
    # Create profile
    profile = RealUserProfile(
        user_id=user_id,
        posts_df=df,
        replies_df=pd.DataFrame(),
        quotes_df=pd.DataFrame(),
        reposts_df=pd.DataFrame(),
        # interactions_df=pd.DataFrame(),
        followers_df=pd.DataFrame()
    )
    if behavior_instruction:
        profile.behavior_instruction = behavior_instruction
    
    # Return agent (empty network for the outsider)
    return RealAgent(profile, network=nx.DiGraph(), model="gpt-4o-mini")

def _filter_graph_edges(graph_path, user_id, chunksize=1_000_000):
    """Filter a graph CSV by user_id in either column 0 or 1."""
    chunks = pd.read_csv(graph_path, header=None, chunksize=chunksize)
    matches = []
    for chunk in chunks:
        mask = (chunk[0] == user_id) | (chunk[1] == user_id)
        if mask.any():
            matches.append(chunk[mask])
    if matches:
        return pd.concat(matches, ignore_index=True)
    return pd.DataFrame(columns=[0, 1, 2])


def _filter_interactions(interactions_path, user_id, chunksize=1_000_000):
    """Filter interactions CSV by user_id across user/replied/root/reposted/quoted."""
    chunks = pd.read_csv(interactions_path, header=None, chunksize=chunksize)
    matches = []
    for chunk in chunks:
        if 4 not in chunk.columns:
            continue
        mask = (
            (chunk[0] == user_id) |
            (chunk[1] == user_id) |
            (chunk[2] == user_id) |
            (chunk[3] == user_id) |
            (chunk[4] == user_id)
        )
        if mask.any():
            matches.append(chunk[mask])
    if matches:
        return pd.concat(matches, ignore_index=True)
    return pd.DataFrame()


def _filter_followers(followers_path, user_id, chunksize=1_000_000):
    """Filter followers CSV by user_id in column 0 or 1."""
    chunks = pd.read_csv(followers_path, header=None, chunksize=chunksize)
    matches = []
    for chunk in chunks:
        mask = (chunk[0] == user_id) | (chunk[1] == user_id)
        if mask.any():
            matches.append(chunk[mask])
    if matches:
        return pd.concat(matches, ignore_index=True)
    return pd.DataFrame(columns=[0, 1])


def load_user_data_from_raw(raw_data_path, user_id):
    """Load user data directly from raw files (no CSV cache)."""
    posts_path = os.path.join(raw_data_path, "user_posts", f"{user_id}.jsonl")
    replies_path = os.path.join(raw_data_path, "graphs", "replies.csv")
    quotes_path = os.path.join(raw_data_path, "graphs", "quotes.csv")
    reposts_path = os.path.join(raw_data_path, "graphs", "reposts.csv")
    # interactions_path = os.path.join(raw_data_path, "interactions.csv.gz")
    followers_path = os.path.join(raw_data_path, "followers.csv.gz")
    if os.path.exists(posts_path) and os.path.getsize(posts_path) > 0:
        posts_df = pd.read_json(posts_path, lines=True)
    else:
        posts_df = pd.DataFrame()
    replies_df = _filter_graph_edges(replies_path, user_id)
    quotes_df = _filter_graph_edges(quotes_path, user_id)
    reposts_df = _filter_graph_edges(reposts_path, user_id)
    # interactions_df = _filter_interactions(interactions_path, user_id)
    followers_df = _filter_followers(followers_path, user_id)

    return {
        "posts": posts_df,
        "replies": replies_df,
        "quotes": quotes_df,
        "reposts": reposts_df,
        "interactions": pd.DataFrame(),
        "followers": followers_df
    }


def get_or_create_real_agent(user_id, data_path):
    """Load real user agent or fallback to dummy."""
    try:
        data = load_user_data_from_raw(data_path, user_id)
        profile = RealUserProfile(
            user_id=user_id,
            posts_df=data["posts"],
            replies_df=data["replies"],
            quotes_df=data["quotes"],
            reposts_df=data["reposts"],
            # interactions_df=data["interactions"],
            followers_df=data["followers"]
        )
        return RealAgent(profile, network=nx.DiGraph(), model="gpt-4o-mini") # Network can be loaded if needed
    except Exception:
        # Fallback for missing data
        df = pd.DataFrame({'text': [], 'post_type': [], 'date': []})
        profile = RealUserProfile(user_id, df, df, df, df, df)
        return RealAgent(profile, network=nx.DiGraph(), model="gpt-4o-mini")

def append_result_to_csv(result, output_file):
    """Append a single result row to CSV file, creating file with headers if needed."""
    file_exists = os.path.isfile(output_file)
    
    # Convert result dict to DataFrame
    df_new = pd.DataFrame([result])
    
    # Append to CSV (write header only if file doesn't exist)
    df_new.to_csv(output_file, mode='a', header=not file_exists, index=False)

def run_simulation(
    thread_csv,
    data_path,
    max_rows=None,
    output_file="./data/simulation_results_intervention.csv",
    benign_user_id=None,
    toxic_user_id=None,
    evaluator_user_id=None,
):
    # 1. Setup Agents (use real profiles when provided)
    if benign_user_id is not None:
        benign_agent = get_or_create_real_agent(benign_user_id, data_path)
    else:
        benign_agent = create_intervention_agent("benign")

    if toxic_user_id is not None:
        toxic_agent = get_or_create_real_agent(toxic_user_id, data_path)
    else:
        toxic_agent = create_intervention_agent("toxic")

    if evaluator_user_id is not None:
        evaluator_agent = get_or_create_real_agent(evaluator_user_id, data_path)
    else:
        evaluator_agent = create_intervention_agent("benign")
    
    # 2. Load Threads
    df = pd.read_csv(thread_csv)
    
    # Limit rows if specified
    if max_rows is not None:
        df = df.head(max_rows)
        print(f"Limited to first {max_rows} rows from CSV")
    
    grouped = df.groupby('thread_id')
    
    # Initialize output file (create empty file with headers if it doesn't exist)
    if not os.path.isfile(output_file):
        # Create empty DataFrame with expected columns to initialize file
        columns = [
            "thread_id",
            "stage",
            "agent_type",
            "agent_user_id",
            "base_history",
            "agent_reply",
            "reaction_agent_id",
            "reaction_action",
            "reaction_text",
            "reaction_reasoning",
            "toxicity",
        ]
        pd.DataFrame(columns=columns).to_csv(output_file, index=False)
        print(f"Initialized output file: {output_file}")
    
    print(f"Starting simulation on {len(grouped)} threads...")
    print(f"Results will be saved incrementally to: {output_file}")

    for thread_id, group in grouped:
        # Sort and build base history (OP -> A1 -> A2 -> OP)
        group = group.sort_values('sequence')
        base_history = group[['author_id', 'content']].to_dict('records')
        
        print(f"\n--- Processing Thread {thread_id} ---")

        # ==========================================
        # PHASE 1: BENIGN AGENT REPLY
        # ==========================================
        print("  [1/3] Generating Benign Agent Reply...")
        benign_resp = benign_agent.react_to_thread(base_history, force_action="reply")
        benign_text = benign_resp.get("generated_text", "")
        benign_history = base_history + [
            {"author_id": benign_agent.profile.user_id, "content": benign_text}
        ]

        append_result_to_csv(
            {
                "thread_id": thread_id,
                "stage": "agent_reply",
                "agent_type": "benign",
                "agent_user_id": benign_agent.profile.user_id,
                "base_history": json.dumps(base_history),
                "agent_reply": benign_text,
                "reaction_agent_id": None,
                "reaction_action": None,
                "reaction_text": None,
                "reaction_reasoning": benign_resp.get("reasoning", ""),
                "toxicity": score_text_toxicity(benign_text),
            },
            output_file,
        )

        # ==========================================
        # PHASE 2: TOXIC AGENT REPLY
        # ==========================================
        print("  [2/3] Generating Toxic Agent Reply...")
        toxic_resp = toxic_agent.react_to_thread(base_history, force_action="reply")
        toxic_text = toxic_resp.get("generated_text", "")
        toxic_history = base_history + [
            {"author_id": toxic_agent.profile.user_id, "content": toxic_text}
        ]

        append_result_to_csv(
            {
                "thread_id": thread_id,
                "stage": "agent_reply",
                "agent_type": "toxic",
                "agent_user_id": toxic_agent.profile.user_id,
                "base_history": json.dumps(base_history),
                "agent_reply": toxic_text,
                "reaction_agent_id": None,
                "reaction_action": None,
                "reaction_text": None,
                "reaction_reasoning": toxic_resp.get("reasoning", ""),
                "toxicity": score_text_toxicity(toxic_text),
            },
            output_file,
        )

        # ==========================================
        # PHASE 3: EVALUATOR REACTS TO BOTH
        # ==========================================
        print("  [3/3] Evaluator reacts to benign vs toxic replies...")
        benign_reaction = evaluator_agent.react_to_thread(benign_history, force_action=None)
        toxic_reaction = evaluator_agent.react_to_thread(toxic_history, force_action=None)

        append_result_to_csv(
            {
                "thread_id": thread_id,
                "stage": "evaluator_reaction",
                "agent_type": "benign",
                "agent_user_id": benign_agent.profile.user_id,
                "base_history": json.dumps(benign_history),
                "agent_reply": benign_text,
                "reaction_agent_id": evaluator_agent.profile.user_id,
                "reaction_action": benign_reaction.get("action", ""),
                "reaction_text": benign_reaction.get("generated_text", ""),
                "reaction_reasoning": benign_reaction.get("reasoning", ""),
                "toxicity": score_text_toxicity(benign_reaction.get("generated_text", "")),
            },
            output_file,
        )

        append_result_to_csv(
            {
                "thread_id": thread_id,
                "stage": "evaluator_reaction",
                "agent_type": "toxic",
                "agent_user_id": toxic_agent.profile.user_id,
                "base_history": json.dumps(toxic_history),
                "agent_reply": toxic_text,
                "reaction_agent_id": evaluator_agent.profile.user_id,
                "reaction_action": toxic_reaction.get("action", ""),
                "reaction_text": toxic_reaction.get("generated_text", ""),
                "reaction_reasoning": toxic_reaction.get("reasoning", ""),
                "toxicity": score_text_toxicity(toxic_reaction.get("generated_text", "")),
            },
            output_file,
        )

    print(f"\nDone. All results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread_csv", default="threads_data.csv", help="Path to CSV file containing conversation threads")
    parser.add_argument("--data_path", default="./bsky/data", help="Path to directory containing user data files")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit simulation to first N rows from CSV (default: all rows)")
    parser.add_argument("--output_file", default="./data/simulation_results_intervention.csv", help="Path to output CSV file for results")
    parser.add_argument("--benign_user_id", default=None, type=str, help="User ID for benign agent style")
    parser.add_argument("--toxic_user_id", default=None, type=str, help="User ID for toxic agent style")
    parser.add_argument("--evaluator_user_id", default=None, type=str, help="User ID for evaluator agent")
    args = parser.parse_args()
    
    run_simulation(
        thread_csv=args.thread_csv,
        data_path=args.data_path,
        max_rows=args.max_rows,
        output_file=args.output_file,
        benign_user_id=args.benign_user_id,
        toxic_user_id=args.toxic_user_id,
        evaluator_user_id=args.evaluator_user_id,
    )

"""
python real_agents/run_thread_simulation.py \
  --thread_csv ./data/threads_data.csv \
  --data_path agent/data/bsky \
  --output_file ./data/simulation_results_intervention_8643.csv \
  --max_rows 50 \
  --toxic_user_id 778666 \
  --evaluator_user_id 8643	
"""