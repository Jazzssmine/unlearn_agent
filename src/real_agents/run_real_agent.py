# src/real_agents/run_real_agent.py

import argparse
import pandas as pd
from real_user_profile import RealUserProfile
from real_feed import build_feed
from real_agent import RealAgent
from real_network import build_real_network

import json

def load_user_data(path, user_id):
    return {
        "posts": pd.read_csv(f"{path}/user_post_{user_id}.csv"),
        "replies": pd.read_csv(f"{path}/replies_with_{user_id}.csv"),
        "quotes": pd.read_csv(f"{path}/quotes_with_{user_id}.csv"),
        "reposts": pd.read_csv(f"{path}/reposts_with_{user_id}.csv"),
        "interactions": pd.read_csv(f"{path}/interactions_{user_id}.csv"),
        "followers": pd.read_csv(f"{path}/followers_{user_id}.csv"),
    }

def main(user_id="39765", neighbor_user_id=None, data_path=None):
    if data_path is None:
        data_path = "agent_39765"
    
    # Load data for target user (the one we're simulating)
    print(f"Loading data for target user: {user_id}")
    target_data = load_user_data(data_path, user_id)
    target_profile = RealUserProfile(
        user_id=user_id,
        posts_df=target_data["posts"],
        replies_df=target_data["replies"],
        quotes_df=target_data["quotes"],
        reposts_df=target_data["reposts"],
        interactions_df=target_data["interactions"],
        followers_df=target_data["followers"]
    )
    import pdb; pdb.set_trace()
    
    # Build network from target user's data
    # network = build_real_network(target_data["followers"], target_data["interactions"])
    network = build_real_network(target_data["followers"])

    # Create agent for target user
    agent = RealAgent(target_profile, network)

    # Build feed from neighbor user's content (if neighbor_user_id is provided)
    if neighbor_user_id is not None:
        print(f"Loading data for neighbor user: {neighbor_user_id}")
        neighbor_data = load_user_data(data_path, neighbor_user_id)
        neighbor_profile = RealUserProfile(
            user_id=neighbor_user_id,
            posts_df=neighbor_data["posts"],
            replies_df=neighbor_data["replies"],
            quotes_df=neighbor_data["quotes"],
            reposts_df=neighbor_data["reposts"],
            interactions_df=neighbor_data["interactions"],
            followers_df=neighbor_data["followers"]
        )
        print(f"Building feed from neighbor user {neighbor_user_id}'s content...")
        feed = build_feed(neighbor_profile)
    else:
        # If no neighbor specified, use target user's own feed (fallback)
        print(f"No neighbor user specified, using target user's own feed")
        feed = build_feed(target_profile)
    
    print(f"Feed contains {len(feed)} items")
    print(f"Target user {user_id} will observe and respond to these items\n")

    # Simulate: target user observes neighbor's feed and takes actions
    logs = []
    for i, item in enumerate(feed):
        current_author = item.get("author_id", neighbor_user_id if neighbor_user_id else user_id)

        print(f"Processing feed item {i+1}/{len(feed)}: {item.get('type', 'unknown')} from user {neighbor_user_id if neighbor_user_id else user_id}")
        action_result = agent.step(item)

        # Print a concise summary to stdout for easier debugging
        print(
            f"  -> Content action: {action_result.get('content_action', '')} | "
            f"Relationship action: {action_result.get('relationship_action', '')}"
        )

        logs.append({
            "feed_item": item,
            "target_user_id": user_id,
            "source_user_id": current_author,
            "content_action": action_result.get("content_action", ""), 
            "relationship_action": action_result.get("relationship_action", ""), 
            "reasoning_content": action_result.get("reasoning_content", ""), 
            "reasoning_prior_behavior": action_result.get("reasoning_prior_behavior", ""),
            "reasoning_relationship": action_result.get("reasoning_relationship", ""), 
            "generated_text": action_result.get("generated_text", ""),
            # Log the relationship status *after* the agent's step is complete
            "is_following_source": agent.network.has_edge(user_id, current_author),
            # Structured view of the model's output
            "action": action_result.get("action", ""),
            "reasoning_current_tweet": action_result.get("reasoning_current_tweet", ""),
            "reasoning_prior_behavior": action_result.get("reasoning_prior_behavior", ""),
            "generated_text": action_result.get("generated_text", ""),
            # Optionally keep the raw dict in case the schema evolves
            "raw_action": action_result,
        })

    print(f"\nSimulation complete. Target user {user_id} processed {len(logs)} feed items.")
    with open(f"./bsky/logs_{user_id}.json", "w") as f:
        json.dump(logs, f)
    return logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real agent simulation")
    parser.add_argument("--user_id", default="39765", type=str, 
                       help="Target user ID to simulate (the agent that will observe and respond)")
    parser.add_argument("--neighbor_user_id", default="39699", type=str,
                       help="Neighbor user ID whose content will be shown in the feed (if not provided, uses target user's own feed)")
    parser.add_argument("--data_path", default="./bsky/data", type=str, 
                       help="Path to the data directory containing user CSV files")
    
    args = parser.parse_args()
    main(user_id=args.user_id, neighbor_user_id=args.neighbor_user_id, data_path=args.data_path)
    """
    python src/real_agents/run_real_agent.py \
        --data_path agent/data/bsky/sample \
        --user_id 39765 --neighbor_user_id 39699

    """