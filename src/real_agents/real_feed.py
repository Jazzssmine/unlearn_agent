# src/real_agents/real_feed.py

import pandas as pd
import random

def _safe_text(value):
    """Normalize text content."""
    if pd.isna(value):
        return "No content"
    return str(value)

def get_topic_similarity_score(user_history_text, candidate_text):
    """
    Simple Jaccard-like similarity: counts overlapping words.
    """
    if not isinstance(candidate_text, str) or not isinstance(user_history_text, str):
        return 0
    
    user_words = set(user_history_text.lower().split())
    candidate_words = set(candidate_text.lower().split())
    
    # Calculate overlap
    intersection = user_words.intersection(candidate_words)
    return len(intersection)

def build_feed(user_profile, network, all_posts_df):
    """
    Builds a curated home feed of 5 items:
    - 2 from accounts the user follows
    - 2 from similar topics (content relevance)
    - 1 random discovery post
    """
    feed_items = []
    
    # 1. Identify relationships
    agent_id = user_profile.user_id
    try:
        # In NetworkX DiGraph, successors(u) are nodes v where u -> v (u follows v)
        following_ids = list(network.successors(agent_id))
    except (KeyError, AttributeError):
        following_ids = []

    # Prepare pools
    # Convert dates to datetime for sorting later
    if 'date' in all_posts_df.columns:
        all_posts_df['date'] = pd.to_datetime(all_posts_df['date'])

    # --- SELECTION 1: FOLLOWING (2 posts) ---
    following_df = all_posts_df[all_posts_df['user_id'].isin(following_ids)]
    
    if len(following_df) >= 2:
        selected_following = following_df.sample(2)
    else:
        selected_following = following_df  # Take what we have if < 2

    # --- SELECTION 2: SIMILAR TOPICS (2 posts) ---
    # We want candidates that are NOT in the 'following' set we just picked
    # and NOT the user's own posts.
    exclude_ids = set(selected_following.index.tolist())
    candidate_pool = all_posts_df[
        (~all_posts_df.index.isin(exclude_ids)) & 
        (all_posts_df['user_id'] != agent_id)
    ]

    # Create a "profile signature" from the user's recent posts
    user_history = " ".join(user_profile.posts['text'].head(20).astype(str).tolist())

    # To keep it fast, we score a random subset of candidates rather than the whole database
    scoring_pool = candidate_pool.sample(min(len(candidate_pool), 100)) if len(candidate_pool) > 0 else candidate_pool
    
    scored_posts = []
    for idx, row in scoring_pool.iterrows():
        score = get_topic_similarity_score(user_history, str(row.get('text', '')))
        scored_posts.append((score, idx))
    
    # Sort by score descending and pick top 2
    scored_posts.sort(key=lambda x: x[0], reverse=True)
    top_topic_indices = [x[1] for x in scored_posts[:2]]
    
    selected_topics = all_posts_df.loc[top_topic_indices]

    # --- SELECTION 3: RANDOM DISCOVERY (1 post) ---
    # Exclude everything selected so far
    exclude_ids.update(selected_topics.index.tolist())
    remaining_pool = all_posts_df[
        (~all_posts_df.index.isin(exclude_ids)) & 
        (all_posts_df['user_id'] != agent_id)
    ]
    
    if len(remaining_pool) > 0:
        selected_random = remaining_pool.sample(1)
    else:
        selected_random = pd.DataFrame()

    # --- COMBINE AND FORMAT ---
    final_df = pd.concat([selected_following, selected_topics, selected_random])
    
    # Sort chronologically (newest first) to mimic a real timeline
    if 'date' in final_df.columns:
        final_df = final_df.sort_values('date', ascending=False)

    for _, row in final_df.iterrows():
        feed_items.append({
            "type": "post", # Assuming these are primary posts
            "content": _safe_text(row.get("text", "")),
            "timestamp": row.get("date", None),
            "author_id": str(row.get("user_id", "unknown")),
            "metadata": dict(row),
        })

    return feed_items