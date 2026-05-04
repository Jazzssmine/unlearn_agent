# src/real_agents/real_network.py

import networkx as nx
import pandas as pd

def build_real_network(followers_df, interactions_df=None):
    """
    Build a network graph from followers and interactions data.
    
    Args:
        followers_df: DataFrame with columns 'follower' and 'followee' 
                      (format: follower follows followee)
        interactions_df: DataFrame with columns 'user_id', 'replied_author', 
                        'thread_root_author', 'reposted_author', 'quoted_author', 'date'
    
    Returns:
        G: NetworkX DiGraph with edges from both followers and interactions
    """
    G = nx.DiGraph()
    
    # Build follower network
    # Check column names - handle both 'follower'/'followee' and 'follower_id'/'followee_id'
    if 'follower' in followers_df.columns and 'followee' in followers_df.columns:
        follower_col = 'follower'
        followee_col = 'followee'
    elif 'follower_id' in followers_df.columns and 'followee_id' in followers_df.columns:
        follower_col = 'follower_id'
        followee_col = 'followee_id'
    else:
        raise ValueError(f"Followers dataframe must have 'follower'/'followee' or 'follower_id'/'followee_id' columns. Found: {list(followers_df.columns)}")
    
    print(f"Building follower network from {len(followers_df)} follower relationships...")
    for _, row in followers_df.iterrows():
        follower = row[follower_col]
        followee = row[followee_col]
        # Skip if either is NaN
        if pd.notna(follower) and pd.notna(followee):
            G.add_edge(follower, followee, edge_type='follows')
    
    # Build interaction network
    if interactions_df is not None:
        print(f"Building interaction network from {len(interactions_df)} interactions...")
        
        # Add edges from replies
        if 'replied_author' in interactions_df.columns:
            reply_edges = interactions_df[['user_id', 'replied_author']].dropna()
            for _, row in reply_edges.iterrows():
                user = row['user_id']
                replied_to = row['replied_author']
                if pd.notna(user) and pd.notna(replied_to):
                    G.add_edge(user, replied_to, edge_type='replied_to')
        
        # Add edges from reposts
        if 'reposted_author' in interactions_df.columns:
            repost_edges = interactions_df[['user_id', 'reposted_author']].dropna()
            for _, row in repost_edges.iterrows():
                user = row['user_id']
                reposted_from = row['reposted_author']
                if pd.notna(user) and pd.notna(reposted_from):
                    G.add_edge(user, reposted_from, edge_type='reposted_from')
        
        # Add edges from quotes
        if 'quoted_author' in interactions_df.columns:
            quote_edges = interactions_df[['user_id', 'quoted_author']].dropna()
            for _, row in quote_edges.iterrows():
                user = row['user_id']
                quoted_from = row['quoted_author']
                if pd.notna(user) and pd.notna(quoted_from):
                    G.add_edge(user, quoted_from, edge_type='quoted_from')
    
    print(f"Network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G
