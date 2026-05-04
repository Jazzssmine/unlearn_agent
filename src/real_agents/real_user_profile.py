# src/real_agents/real_user_profile.py

import pandas as pd

class RealUserProfile:
    """
    A profile built from REAL user data, unlike the synthetic personas
    used in LLM-SocioPol.
    """
    def __init__(
        self,
        user_id: str,
        posts_df: pd.DataFrame,
        replies_df: pd.DataFrame,
        quotes_df: pd.DataFrame, 
        reposts_df: pd.DataFrame,
        # interactions_df: pd.DataFrame,
        followers_df: pd.DataFrame
    ):
        self.user_id = user_id

        # Store raw data
        self.posts = posts_df
        self.replies = replies_df
        self.quotes = quotes_df
        self.reposts = reposts_df
        # self.interactions = interactions_df
        self.followers = followers_df

        # Basic behavior stats
        self.post_count = len(posts_df)
        self.reply_count = len(replies_df)
        self.quote_count = len(quotes_df)
        self.repost_count = len(reposts_df)
        # self.interaction_count = len(interactions_df)
        self.follower_count = len(followers_df)

    def summary(self):
        return {
            "user_id": self.user_id,
            "posts": self.post_count,
            "replies": self.reply_count,
            "quotes": self.quote_count,
            "reposts": self.repost_count,
            # "interactions": self.interaction_count,
            "followers": self.follower_count,
        }
