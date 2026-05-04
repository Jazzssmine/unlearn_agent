# src/real_agents/real_agent.py

import sys
from pathlib import Path
import networkx as nx  

# Add parent directory to path to allow imports from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.llm_utils import gen_completion, parse_json
from src.real_agents.real_memory import MemoryStream
from src.real_agents.real_reflection import ReflectionEngine
from src.real_agents.real_planning import PlanningEngine

class RealAgent:
    def __init__(self, profile, network: nx.DiGraph, model: str = "gpt-4o-mini"):
        self.profile = profile
        self.memory = MemoryStream()
        self.network = network
        self.reflection_engine = ReflectionEngine()
        self.planning_engine = PlanningEngine()
        self.current_plan = None
        self.model = model
        
    def observe(self, item):
        # Ensure content is always treated as a string to avoid errors
        content = item.get("content", "")
        content_str = str(content)
        author_id = item.get("author_id", "unknown")
        obs = f"User saw a {item.get('type', 'unknown')} from {author_id}: {content_str[:200]}"
        self.memory.add_memory(obs, importance=1.0)

    def reflect_if_needed(self):
        if len(self.memory.episodic) % 10 == 0:
            reflections = self.reflection_engine.reflect(self.memory)
            self.memory.add_memory("REFLECTION: " + reflections, importance=5.0, semantic=True)
            self.current_plan = self.planning_engine.create_plan(reflections)

    def get_relationship_context(self, target_user_id):
        """Check the graph to see if we follow the target user."""
        if self.network.has_edge(self.profile.user_id, target_user_id):
            return "You CURRENTLY FOLLOW this user."
        else:
            return "You DO NOT follow this user."

    def act(self, item):
        """
        Decide how the simulated user should react to a single feed item.

        The model is asked to:
        1. Reason about the current content.
        2. Connect it to the user's prior behavior.
        3. Choose an action.
        4. Optionally generate text (for reply/quote/repost).

        Returns a structured dict with keys:
          - reasoning_current_tweet
          - reasoning_prior_behavior
          - action              # one of: reply, quote, repost, like, ignore
          - generated_text      # non-empty only for reply/quote/repost
        """
        history_texts = self.profile.posts["text"].head(10).tolist()
        content = str(item.get("content", ""))
        author_id = str(item.get("author_id", ""))
        # Get graph info
        relationship_status = self.get_relationship_context(author_id)

        prompt = f"""
You are simulating user {self.profile.user_id}.

Here is a sample of their recent post history (what they tend to say and how they sound):
{history_texts}

You are observing a piece of content on your feed:
User ID: {author_id}
Content: "{content}"
Relationship Status: {relationship_status}

Your task is to make two independent decisions based on this observation:

DECISION 1: Content Interaction
Decide exactly ONE content action from this set:
- "reply" (engage with text)
- "quote" (share with comment)
- "repost" (share without comment)
- "like" (acknowledge)
- "ignore" (do nothing with the content)

DECISION 2: Relationship Update
Decide exactly ONE relationship action for the author ({author_id}) from this set:
- "follow" (start following this user)
- "unfollow" (stop following this user)

If the content action is "reply" or "quote", write the text.

Respond ONLY with a single JSON object:
{{
"reasoning_content": "<reasoning for content action, based on content and relationship>",
"reasoning_relationship": "<reasoning for follow/unfollow action, based on content and relationship>",
"reasoning_prior_behavior": "<how this relates to the user's past behavior>",
"content_action": "<reply|quote|repost|like|ignore>",
"relationship_action": "<follow|unfollow>",
"generated_text": "<text if reply/quote, else empty string>"
}}
"""
        raw = gen_completion([{"role": "user", "content": prompt}], model=self.model)

        # Parse into a structured dict so downstream code can rely on keys.
        parsed = parse_json(
            raw,
            target_keys=[
                "reasoning_content",
                "reasoning_relationship",
                "reasoning_prior_behavior",
                "content_action",
                "relationship_action",
                "generated_text",
            ],
        )
        # Defaults
        for key in ["reasoning_content", "reasoning_relationship", "reasoning_prior_behavior", "content_action", "relationship_action", "generated_text"]:
            parsed.setdefault(key, "")
        return parsed

    def step(self, item):
        self.observe(item)
        self.reflect_if_needed()
        action_struct = self.act(item)
        content_action = action_struct.get('content_action', '').lower()
        relationship_action = action_struct.get('relationship_action', '').lower()
        author_id = str(item.get("author_id", ""))

        # --- Execute Graph Updates (Relationship Action) ---
        if relationship_action == "follow":
            if not self.network.has_edge(self.profile.user_id, author_id):
                self.network.add_edge(self.profile.user_id, author_id, edge_type="follows")
                print(f"  [GRAPH UPDATE] User {self.profile.user_id} followed {author_id}")
        
        elif relationship_action == "unfollow":
            if self.network.has_edge(self.profile.user_id, author_id):
                try:
                    self.network.remove_edge(self.profile.user_id, author_id)
                    print(f"  [GRAPH UPDATE] User {self.profile.user_id} unfollowed {author_id}")
                except nx.NetworkXError:
                    # Handle case where edge might have already been removed in a multi-agent scenario
                    pass

        # Store a compact summary of the chosen action plus a bit of reasoning.
        summary = (
            f"CONTENT ACTION: {content_action} | "
            f"RELATIONSHIP ACTION: {relationship_action} | "
            f"REASON: {action_struct.get('reasoning_content', '')[:100]} | "
            f"PRIOR BEHAVIOR: {action_struct.get('reasoning_prior_behavior', '')[:100]}"
        )
        self.memory.add_memory(summary, importance=2.0)
        return action_struct

    def react_to_thread(self, thread_history: list, force_action=None):
        """
        Analyze a conversation thread and decide on an intervention.
        
        Args:
            thread_history: List of dicts [{'author_id': str, 'content': str}]
            force_action: If set to "reply", forces the agent to generate text.
                          If None, the agent decides between 'reply' or 'ignore'.
        """
        # 1. Format the thread for the LLM
        thread_text = ""
        for i, msg in enumerate(thread_history):
            author_id = str(msg.get("author_id", "unknown"))
            # Support both history schemas: {"content": ...} and {"text": ...}.
            content = msg.get("content")
            if content is None:
                content = msg.get("text", "")
            thread_text += f"[{i+1}] User {author_id}: {content}\n"

        # Use recent posts to define the persona/style
        style_context = self.profile.posts["text"].head(20).tolist()
        
        # Adjust prompt based on whether we are forcing an action
        task_instruction = ""
        if force_action == "reply":
            task_instruction = """
                1. You MUST join this conversation.
                2. Write the text you would post to contribute to this thread.
                """
        else:
            task_instruction = """
                1. Decide if you would strictly STAY SILENT ("ignore") or JOIN IN ("reply").
                - Only "reply" if you feel strongly compelled.
                2. If you decide to "reply", write the text.
                """

        prompt = f"""
You are simulating user {self.profile.user_id}.

Your personality and speech style are defined by your past posts:
{style_context}

Additional behavior instruction:
{getattr(self.profile, "behavior_instruction", "None")}

You are reading this conversation thread:
{thread_text}

Your task:
1. Analyze the conversation context.
{task_instruction}

Respond ONLY with a single JSON object:
{{
"reasoning": "<why you chose to reply or ignore, and why you wrote what you wrote>",
"action": "{ 'reply' if force_action == 'reply' else '<reply|ignore>' }",
"generated_text": "<your response text if replying, else empty string>"
}}
"""
        raw = gen_completion([{"role": "user", "content": prompt}], model=self.model)

        parsed = parse_json(raw, target_keys=["reasoning", "action", "generated_text"])

        # Enforce defaults and overrides
        if force_action:
            parsed["action"] = force_action

        parsed.setdefault("action", "ignore")
        parsed.setdefault("generated_text", "")

        return parsed

    def build_react_to_thread_messages(self, thread_history: list, force_action=None) -> list:
        """Return the messages list that react_to_thread would pass to gen_completion."""
        thread_text = ""
        for i, msg in enumerate(thread_history):
            author_id = str(msg.get("author_id", "unknown"))
            content = msg.get("content")
            if content is None:
                content = msg.get("text", "")
            thread_text += f"[{i+1}] User {author_id}: {content}\n"

        style_context = self.profile.posts["text"].head(20).tolist()

        if force_action == "reply":
            task_instruction = """
                1. You MUST join this conversation.
                2. Write the text you would post to contribute to this thread.
                """
        else:
            task_instruction = """
                1. Decide if you would strictly STAY SILENT ("ignore") or JOIN IN ("reply").
                - Only "reply" if you feel strongly compelled.
                2. If you decide to "reply", write the text.
                """

        prompt = f"""
You are simulating user {self.profile.user_id}.

Your personality and speech style are defined by your past posts:
{style_context}

Additional behavior instruction:
{getattr(self.profile, "behavior_instruction", "None")}

You are reading this conversation thread:
{thread_text}

Your task:
1. Analyze the conversation context.
{task_instruction}

Respond ONLY with a single JSON object:
{{
"reasoning": "<why you chose to reply or ignore, and why you wrote what you wrote>",
"action": "{ 'reply' if force_action == 'reply' else '<reply|ignore>' }",
"generated_text": "<your response text if replying, else empty string>"
}}
"""
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def parse_react_to_thread_response(raw: str, force_action=None) -> dict:
        """Parse a raw LLM response from react_to_thread into a decision dict."""
        parsed = parse_json(raw, target_keys=["reasoning", "action", "generated_text"])
        if force_action:
            parsed["action"] = force_action
        parsed.setdefault("action", "ignore")
        parsed.setdefault("generated_text", "")
        return parsed