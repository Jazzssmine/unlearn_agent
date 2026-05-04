# src/real_agents/real_reflection.py

import sys
from pathlib import Path

# Add parent directory to path to allow imports from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.llm_utils import gen_completion

class ReflectionEngine:
    """
    Generate high-level insights from episodic memory.
    Similar to Stanford GenAgents 'reflection' module.
    """
    REFLECTION_PROMPT = """
You are analyzing a user's recent experiences. 
Here are some memories:

{memories}

Extract 1-2 high-level reflections summarizing what matters to the user.
Write short bullet points.
"""

    def reflect(self, memory_stream):
        recent = memory_stream.retrieve_relevant("", top_k=10)
        prompt = self.REFLECTION_PROMPT.format(
            memories="\n".join(["- " + r for r in recent])
        )
        reflections = gen_completion([{"role": "user", "content": prompt}])
        return reflections
