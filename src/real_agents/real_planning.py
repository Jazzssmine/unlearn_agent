# src/real_agents/real_planning.py

import sys
from pathlib import Path

# Add parent directory to path to allow imports from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.llm_utils import gen_completion

class PlanningEngine:
    """
    Creates short-term & long-term plans.
    """
    PLAN_PROMPT = """
Given the user's reflections:
{reflections}

Generate:
1. Long-term goals
2. Short-term action plan (very concrete)
"""

    def create_plan(self, reflections):
        prompt = self.PLAN_PROMPT.format(reflections=reflections)
        plan = gen_completion([{"role": "user", "content": prompt}])
        return plan
