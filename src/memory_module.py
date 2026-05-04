from typing import Any, Dict, List

from utils.llm_utils import gen_completion


class MemoryModule:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.memory_state = ""
        self.model = model
        self.memory_history: List[Dict[str, Any]] = []

    def llm_call(self, messages: List[Dict[str, str]], max_tokens: int = 150) -> str:
        return (
            gen_completion(
                messages,
                model=self.model,
                temperature=0.0,
                max_tokens=max_tokens,
                max_retries=3,
                retry_delay=2.0,
            )
            .strip()
        )

    def _clean_summary_response(self, raw_response: str) -> str:
        text = (raw_response or "").strip()
        lowered = text.lower()

        # If model echoed prompt, keep only content after the final summary marker.
        last_idx = -1
        for marker in ("updated summary:", "summary:"):
            idx = lowered.rfind(marker)
            if idx > last_idx:
                last_idx = idx
                marker_len = len(marker)
        if last_idx >= 0:
            text = text[last_idx + marker_len :].strip()
            lowered = text.lower()

        # Strip common continuation prefixes often produced by instruction-tuned models.
        cut_points = []
        for marker in ("let's try", "here is", "note:"):
            idx = lowered.find(marker)
            if idx > 0:
                cut_points.append(idx)
        if cut_points:
            text = text[: min(cut_points)].strip()

        if len(text) > 500:
            text = text[:500].rstrip()

        return text

    def initialize(self, seed_post: str) -> str:
        """Initialize memory from the seed post."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise summarizer. Always respond with ONLY "
                    "a 1-2 sentence summary. Do not generate examples, "
                    "continuations, or additional text."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize this social media post. Capture topic and tone.\n"
                    f"Post: {seed_post}"
                ),
            },
        ]
        self.memory_state = self._clean_summary_response(self.llm_call(messages, max_tokens=150))
        self.memory_history.append(
            {
                "turn": 0,
                "trigger_message": seed_post,
                "memory_after": self.memory_state,
                "event": "initialize",
            }
        )
        return self.memory_state

    def update(self, new_message: str, turn: int) -> str:
        """Update memory after observing a new message in the thread."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise summarizer. Always respond with ONLY "
                    "a 2-3 sentence updated summary. Do not generate examples, "
                    "continuations, or additional text. Do not repeat the prompt."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current conversation summary: {self.memory_state}\n\n"
                    f"New message in the discussion: {new_message}\n\n"
                    "Write an updated 2-3 sentence summary:"
                ),
            },
        ]
        self.memory_state = self._clean_summary_response(self.llm_call(messages, max_tokens=150))
        self.memory_history.append(
            {
                "turn": turn,
                "trigger_message": new_message,
                "memory_after": self.memory_state,
                "event": "update",
            }
        )
        return self.memory_state

    def set_state(self, new_state: str, turn: int, trigger_message: str, event: str = "set_state") -> str:
        self.memory_state = (new_state or "").strip()
        self.memory_history.append(
            {
                "turn": turn,
                "trigger_message": trigger_message,
                "memory_after": self.memory_state,
                "event": event,
            }
        )
        return self.memory_state

    def log_no_update(self, turn: int, trigger_message: str, reason: str = "skipped_update") -> None:
        self.memory_history.append(
            {
                "turn": turn,
                "trigger_message": trigger_message,
                "memory_after": self.memory_state,
                "event": reason,
            }
        )

    def get_state(self) -> str:
        return self.memory_state

    def get_history(self) -> List[Dict[str, Any]]:
        return self.memory_history

    def reset(self):
        self.memory_state = ""
        self.memory_history = []
