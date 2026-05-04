# src/real_agents/real_memory.py

import datetime
from typing import List, Dict

class MemoryItem:
    def __init__(self, content: str, importance: float, timestamp=None):
        self.content = content
        self.importance = importance
        self.timestamp = timestamp or datetime.datetime.utcnow()

class MemoryStream:
    """
    Stanford Generative Agents style memory system:
      - episodic memory (events)
      - semantic memory (facts)
    """
    def __init__(self, max_size=2000):
        self.episodic: List[MemoryItem] = []
        self.semantic: List[MemoryItem] = []
        self.max_size = max_size

    def add_memory(self, content: str, importance: float = 1.0, semantic=False):
        item = MemoryItem(content, importance)
        if semantic:
            self.semantic.append(item)
        else:
            self.episodic.append(item)

        # Trim long memory stream
        if len(self.episodic) > self.max_size:
            self.episodic.pop(0)

    def retrieve_relevant(self, query: str, top_k=5) -> List[str]:
        """
        Retrieves memories with highest lexical overlap.
        You can improve this to use embeddings later.
        """
        scored = []
        for mem in self.episodic + self.semantic:
            score = sum([1 for w in query.split() if w.lower() in mem.content.lower()])
            scored.append((score + mem.importance, mem))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [m.content for _, m in scored[:top_k]]
