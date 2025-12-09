from memory.base_memory import BaseMemory, MemoryState
from memory.no_memory import NoMemory
from memory.full_history import FullHistoryMemory
from memory.summary_memory import SummaryMemory
from memory.structured_memory import StructuredMemory
from memory.hybrid_memory import HybridMemory

MEMORY_REGISTRY: dict[str, type[BaseMemory]] = {
    "none": NoMemory,
    "full_history": FullHistoryMemory,
    "summary": SummaryMemory,
    "structured": StructuredMemory,
    "hybrid": HybridMemory,
}


def create_memory(memory_type: str, **kwargs) -> BaseMemory:
    """Factory function to create memory instances."""
    if memory_type not in MEMORY_REGISTRY:
        raise ValueError(f"Unknown memory type: {memory_type}. Available: {list(MEMORY_REGISTRY.keys())}")
    return MEMORY_REGISTRY[memory_type](**kwargs)


__all__ = [
    "BaseMemory",
    "MemoryState",
    "NoMemory",
    "FullHistoryMemory",
    "SummaryMemory",
    "StructuredMemory",
    "HybridMemory",
    "MEMORY_REGISTRY",
    "create_memory",
]

