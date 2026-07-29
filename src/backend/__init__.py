"""backend -- the model-scoring abstraction. `Backend` is the protocol the referee scores
through; `MockBackend` is the deterministic Mac stand-in; `HFBackend` is the cluster-fenced real
model (torch/transformers imported lazily inside `load()`, so this import is safe on the Mac)."""
from .base import Backend, Box
from .hf import HFBackend
from .mock import MockBackend

__all__ = ["Backend", "Box", "MockBackend", "HFBackend"]
