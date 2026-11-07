"""Pack a conversation into a context window without losing what matters."""

from .items import Item
from .pack import (
    DEFAULT_SECTIONS,
    BudgetError,
    Dropped,
    Packed,
    PackResult,
    Section,
    pack,
)
from .shrink import shrink_to
from .tokens import Estimator

__version__ = "0.1.0"

__all__ = [
    "Item",
    "Section",
    "DEFAULT_SECTIONS",
    "PackResult",
    "Packed",
    "Dropped",
    "BudgetError",
    "Estimator",
    "pack",
    "shrink_to",
    "__version__",
]
