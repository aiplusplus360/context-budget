"""The unit the packer moves around: one block of context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Item:
    """One candidate block of context.

    ``required`` is the important field. A required item is a hard constraint:
    system instructions and the user's actual question are not things a packer
    is allowed to quietly drop when the window gets tight. If required items do
    not fit, packing fails loudly instead.
    """

    id: str
    content: str
    section: str = "context"
    # Higher priority wins when space is short, within a section.
    priority: int = 0
    # Higher is newer. Breaks ties after priority, so recent turns beat stale
    # ones at the same priority.
    recency: float = 0.0
    # Retrieval only: higher is closer to the query. Used for placement, so the
    # strongest document ends up nearest the question.
    relevance: float = 0.0
    required: bool = False
    shrinkable: bool = True
    # Below this many tokens a fragment is not worth including at all.
    min_tokens: int = 24
    kind: str = "prose"
    # Pre-computed token count, if the caller already has an exact one. When
    # set, the item is packed or dropped whole: shrinking it would mean
    # measuring the result in different units than the ones supplied.
    tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("item id must not be empty")
        if self.tokens is not None and self.tokens < 0:
            raise ValueError(f"item {self.id!r}: tokens must be >= 0")
        if self.min_tokens < 0:
            raise ValueError(f"item {self.id!r}: min_tokens must be >= 0")
