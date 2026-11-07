"""Deterministic packing of context items into a token budget.

Two properties matter more than cleverness here:

**It is deterministic.** The same items and budget produce a byte-identical
prompt. That is not a nicety — providers bill cached prefixes at a fraction of
the input price, and a packer that reshuffles blocks on every turn destroys the
cache it could have used. It also makes this testable, which "roughly similar
output" is not.

**It fails loudly.** Silently dropping the system prompt or the user's question
is the worst possible outcome of a tight budget, and it is the default outcome
of the naive implementation (truncate the tail). Required items either fit or
raise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .items import Item
from .shrink import shrink_to
from .tokens import Estimator

# Conservative cost of the two marker lines around each block. They are around
# 50 characters, so this is deliberately generous.
_MARKER_TOKENS = 16


class BudgetError(Exception):
    """Required items alone do not fit in the available budget.

    Never resolved by dropping them: the caller has to raise the budget or
    shorten the instructions.
    """

    def __init__(self, message: str, *, needed: int, available: int,
                 items: Sequence[str]) -> None:
        super().__init__(message)
        self.needed = needed
        self.available = available
        self.items = list(items)


@dataclass(frozen=True)
class Section:
    """Where a block belongs, how much of the budget it may claim, and how it
    is ordered inside the final prompt.

    ``order="relevance"`` renders the *least* relevant block first, so the most
    relevant one ends up immediately before the task. Models attend most
    reliably to the start and the end of a long prompt; burying the best
    document in the middle is how retrieval silently stops working.
    """

    name: str
    weight: float = 1.0
    order: str = "priority"

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"section {self.name!r}: weight must be >= 0")
        if self.order not in ("priority", "relevance", "chronological"):
            raise ValueError(
                f"section {self.name!r}: order must be 'priority', 'relevance' "
                "or 'chronological'"
            )


DEFAULT_SECTIONS: Sequence[Section] = (
    Section("system", weight=0.10),
    Section("context", weight=0.60, order="relevance"),
    # Oldest first, newest last: the most recent turn sits next to the
    # question, where attention is strongest.
    Section("history", weight=0.30, order="chronological"),
    # The task is the one thing that must never be optional, so the default
    # has no discretionary weight: a non-required item here is a mistake.
    Section("task", weight=0.00),
)


@dataclass(frozen=True)
class Packed:
    id: str
    section: str
    tokens: int
    text: str
    shrunk: bool


@dataclass(frozen=True)
class Dropped:
    id: str
    section: str
    tokens: int
    reason: str


@dataclass
class PackResult:
    text: str
    packed: List[Packed]
    dropped: List[Dropped]
    tokens_used: int
    usable: int

    @property
    def free(self) -> int:
        return self.usable - self.tokens_used

    def report(self) -> str:
        lines = [
            f"budget: {self.usable} usable, {self.tokens_used} used, {self.free} free",
            f"included: {len(self.packed)}, dropped: {len(self.dropped)}",
        ]
        for item in self.packed:
            note = " (shrunk)" if item.shrunk else ""
            lines.append(f"  + [{item.section}] {item.id} ({item.tokens} tok){note}")
        for item in self.dropped:
            lines.append(f"  - [{item.section}] {item.id} ({item.tokens} tok): {item.reason}")
        return "\n".join(lines)


def _cost(item: Item, estimator: Estimator) -> int:
    if item.tokens is not None:
        return item.tokens
    return (
        estimator.count(item.content, item.kind)
        + estimator.overhead_per_item
        + _MARKER_TOKENS
    )


def _include_key(key: str) -> Callable[[Item], tuple]:
    """Order used to decide *who gets the space*."""
    if key == "relevance":
        return lambda i: (-i.relevance, -i.priority, i.id)
    return lambda i: (-i.priority, -i.recency, i.id)


def _render_key(key: str) -> Callable[[Item], tuple]:
    """Order used to lay out the final prompt."""
    if key == "relevance":
        # Ascending: the most relevant document lands last, next to the task.
        return lambda i: (i.relevance, -i.priority, i.id)
    if key == "chronological":
        # Oldest first, newest last. Newest-first rendering would push the most
        # recent turn — usually the one the question depends on — furthest from
        # the question.
        return lambda i: (-i.priority, i.recency, i.id)
    return lambda i: (-i.priority, -i.recency, i.id)


def pack(
    items: Sequence[Item],
    budget: int,
    *,
    reserve: int = 0,
    sections: Sequence[Section] = DEFAULT_SECTIONS,
    estimator: Optional[Estimator] = None,
) -> PackResult:
    """Fit ``items`` into ``budget`` tokens, keeping ``reserve`` for the reply."""
    if budget <= 0:
        raise ValueError(f"budget must be positive (got {budget})")
    if reserve < 0:
        raise ValueError(f"reserve must be >= 0 (got {reserve})")
    usable = budget - reserve
    if usable <= 0:
        raise ValueError(
            f"budget {budget} minus reserve {reserve} leaves no room for context"
        )

    estimator = estimator or Estimator()
    specs: Dict[str, Section] = {s.name: s for s in sections}
    order = [s.name for s in sections]

    by_id: Dict[str, Item] = {}
    cost: Dict[str, int] = {}
    for item in items:
        if item.id in by_id:
            raise ValueError(f"duplicate item id {item.id!r}")
        if item.section not in specs:
            raise ValueError(
                f"item {item.id!r} uses unknown section {item.section!r}; known sections: {order}"
            )
        by_id[item.id] = item
        cost[item.id] = _cost(item, estimator)

    required = [i for i in items if i.required]
    discretionary = [i for i in items if not i.required]

    for item in discretionary:
        if specs[item.section].weight <= 0:
            raise ValueError(
                f"section {item.section!r} has weight 0, so item {item.id!r} could never "
                "be packed; mark the item required or give the section a weight"
            )

    required_cost = sum(cost[i.id] for i in required)
    if required_cost > usable:
        raise BudgetError(
            f"required items need {required_cost} tokens but only {usable} are available "
            f"(budget {budget}, reserve {reserve}); raise the budget or shorten: "
            + ", ".join(sorted(i.id for i in required)),
            needed=required_cost,
            available=usable,
            items=sorted(i.id for i in required),
        )

    packed: List[Packed] = [Packed(i.id, i.section, cost[i.id], i.content, False) for i in required]
    placed = {i.id for i in required}
    dropped: List[Dropped] = []
    used = required_cost

    # Per-section caps, from the weights of the sections that actually have
    # discretionary candidates.
    present = [s for s in sections if any(i.section == s.name for i in discretionary)]
    weight_sum = sum(s.weight for s in present) or 1.0
    remaining_total = usable - required_cost
    caps = {
        s.name: int(math.floor(remaining_total * s.weight / weight_sum))
        for s in present
    }

    for section in present:
        room = caps[section.name]
        candidates = sorted(
            (i for i in discretionary if i.section == section.name),
            key=_include_key(section.order),
        )
        for item in candidates:
            need = cost[item.id]
            if need <= room:
                packed.append(Packed(item.id, section.name, need, item.content, False))
                placed.add(item.id)
                room -= need
                used += need
                continue
            if item.shrinkable and item.tokens is None:
                # Shrinking is only honest when we measure the result the same
                # way we measured the item. A caller-supplied exact count is in
                # their units, not ours, so such an item is packed or dropped
                # whole rather than silently re-measured in different units.
                block = room - estimator.overhead_per_item - _MARKER_TOKENS
                if block >= item.min_tokens:
                    text, _ = shrink_to(item.content, block, estimator, item.kind)
                    if text:
                        shrunk = (
                            estimator.count(text, item.kind)
                            + estimator.overhead_per_item
                            + _MARKER_TOKENS
                        )
                        if shrunk <= room:
                            packed.append(Packed(item.id, section.name, shrunk, text, True))
                            placed.add(item.id)
                            room -= shrunk
                            used += shrunk
                            continue
            droppable = item.shrinkable and item.tokens is None
            dropped.append(
                Dropped(
                    item.id,
                    section.name,
                    need,
                    f"did not fit in {max(room, 0)} remaining tokens"
                    + (" and could not be shrunk usefully" if droppable else ""),
                )
            )

    # A section that cannot use its share must not strand it. Anything that
    # still fits in the *global* remainder is placed, in section order, until a
    # full pass adds nothing.
    progress = True
    while progress:
        progress = False
        for item in discretionary:
            if item.id in placed or used + cost[item.id] > usable:
                continue
            packed.append(Packed(item.id, item.section, cost[item.id], item.content, False))
            placed.add(item.id)
            used += cost[item.id]
            progress = True

    # And if room appeared after shrinking, give a shrunk block its text back.
    for index, block in enumerate(list(packed)):
        if not block.shrunk:
            continue
        item = by_id[block.id]
        full = cost[item.id]
        if used - block.tokens + full <= usable:
            packed[index] = Packed(item.id, block.section, full, item.content, False)
            used = used - block.tokens + full

    dropped = [d for d in dropped if d.id not in placed]

    rendered = _render(packed, sections, by_id)
    tokens_used = sum(p.tokens for p in packed)
    if tokens_used > usable:  # pragma: no cover - guarded by construction
        raise AssertionError(
            f"packer exceeded its own budget: {tokens_used} > {usable}"
        )

    return PackResult(
        text=rendered,
        packed=_ordered(packed, sections, by_id),
        dropped=dropped,
        tokens_used=tokens_used,
        usable=usable,
    )


def _ordered(packed: Sequence[Packed], sections: Sequence[Section],
             by_id: Dict[str, Item]) -> List[Packed]:
    result: List[Packed] = []
    for section in sections:
        group = [p for p in packed if p.section == section.name]
        group.sort(key=lambda p: _render_key(section.order)(by_id[p.id]))
        result.extend(group)
    return result


def _render(packed: Sequence[Packed], sections: Sequence[Section],
            by_id: Dict[str, Item]) -> str:
    blocks = []
    for block in _ordered(packed, sections, by_id):
        blocks.append(
            f"<<<{block.section}:{block.id} tokens={block.tokens}>>>\n"
            f"{block.text}\n"
            f"<<<end {block.section}:{block.id}>>>"
        )
    return "\n\n".join(blocks)
