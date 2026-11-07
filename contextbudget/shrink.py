"""Middle-out truncation.

Cutting a document to its first N characters is the common approach and the
worst one: for a file that matters because of what is at the bottom — a
function definition, a conclusion, an error — the head is often the least
useful part.

Keeping the head *and* the tail and marking the gap keeps both the framing and
the payload, and tells the model that something was removed rather than leaving
it to infer that the document simply ends there.
"""

from __future__ import annotations

from typing import Tuple

from .tokens import DEFAULT_KIND, Estimator

MARKER = "\n…[{omitted} characters omitted]…\n"

# How the surviving budget is split between the head and the tail.
HEAD_RATIO = 0.6


def _join(content: str, head: int, tail: int) -> str:
    omitted = len(content) - head - tail
    if omitted <= 0:
        return content
    return content[:head] + MARKER.format(omitted=omitted) + content[len(content) - tail:]


def shrink_to(
    content: str,
    max_tokens: int,
    estimator: Estimator,
    kind: str = DEFAULT_KIND,
    *,
    head_ratio: float = HEAD_RATIO,
) -> Tuple[str, int]:
    """Fit ``content`` into ``max_tokens``, keeping its head and tail.

    Returns ``(text, omitted_characters)``. ``text`` is empty when even the
    omission marker does not fit, which tells the caller to drop the item
    rather than emit a useless fragment.
    """
    if max_tokens <= 0:
        return "", len(content)
    if estimator.count(content, kind) <= max_tokens:
        return content, 0

    if not 0.0 <= head_ratio <= 1.0:
        raise ValueError("head_ratio must be between 0 and 1")

    # Largest number of original characters that still fits once the marker is
    # inserted. Binary search because the marker's own length depends on how
    # much was removed, and both grow monotonically with the split.
    low, high = 0, len(content)
    best = ""
    while low <= high:
        keep = (low + high) // 2
        head = int(keep * head_ratio)
        candidate = _join(content, head, keep - head)
        if estimator.count(candidate, kind) <= max_tokens:
            best = candidate
            low = keep + 1
        else:
            high = keep - 1

    return best, len(content) - _kept(best)


def _kept(text: str) -> int:
    """Characters of the original document that survived in ``text``.

    The marker begins and ends with a newline, so those must be excluded from
    the partition or the reported gap is off by two.
    """
    if "…[" not in text:
        return len(text)
    head, _, rest = text.partition("\n…[")
    tail = rest.partition("]…\n")[2]
    return len(head) + len(tail)
