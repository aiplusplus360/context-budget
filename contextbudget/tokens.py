"""Token estimation without a tokenizer.

An exact count needs the model's vocabulary, which means shipping (or calling)
a tokenizer. A packer does not need an exact count; it needs a *bound it can
trust*, so every knob here rounds up.

The asymmetry is not close:

* over-estimating wastes a few percent of the window;
* under-estimating overflows it, and providers respond by rejecting the request
  or silently trimming it — and the part they trim is the end, which is where
  the actual question lives.
"""

from __future__ import annotations

import math
from typing import Dict

# Rough characters-per-token by content type. Code and JSON are denser than
# prose because identifiers, punctuation and keys split more finely.
_KIND_RATIOS: Dict[str, float] = {
    "prose": 4.2,
    "code": 3.2,
    "json": 3.0,
    "table": 3.5,
}

DEFAULT_KIND = "prose"


class Estimator:
    """A deliberately conservative estimate of how many tokens text costs."""

    def __init__(self, *, safety: float = 1.05, overhead_per_item: int = 6) -> None:
        if safety < 1.0:
            raise ValueError(
                f"safety must be >= 1.0 (got {safety}); below 1.0 the estimate can "
                "under-count and the prompt can overflow the window"
            )
        if overhead_per_item < 0:
            raise ValueError("overhead_per_item must be >= 0")
        self.safety = safety
        self.overhead_per_item = overhead_per_item

    @staticmethod
    def ratio(kind: str) -> float:
        try:
            return _KIND_RATIOS[kind]
        except KeyError:
            raise ValueError(
                f"unknown content kind {kind!r}; expected one of {sorted(_KIND_RATIOS)}"
            ) from None

    def count(self, text: str, kind: str = DEFAULT_KIND) -> int:
        """Tokens for ``text`` alone, rounded up."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.ratio(kind) * self.safety))

    def item_tokens(self, text: str, kind: str = DEFAULT_KIND) -> int:
        """Tokens for one packed block: the text plus its role overhead."""
        return self.count(text, kind) + self.overhead_per_item
