"""CLI: pack a JSON set of items into a budget.

    context-budget prompt.json --budget 8000 --reserve 1000
    context-budget prompt.json --budget 2000 --report
    context-budget prompt.json --budget 2000 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .items import Item
from .pack import BudgetError, Section, pack
from .tokens import Estimator

EXIT_OK = 0
EXIT_TOO_TIGHT = 1
EXIT_INVALID = 2


def _load_items(path: str) -> Dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from None

    if not isinstance(raw, dict):
        raise ValueError("input must be a JSON object with an 'items' key")
    if "items" not in raw or not isinstance(raw["items"], list):
        raise ValueError("input must have an 'items' list")
    return raw


def _build_items(raw: List[Dict[str, Any]]) -> List[Item]:
    items: List[Item] = []
    known = {
        "id", "content", "section", "priority", "recency", "relevance",
        "required", "shrinkable", "min_tokens", "kind", "tokens",
    }
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"items[{index}] must be an object")
        if "id" not in entry:
            raise ValueError(f"items[{index}] is missing 'id'")
        unknown = set(entry) - known
        if unknown:
            raise ValueError(
                f"items[{index}] has unknown field(s) {sorted(unknown)}; "
                f"known: {sorted(known)}"
            )
        items.append(Item(**entry))
    return items


def _sections(raw: Optional[List[Dict[str, Any]]]) -> Optional[Sequence[Section]]:
    if raw is None:
        return None
    return [
        Section(
            name=entry["name"],
            weight=float(entry.get("weight", 1.0)),
            order=entry.get("order", "priority"),
        )
        for entry in raw
    ]


def cmd_pack(args: argparse.Namespace) -> int:
    try:
        document = _load_items(args.input)
        items = _build_items(document["items"])
        sections = _sections(document.get("sections"))
        budget = args.budget or document.get("budget")
        if budget is None:
            raise ValueError("no budget: pass --budget or set 'budget' in the input")
        reserve = args.reserve if args.reserve is not None else document.get("reserve", 0)
        estimator = Estimator(safety=args.safety)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID

    kwargs = {"reserve": reserve, "estimator": estimator}
    if sections is not None:
        kwargs["sections"] = sections

    try:
        result = pack(items, int(budget), **kwargs)
    except BudgetError as exc:
        # This is not a crash: it is the tool refusing to silently drop an
        # instruction, and telling the caller exactly what to change.
        print(f"error: {exc}", file=sys.stderr)
        print(f"       required items: {', '.join(exc.items)}", file=sys.stderr)
        return EXIT_TOO_TIGHT
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID

    if args.json:
        print(json.dumps(
            {
                "usable": result.usable,
                "tokens_used": result.tokens_used,
                "free": result.free,
                "packed": [
                    {"id": p.id, "section": p.section, "tokens": p.tokens,
                     "shrunk": p.shrunk}
                    for p in result.packed
                ],
                "dropped": [
                    {"id": d.id, "section": d.section, "tokens": d.tokens,
                     "reason": d.reason}
                    for d in result.dropped
                ],
            },
            indent=2,
        ))
        return EXIT_OK

    if not args.quiet:
        print(result.report(), file=sys.stderr)
    if not args.report:
        print(result.text)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="context-budget",
        description="Pack context into a token budget, deterministically.",
    )
    parser.add_argument("input", help="JSON file of items")
    parser.add_argument("--budget", type=int, help="model context window in tokens")
    parser.add_argument("--reserve", type=int, help="tokens to leave for the reply")
    parser.add_argument("--safety", type=float, default=1.05,
                        help="estimate multiplier; must be >= 1.0")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--report", action="store_true", help="report only, no prompt")
    parser.add_argument("--quiet", action="store_true", help="suppress the report")
    parser.set_defaults(func=cmd_pack)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
