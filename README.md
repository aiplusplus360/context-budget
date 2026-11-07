# context-budget

**Pack a conversation into a context window without losing the things that
matter — deterministically.**

[![CI](https://github.com/aiplusplus360/context-budget/actions/workflows/ci.yml/badge.svg)](https://github.com/aiplusplus360/context-budget/actions/workflows/ci.yml)

```
$ context-budget examples/refund.json --budget 420 --reserve 120 --report
budget: 300 usable, 271 used, 29 free
included: 5, dropped: 2
  + [system] rules (66 tok)
  + [context] metering-guide (49 tok) (shrunk)
  + [context] refund-policy (82 tok)
  + [history] turn-2 (37 tok)
  + [task] question (37 tok)
  - [context] style-guide (50 tok): did not fit in 0 remaining tokens and could not be shrunk usefully
  - [history] turn-1 (48 tok): did not fit in 28 remaining tokens and could not be shrunk usefully
```

Zero dependencies. Python 3.9+. One file of pure logic per concern.

---

## The problem

Every LLM application eventually writes this:

```python
prompt = (system + "\n\n" + docs[:8000] + "\n\n" + history[-10:] + "\n\n" + question)
```

It works until it doesn't, and the failure is silent in three different ways:

1. **The budget is in characters, the limit is in tokens.** The prompt goes over
   and the provider trims the *end* — which is where the question was.
2. **`docs[:8000]` cuts mid-sentence** and keeps the introduction of a document
   while discarding the part that answers the question.
3. **Nothing records what was dropped**, so when answer quality falls nobody can
   tell whether retrieval got worse or the packer silently ate the context.

`context-budget` makes all three explicit and testable.

---

## Install

```bash
pip install context-budget          # or: pip install -e .
```

## Use it

```python
from contextbudget import Item, pack

result = pack(
    [
        Item("rules", SYSTEM_PROMPT, section="system", required=True),
        Item("doc-1", chunk, section="context", relevance=0.91),
        Item("doc-2", other_chunk, section="context", relevance=0.42),
        Item("turn-1", "Customer: charged twice.", section="history", recency=1),
        Item("question", user_text, section="task", required=True),
    ],
    budget=8192,
    reserve=1024,          # keep room for the reply
)

provider.complete(result.text)
print(result.report())     # what was included, what was dropped, and why
```

### CLI

```bash
context-budget prompt.json --budget 8192 --reserve 1024   # prompt to stdout
context-budget prompt.json --budget 2000 --report         # report only
context-budget prompt.json --json                         # machine readable
```

```json
{
  "budget": 8192,
  "reserve": 1024,
  "items": [
    { "id": "rules", "section": "system", "required": true, "content": "..." },
    { "id": "doc-1", "section": "context", "relevance": 0.91, "content": "..." }
  ]
}
```

Exit codes: `0` packed, `1` required items do not fit (with the reason), `2` bad
input. A CI step can tell "the budget is too tight" apart from "the tool broke".

---

## The four decisions that make this different

### 1. Required items are a hard constraint, not a preference

`required=True` means *this must be in the prompt*. If required items alone
exceed the budget, packing raises `BudgetError` naming them:

```
required items need 4120 tokens but only 1024 are available
(budget 2048, reserve 1024); raise the budget or shorten: rules, question
```

The naive implementation truncates the tail and deletes the question. That is
the single worst failure mode in this problem, and it should be a loud error.

### 2. Output is byte-identical for identical input

Providers bill cached prefixes at a fraction of the input price. A packer that
reshuffles blocks between turns destroys the cache it could have used — and
makes the whole system untestable.

Two consequences, both tested:

* packing the same items twice produces the same string;
* the rendered block for an item never changes because *other* items were added
  or removed, so earlier blocks stay cacheable.

### 3. Placement follows where attention actually is

Models attend most reliably to the beginning and end of a long prompt.

| Section | Rendered | Why |
|---|---|---|
| `system` | first | Instructions frame everything after them |
| `context` | ascending relevance | The best document ends up **last**, next to the question |
| `history` | chronological | The newest turn sits closest to the question |
| `task` | last | The question is the last thing read |

Which is also why **inclusion** order is the opposite of **render** order: the
most relevant document wins the space *and* lands nearest the question.

### 4. Tokens are estimated conservatively, on purpose

An exact count needs the model's vocabulary. A packer needs a bound it can
trust, so:

* characters-per-token is per content type — prose 4.2, code 3.2, JSON 3.0;
* every count rounds up, with a 1.05 safety multiplier by default;
* each block carries a fixed overhead for its markers.

Over-estimating wastes a few percent of the window. Under-estimating overflows
it, and the provider's response is to trim the end of the prompt. The asymmetry
is not close, so `safety < 1.0` is rejected outright.

---

## What happens when something does not fit

In order, per section:

1. **Fits whole** → included, unchanged.
2. **Does not fit, but is shrinkable** → truncated *middle-out*: the head and
   tail survive, the gap is marked with `…[N characters omitted]…`. A document
   that matters at the bottom (a definition, a conclusion, an error) keeps it.
3. **Too small to be worth it** (below `min_tokens`, default 24) → dropped, with
   the reason recorded.
4. **Not shrinkable** → dropped whole, with the reason recorded.

Then, because a section that cannot use its share must not strand it:

5. Leftover budget is **redistributed** to anything still waiting, in section
   order, until a full pass adds nothing.
6. If room appeared, any block that was shrunk **regains its original text**.

An item carrying a caller-supplied exact `tokens=` count is never shrunk: the
number is in the caller's units, and re-measuring the result in different units
would be a silent lie. It is packed or dropped whole.

---

## Design notes

- **Sections carry weights, not fixed sizes.** A weight is a share of what is
  left after required items, recomputed from the sections that actually have
  candidates. Small budgets therefore do not starve a section that is empty.
- **A discretionary item in a zero-weight section is an error.** The default
  `task` section has weight 0 because a non-required question is almost always a
  bug, and it is one that would only show up under load.
- **Malformed input fails at construction.** Empty ids, duplicate ids, unknown
  sections, unknown fields and negative counts are rejected with the item named.
- **`--report` is not a debugging aid.** Knowing what was dropped is part of the
  output; a packer you cannot audit is a packer you cannot tune.

### What this deliberately does not do

- **No tokenizer.** Exact counts are the caller's problem, via `tokens=`.
- **No ranking.** It packs what you give it in the order you rank it. Retrieval
  quality is a separate problem — see [`rag-eval`](https://github.com/aiplusplus360/rag-eval).
- **No summarisation.** Shrinking is lossy and says so. If you need a compressed
  document, compress it before packing.

---

## Development

```bash
python -m unittest discover -s tests -v
```

59 tests, no third-party runtime dependencies.

```
contextbudget/
  tokens.py    conservative estimation, per content type
  items.py     the unit being packed
  shrink.py    middle-out truncation
  pack.py      allocation, placement, redistribution, reporting
  cli.py       JSON in, prompt out
```

## License

MIT
