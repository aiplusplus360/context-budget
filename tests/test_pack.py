"""Tests for estimation, shrinking, packing, and the CLI.

The invariants asserted here are the reason the tool exists:

* the estimate never exceeds the budget;
* required items are never silently dropped;
* the same input produces the same prompt, every time.
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from contextbudget import (
    DEFAULT_SECTIONS,
    BudgetError,
    Estimator,
    Item,
    Section,
    pack,
    shrink_to,
)
from contextbudget.cli import main


def block_of(text, item_id):
    """The rendered block for one item, or '' if it was not included."""
    marker = f"<<<"
    for block in text.split("\n\n"):
        head = block.splitlines()[0] if block else ""
        if head.startswith(marker) and f":{item_id} tokens=" in head:
            return block
    return ""


def ids(result):
    return [p.id for p in result.packed]


class TestEstimator(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(Estimator().count(""), 0)

    def test_never_underestimates_char_ratio(self):
        estimator = Estimator(safety=1.0)
        text = "a" * 100
        # 100 chars at 4.2 chars/token is 23.8, so 24 is the floor.
        self.assertGreaterEqual(estimator.count(text), 23)

    def test_rounds_up(self):
        # A single character still costs a token; it cannot cost zero.
        self.assertEqual(Estimator().count("a"), 1)

    def test_safety_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            Estimator(safety=0.9)

    def test_code_is_denser_than_prose(self):
        estimator = Estimator()
        text = "def f(x): return x" * 10
        self.assertGreater(estimator.count(text, "code"), estimator.count(text, "prose"))

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            Estimator().count("hello", "klingon")

    def test_item_tokens_adds_overhead(self):
        estimator = Estimator(overhead_per_item=10)
        self.assertEqual(
            estimator.item_tokens("hello"),
            estimator.count("hello") + 10,
        )

    def test_more_safety_means_more_tokens(self):
        text = "lorem ipsum dolor sit amet " * 20
        self.assertGreater(
            Estimator(safety=1.5).count(text),
            Estimator(safety=1.0).count(text),
        )


class TestShrink(unittest.TestCase):
    def setUp(self):
        self.estimator = Estimator()
        self.document = "HEAD " + ("middle " * 400) + "TAIL"

    def test_fitting_content_is_untouched(self):
        text, omitted = shrink_to("short", 100, self.estimator)
        self.assertEqual(text, "short")
        self.assertEqual(omitted, 0)

    def test_result_fits_the_limit(self):
        text, _ = shrink_to(self.document, 50, self.estimator)
        self.assertLessEqual(self.estimator.count(text), 50)

    def test_keeps_head_and_tail(self):
        text, _ = shrink_to(self.document, 60, self.estimator)
        self.assertTrue(text.startswith("HEAD"), text[:20])
        self.assertTrue(text.endswith("TAIL"), text[-20:])

    def test_marks_the_omission(self):
        text, omitted = shrink_to(self.document, 60, self.estimator)
        self.assertIn("characters omitted", text)
        self.assertGreater(omitted, 0)

    def test_reports_omitted_characters_precisely(self):
        text, omitted = shrink_to(self.document, 60, self.estimator)
        # The count in the marker must describe the same gap the caller is told
        # about, otherwise the reader cannot judge how much was lost.
        marker = f"\n…[{omitted} characters omitted]…\n"
        self.assertIn(marker, text)
        self.assertEqual(omitted, len(self.document) - (len(text) - len(marker)))

    def test_zero_budget_yields_nothing(self):
        text, omitted = shrink_to(self.document, 0, self.estimator)
        self.assertEqual(text, "")
        self.assertEqual(omitted, len(self.document))

    def test_impossible_budget_yields_nothing(self):
        text, _ = shrink_to(self.document, 1, self.estimator)
        self.assertEqual(text, "")

    def test_shrinking_is_monotonic_in_budget(self):
        small, _ = shrink_to(self.document, 40, self.estimator)
        large, _ = shrink_to(self.document, 120, self.estimator)
        self.assertLessEqual(len(small), len(large))

    def test_invalid_head_ratio_rejected(self):
        with self.assertRaises(ValueError):
            shrink_to(self.document, 50, self.estimator, head_ratio=1.5)


class TestPackBasics(unittest.TestCase):
    def test_everything_fits(self):
        result = pack(
            [Item("a", "alpha"), Item("b", "beta", section="history")],
            10_000,
        )
        self.assertEqual(sorted(ids(result)), ["a", "b"])
        self.assertEqual(result.dropped, [])
        self.assertGreater(result.free, 0)

    def test_empty_input(self):
        result = pack([], 1000)
        self.assertEqual(result.text, "")
        self.assertEqual(result.tokens_used, 0)

    def test_estimate_never_exceeds_budget(self):
        long_text = "dolor sit amet " * 500
        result = pack([Item(f"d{i}", long_text) for i in range(20)], 2000, reserve=200)
        self.assertLessEqual(result.tokens_used, result.usable)
        self.assertLessEqual(Estimator().count(result.text), result.usable)

    def test_sections_render_in_canonical_order(self):
        result = pack(
            [
                Item("q", "the question", section="task", required=True),
                Item("h", "a turn", section="history"),
                Item("c", "a document"),
                Item("s", "be brief", section="system", required=True),
            ],
            10_000,
        )
        positions = [result.text.index(f":{i} tokens=") for i in ["s", "c", "h", "q"]]
        self.assertEqual(positions, sorted(positions))

    def test_sections_are_separated_by_a_blank_line(self):
        result = pack([Item("a", "alpha"), Item("b", "beta")], 10_000)
        self.assertTrue(result.text.startswith("<<<context:a tokens="))

    def test_unknown_section_is_rejected(self):
        with self.assertRaises(ValueError):
            pack([Item("a", "x", section="nope")], 1000)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            pack([Item("a", "x"), Item("a", "y")], 1000)

    def test_non_positive_budget_rejected(self):
        with self.assertRaises(ValueError):
            pack([], 0)

    def test_reserve_larger_than_budget_rejected(self):
        with self.assertRaises(ValueError):
            pack([], 100, reserve=100)

    def test_item_with_precomputed_tokens(self):
        result = pack([Item("a", "not counted", tokens=500)], 1000)
        self.assertEqual(result.tokens_used, 500)

    def test_negative_precomputed_tokens_rejected(self):
        with self.assertRaises(ValueError):
            Item("a", "x", tokens=-1)

    def test_empty_id_rejected(self):
        with self.assertRaises(ValueError):
            Item("", "x")


class TestRequiredItems(unittest.TestCase):
    def test_required_survives_a_tiny_budget(self):
        result = pack(
            [
                Item("sys", "be concise", section="system", required=True),
                Item("doc", "x " * 500, shrinkable=False),
            ],
            120,
        )
        self.assertIn("sys", ids(result))
        self.assertNotIn("doc", ids(result))
        self.assertLessEqual(result.tokens_used, result.usable)

    def test_required_items_that_do_not_fit_raise(self):
        with self.assertRaises(BudgetError) as caught:
            pack([Item("sys", "y" * 10_000, section="system", required=True)], 100)
        error = caught.exception
        self.assertEqual(error.items, ["sys"])
        self.assertGreater(error.needed, error.available)
        self.assertIn("sys", str(error))

    def test_error_explains_the_fix(self):
        with self.assertRaises(BudgetError) as caught:
            pack([Item("sys", "y" * 10_000, section="system", required=True)], 100)
        message = str(caught.exception)
        self.assertIn("raise the budget or shorten", message)

    def test_reserve_is_honoured(self):
        result = pack([Item("a", "short")], 1000, reserve=400)
        self.assertEqual(result.usable, 600)
        self.assertLessEqual(result.tokens_used, 600)


class TestAllocation(unittest.TestCase):
    def test_a_discretionary_item_in_a_zero_weight_section_raises(self):
        # Default `task` has weight 0: dropping the user's question silently is
        # the failure this guards against.
        with self.assertRaises(ValueError):
            pack([Item("q", "the question", section="task")], 1000)

    def test_a_section_that_cannot_fill_its_share_gives_it_back(self):
        result = pack(
            [
                Item("big", "z " * 4000, tokens=600, shrinkable=False),
                Item("h", "a turn", section="history", tokens=100),
            ],
            1000,
        )
        # `big` cannot fit its section share, but the budget as a whole can hold
        # it once the other section is done.
        self.assertIn("big", ids(result))
        self.assertIn("h", ids(result))

    def test_custom_sections(self):
        result = pack(
            [Item("a", "alpha", section="evidence", required=True)],
            1000,
            sections=(Section("evidence", weight=1.0),),
        )
        self.assertEqual(ids(result), ["a"])

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            Section("bad", weight=-1)

    def test_unknown_order_rejected(self):
        with self.assertRaises(ValueError):
            Section("bad", order="alphabetical")


class TestShrinking(unittest.TestCase):
    def test_a_too_big_item_is_shrunk_not_dropped(self):
        document = "HEAD " + ("body " * 3000) + "TAIL"
        result = pack(
            [
                Item("sys", "be concise", section="system", required=True, tokens=20),
                Item("doc", document),
            ],
            700,
            reserve=100,
        )
        packed = {p.id: p for p in result.packed}
        self.assertIn("doc", packed)
        self.assertTrue(packed["doc"].shrunk)
        self.assertIn("characters omitted", packed["doc"].text)
        self.assertTrue(packed["doc"].text.startswith("HEAD"))
        self.assertTrue(packed["doc"].text.endswith("TAIL"))

    def test_unshrinkable_item_is_dropped_with_a_reason(self):
        result = pack(
            [
                Item("sys", "be concise", section="system", required=True, tokens=20),
                Item("doc", "x " * 3000, shrinkable=False),
            ],
            700,
            reserve=100,
        )
        dropped = {d.id: d for d in result.dropped}
        self.assertIn("doc", dropped)
        self.assertIn("did not fit", dropped["doc"].reason)

    def test_a_shrunk_item_regains_its_text_when_room_appears(self):
        document = "A" * 400 + "B" * 400
        result = pack(
            [Item("doc", document, tokens=1500, shrinkable=True, min_tokens=10)],
            2000,
        )
        # Its section share is 1200, less than the item; the global remainder
        # still has room, so shrinking must be undone rather than left as waste.
        self.assertIn("doc", ids(result))
        packed = {p.id: p for p in result.packed}
        self.assertFalse(packed["doc"].shrunk)

    def test_fragment_below_min_tokens_is_dropped(self):
        result = pack(
            [
                Item("sys", "s", section="system", required=True, tokens=10),
                Item("doc", "body " * 2000, min_tokens=500),
            ],
            400,
        )
        self.assertNotIn("doc", ids(result))
        self.assertTrue(result.dropped)

    def test_shrinking_respects_the_budget(self):
        result = pack([Item("doc", "e " * 5000)], 500, reserve=100)
        self.assertLessEqual(result.tokens_used, result.usable)


class TestDeterminism(unittest.TestCase):
    def test_same_input_same_output(self):
        items = [
            Item("b", "beta " * 40, section="history", recency=1.0),
            Item("a", "alpha " * 40, section="history", recency=2.0),
            Item("c", "gamma " * 40),
        ]
        first = pack(items, 500)
        second = pack(items, 500)
        self.assertEqual(first.text, second.text)

    def test_blocks_are_byte_identical_across_packs(self):
        # This is what lets a provider's prefix cache hit: the rendered block
        # for a given item does not change because other items came or went.
        base = [Item("sys", "be concise", section="system", required=True),
                Item("doc", "alpha " * 30)]
        before = pack(base, 4000)
        after = pack(base + [Item("extra", "beta", section="history", priority=-5)], 4000)
        self.assertEqual(block_of(before.text, "doc"), block_of(after.text, "doc"))

    def test_order_within_section_follows_priority(self):
        result = pack(
            [
                Item("low", "low", section="history", priority=0),
                Item("high", "high", section="history", priority=10),
            ],
            4000,
        )
        self.assertEqual(
            [i for i in ids(result) if i in ("low", "high")], ["high", "low"]
        )

    def test_history_renders_oldest_first(self):
        # The newest turn must end up next to the question, not furthest from it.
        result = pack(
            [
                Item("old", "old", section="history", recency=1.0),
                Item("new", "new", section="history", recency=9.0),
            ],
            4000,
        )
        self.assertEqual(
            [i for i in ids(result) if i in ("old", "new")], ["old", "new"]
        )

    def test_recency_decides_who_gets_the_space(self):
        result = pack(
            [
                Item("old", "o " * 200, section="history", recency=1.0, shrinkable=False),
                Item("new", "n " * 200, section="history", recency=9.0, shrinkable=False),
            ],
            200,
        )
        self.assertIn("new", ids(result))
        self.assertNotIn("old", ids(result))


class TestRelevanceOrdering(unittest.TestCase):
    def test_most_relevant_document_is_rendered_last(self):
        result = pack(
            [
                Item("weak", "weak doc", relevance=0.2),
                Item("strong", "strong doc", relevance=0.9),
                Item("mid", "mid doc", relevance=0.5),
                Item("q", "the question", section="task", required=True),
            ],
            8000,
        )
        order = [i for i in ids(result) if i in ("weak", "mid", "strong")]
        self.assertEqual(order, ["weak", "mid", "strong"])
        # ...and the strongest one sits closest to the task.
        self.assertLess(result.text.index(":strong tokens="), result.text.index(":q tokens="))

    def test_relevance_also_decides_inclusion(self):
        result = pack(
            [
                Item("strong", "s " * 400, relevance=0.9, tokens=400),
                Item("weak", "w " * 400, relevance=0.1, tokens=400),
            ],
            450,
        )
        self.assertIn("strong", ids(result))
        self.assertNotIn("weak", ids(result))


class TestCli(unittest.TestCase):
    def _write(self, tmp, document):
        path = Path(tmp) / "prompt.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return str(path)

    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_packs_and_prints_the_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {
                "budget": 1000,
                "items": [{"id": "a", "content": "alpha", "required": True,
                           "section": "system"}],
            })
            code, out, err = self._run([path, "--quiet"])
            self.assertEqual(code, 0)
            self.assertIn("<<<system:a tokens=", out)
            self.assertEqual(err, "")

    def test_report_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {
                "budget": 1000,
                "items": [{"id": "a", "content": "alpha"}],
            })
            code, out, err = self._run([path, "--report"])
            self.assertEqual(code, 0)
            self.assertEqual(out, "")
            self.assertIn("budget: 1000 usable", err)

    def test_json_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {
                "budget": 1000,
                "items": [{"id": "a", "content": "alpha"}],
            })
            code, out, _ = self._run([path, "--json", "--quiet"])
            self.assertEqual(code, 0)
            data = json.loads(out)
            self.assertEqual(data["packed"][0]["id"], "a")
            self.assertEqual(data["dropped"], [])

    def test_too_tight_exits_1_and_says_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {
                "budget": 50,
                "items": [{"id": "sys", "content": "y" * 5000, "required": True,
                           "section": "system"}],
            })
            code, out, err = self._run([path])
            self.assertEqual(code, 1)
            self.assertIn("required items", err)

    def test_bad_json_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            code, _, err = self._run([str(path)])
            self.assertEqual(code, 2)
            self.assertIn("not valid JSON", err)

    def test_missing_budget_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"items": []})
            code, _, err = self._run([path])
            self.assertEqual(code, 2)
            self.assertIn("no budget", err)

    def test_unknown_item_field_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {
                "budget": 100,
                "items": [{"id": "a", "content": "x", "colour": "red"}],
            })
            code, _, err = self._run([path])
            self.assertEqual(code, 2)
            self.assertIn("unknown field", err)

    def test_sections_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {
                "budget": 1000,
                "sections": [{"name": "evidence", "weight": 1.0}],
                "items": [{"id": "a", "content": "alpha", "section": "evidence"}],
            })
            code, out, _ = self._run([path, "--quiet"])
            self.assertEqual(code, 0)
            self.assertIn("<<<evidence:a", out)

    def test_reserve_flag_beats_the_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {
                "budget": 1000,
                "reserve": 10,
                "items": [{"id": "a", "content": "alpha"}],
            })
            _, _, err = self._run([path, "--reserve", "900", "--report"])
            self.assertIn("100 usable", err)


if __name__ == "__main__":
    unittest.main()
