"""Tests for TextManager."""

import unittest
from src.engine.text_manager import TextManager


class TestTextManager(unittest.TestCase):
    """Test cases for TextManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "text": {
                "bracket_open": "[",
                "bracket_close": "]",
                "option_prefix": "Option",
            },
        }
        self.tm = TextManager(self.config)

    def test_load_simple_text(self):
        """Test loading simple text."""
        text = """1. This is paragraph one.

2. This is paragraph two.

3. This is paragraph three."""

        self.tm.load_draft_text(text)
        self.assertEqual(len(self.tm.paragraphs), 3)
        self.assertEqual(self.tm.paragraphs[0].paragraph_id, 1)

    def test_detect_brackets(self):
        """Test bracket detection."""
        text = """1. This is [bracketed text] here.

2. This is clean text.

3. This has [multiple] [brackets]."""

        self.tm.load_draft_text(text)
        self.assertTrue(self.tm.paragraphs[0].is_bracketed)
        self.assertFalse(self.tm.paragraphs[1].is_bracketed)
        self.assertTrue(self.tm.paragraphs[2].is_bracketed)

    def test_get_full_text(self):
        """Test full text retrieval."""
        text = """1. Paragraph one.

2. Paragraph two."""

        self.tm.load_draft_text(text)
        full = self.tm.get_full_text()
        self.assertIn("Paragraph one", full)
        self.assertIn("Paragraph two", full)

    def test_get_disputed_paragraphs(self):
        """Test disputed paragraph retrieval."""
        text = """1. This is [disputed].

2. This is agreed.

3. This is also [disputed]."""

        self.tm.load_draft_text(text)
        disputed = self.tm.get_disputed_paragraphs()
        self.assertEqual(len(disputed), 2)

    def test_add_amendment(self):
        """Test adding amendments."""
        text = "1. Test paragraph."
        self.tm.load_draft_text(text)

        amendment = self.tm.add_amendment(
            agent_id="EU",
            paragraph_id=1,
            amendment_type="modify",
            original_text="Test",
            proposed_text="Revised test",
        )

        self.assertIsNotNone(amendment)
        self.assertEqual(amendment.agent_id, "EU")
        self.assertEqual(amendment.amendment_type, "modify")

        para = self.tm.get_paragraph(1)
        self.assertEqual(len(para.amendments), 1)
        self.assertEqual(para.status, "discussed")

    def test_get_unchanged_paragraph_ids_returns_paragraph_without_amendments(self):
        """Original paragraphs with no amendments should be reported as unchanged."""
        text = """1. First paragraph.

2. Second paragraph."""
        self.tm.load_draft_text(text)

        unchanged_ids = self.tm.get_unchanged_paragraph_ids()

        self.assertEqual(unchanged_ids, [1, 2])

    def test_get_unchanged_paragraph_ids_excludes_paragraph_with_amendments(self):
        """Paragraphs with any amendment history should be excluded."""
        text = """1. First paragraph.

2. Second paragraph."""
        self.tm.load_draft_text(text)
        self.tm.add_amendment(
            agent_id="EU",
            paragraph_id=2,
            amendment_type="modify",
            original_text="Second paragraph.",
            proposed_text="Updated second paragraph.",
        )

        unchanged_ids = self.tm.get_unchanged_paragraph_ids()

        self.assertEqual(unchanged_ids, [1])

    def test_mark_paragraph_agreed(self):
        """Test marking paragraph as agreed."""
        text = "1. This is [bracketed text]."
        self.tm.load_draft_text(text)

        self.assertTrue(self.tm.paragraphs[0].is_bracketed)
        self.tm.mark_paragraph_agreed(1)

        para = self.tm.get_paragraph(1)
        self.assertFalse(para.is_bracketed)
        self.assertEqual(para.status, "agreed")
        self.assertNotIn("[", para.text)

    def test_mark_paragraph_disputed(self):
        """Test marking paragraph as disputed."""
        text = "1. Test paragraph."
        self.tm.load_draft_text(text)

        self.tm.mark_paragraph_disputed(1)
        para = self.tm.get_paragraph(1)
        self.assertEqual(para.status, "disputed")

    def test_apply_chair_revision(self):
        """Test chair revision."""
        text = "1. Original text."
        self.tm.load_draft_text(text)

        self.tm.apply_chair_revision(
            paragraph_id=1,
            new_text="Revised text by chair.",
            note="Compromise proposal",
        )

        para = self.tm.get_paragraph(1)
        self.assertEqual(para.text, "Revised text by chair.")
        self.assertEqual(para.notes, "Compromise proposal")

    def test_update_full_text(self):
        """Test full text replacement."""
        text = "1. Original."
        self.tm.load_draft_text(text)

        new_text = "1. New first.\n\n2. New second."
        self.tm.update_full_text(new_text)

        self.assertEqual(len(self.tm.paragraphs), 2)

    def test_update_full_text_splits_numbered_lines_without_blank_lines(self):
        """Chair text without blank lines should still keep paragraphs."""
        text = "The Conference,\n1. Decides first;\n2. Decides second;\n3. Decides third;"
        self.tm.update_full_text(text, source="chair")

        self.assertEqual(len(self.tm.paragraphs), 4)
        self.assertFalse(self.tm.paragraphs[0].is_numbered)
        self.assertEqual(self.tm.paragraphs[1].original_number, "1.")
        self.assertEqual(self.tm.paragraphs[3].original_number, "3.")
        self.assertIn("2. Decides second", self.tm.get_full_text())

    def test_text_evolution_tracking(self):
        """Test that text changes are tracked."""
        text = "1. Test."
        self.tm.load_draft_text(text)
        initial_history_len = len(self.tm.history)

        self.tm.apply_chair_revision(1, "Revised.")
        self.assertGreater(len(self.tm.history), initial_history_len)

    def test_get_disputed_points_summary(self):
        """Test disputed points summary."""
        text = """1. This is [disputed point A].

2. This is clean.

3. This is [disputed point B]."""

        self.tm.load_draft_text(text)
        summary = self.tm.get_disputed_points_summary()
        self.assertEqual(len(summary), 2)
        self.assertTrue(any("1" in s for s in summary))
        self.assertTrue(any("3" in s for s in summary))

    def test_extract_options(self):
        """Test option extraction from text."""
        text = """1. Option 1: First option text. Option 2: Second option text."""
        self.tm.load_draft_text(text)
        para = self.tm.get_paragraph(1)
        self.assertTrue(len(para.options) >= 1)

    def test_empty_text(self):
        """Test loading empty text."""
        self.tm.load_draft_text("")
        self.assertEqual(len(self.tm.paragraphs), 0)

    def test_complex_negotiation_text(self):
        """Test with realistic negotiation text."""
        text = """1. The Conference of the Parties serving as the meeting of the Parties to the Paris Agreement,

2. [Recalling the provisions of Articles 6.8 and 6.9 of the Paris Agreement,]

3. [Option 1: Decides to establish a framework for non-market approaches;]
[Option 2: Decides to establish a work programme on non-market approaches;]

4. Invites Parties to [submit/communicate] their views on non-market approaches by [date];

5. Requests the Subsidiary Body for Scientific and Technological Advice to continue its consideration of this matter at its [next session]."""

        self.tm.load_draft_text(text)
        self.assertTrue(len(self.tm.paragraphs) >= 4)

        # Check that brackets are detected
        bracketed = [p for p in self.tm.paragraphs if p.is_bracketed]
        self.assertTrue(len(bracketed) >= 2)

    def test_get_adoption_ready_text_resolves_inline_options_conservatively(self):
        """Inline lexical alternatives should keep grammar while dropping add-ons."""
        text = (
            "1. Invites Parties [and observers] to submit [via the submission portal] "
            "their views by [next/third] session."
        )
        self.tm.load_draft_text(text)

        clean = self.tm.get_adoption_ready_text()

        self.assertNotIn("[", clean)
        self.assertNotIn("and observers", clean)
        self.assertNotIn("submission portal", clean)
        self.assertIn("by next session.", clean)

    def test_get_adoption_ready_text_resolves_nested_brackets_recursively(self):
        """A fully bracketed paragraph should still resolve nested choices cleanly."""
        text = (
            "1. [Requests the "
            "[Subsidiary Body for Scientific and Technological Advice] "
            "[a dedicated body] to prepare guidance [and adoption];]"
        )
        self.tm.load_draft_text(text)

        clean = self.tm.get_adoption_ready_text()

        self.assertNotIn("[", clean)
        self.assertIn("1. Requests the a dedicated body to prepare guidance;", clean)
        self.assertNotIn("and adoption", clean)

    def test_get_adoption_ready_text_keeps_substantive_list_items(self):
        """Single bracketed list items should not collapse into empty labels."""
        text = "1. Invites Parties to submit views on: (a) [Scope of work;] (b) [Governance arrangements;]"
        self.tm.load_draft_text(text)

        clean = self.tm.get_adoption_ready_text()

        self.assertNotIn("[", clean)
        self.assertIn("(a) Scope of work;", clean)
        self.assertIn("(b) Governance arrangements;", clean)

    def test_get_adoption_ready_text_keeps_leading_predicate_phrase(self):
        """Dropping a leading verb phrase must not leave a broken noun fragment."""
        text = (
            "1. Decides that the framework shall aim to:\n"
            "(d) [Promote complementarity and coherence with] the mechanisms established under Article 6, paragraphs 2 and 4;"
        )
        self.tm.load_draft_text(text)

        clean = self.tm.get_adoption_ready_text()

        self.assertNotIn("[", clean)
        self.assertIn(
            "(d) Promote complementarity and coherence with the mechanisms established under Article 6, paragraphs 2 and 4;",
            clean,
        )
        self.assertNotIn("(d) the mechanisms established", clean)


if __name__ == "__main__":
    unittest.main()
