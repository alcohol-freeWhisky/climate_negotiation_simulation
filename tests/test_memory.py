"""Tests for negotiation memory behavior."""

import unittest

from src.memory.negotiation_memory import MemoryEntry, NegotiationMemory


class TestNegotiationMemory(unittest.TestCase):
    """Targeted tests for prompt context ordering and compression."""

    def test_get_context_for_prompt_prioritizes_recent_rounds(self):
        memory = NegotiationMemory(agent_id="TEST_AGENT")
        memory.summary_memory = [
            MemoryEntry(
                sequence_id=-1,
                round_number=1,
                phase="informal_consultations",
                agent_id="SYSTEM",
                content="Older summary",
                entry_type="summary",
            ),
            MemoryEntry(
                sequence_id=-1,
                round_number=8,
                phase="informal_consultations",
                agent_id="SYSTEM",
                content="Recent summary",
                entry_type="summary",
            ),
        ]
        memory.add_statement(
            round_number=2,
            phase="informal_consultations",
            agent_id="TEST_AGENT",
            content="Older working entry",
        )
        memory.add_statement(
            round_number=9,
            phase="informal_consultations",
            agent_id="TEST_AGENT",
            content="Recent working entry",
        )

        context = memory.get_context_for_prompt(current_round=10)

        self.assertLess(
            context.index("[informal_consultations R8 summary] Recent summary"),
            context.index("[informal_consultations R1 summary] Older summary"),
        )
        self.assertLess(
            context.index("[informal_consultations R9] YOU: Recent working entry"),
            context.index("[informal_consultations R2] YOU: Older working entry"),
        )

    def test_maybe_compress_merges_entries_from_same_round(self):
        memory = NegotiationMemory(agent_id="TEST_AGENT", working_memory_size=1)
        memory.working_memory = [
            MemoryEntry(
                sequence_id=0,
                round_number=3,
                phase="first_reading",
                agent_id="A",
                content="Entry one",
                entry_type="statement",
            ),
            MemoryEntry(
                sequence_id=1,
                round_number=3,
                phase="first_reading",
                agent_id="B",
                content="Entry two",
                entry_type="statement",
            ),
            MemoryEntry(
                sequence_id=2,
                round_number=3,
                phase="first_reading",
                agent_id="C",
                content="Entry three",
                entry_type="statement",
            ),
            MemoryEntry(
                sequence_id=3,
                round_number=4,
                phase="first_reading",
                agent_id="D",
                content="Entry four",
                entry_type="statement",
            ),
        ]

        memory._maybe_compress()

        self.assertEqual(len(memory.summary_memory), 1)
        self.assertEqual(memory.summary_memory[0].phase, "first_reading")
        self.assertEqual(memory.summary_memory[0].round_number, 3)
        self.assertIn("A: Entry one", memory.summary_memory[0].content)
        self.assertIn("B: Entry two", memory.summary_memory[0].content)
        self.assertIn("C: Entry three", memory.summary_memory[0].content)


if __name__ == "__main__":
    unittest.main()
