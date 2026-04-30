"""
Negotiation Memory - Manages conversation history and context for agents.
Implements a tiered memory system to handle context window limitations.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry."""
    sequence_id: int         # Global monotonic counter (never resets)
    round_number: int        # Phase-local round number (for display)
    phase: str
    agent_id: str
    content: str
    entry_type: str          # "statement", "proposal", "decision", "summary"
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)


class NegotiationMemory:
    """
    Tiered memory system for negotiation agents.

    Three tiers:
    1. Working Memory: Recent exchanges (last N entries by sequence_id)
    2. Summary Memory: Compressed summaries of earlier entries
    3. Core Memory: Permanent facts (agent identity, red lines, draft text)

    FIX for Issue #5: All ordering and compression decisions use
    ``sequence_id`` (a global monotonic counter) instead of the raw
    ``round_number`` which can restart across negotiation phases.
    """

    def __init__(
        self,
        agent_id: str,
        working_memory_size: int = 30,
        max_summary_entries: int = 20,
        max_tokens_budget: int = 6000,
    ):
        self.agent_id = agent_id
        self.working_memory_size = working_memory_size
        self.max_summary_entries = max_summary_entries
        self.max_tokens_budget = max_tokens_budget

        # Global counter – never reset
        self._next_sequence_id = 0

        # Core memory - never compressed or removed
        self.core_memory: Dict[str, str] = {}

        # Working memory - recent detailed exchanges
        self.working_memory: List[MemoryEntry] = []

        # Summary memory - compressed earlier entries
        self.summary_memory: List[MemoryEntry] = []

        # Full history (for logging, not used in prompts)
        self.full_history: List[MemoryEntry] = []

        # Track concessions and commitments
        self.concessions_made: List[Dict[str, Any]] = []
        self.concessions_received: List[Dict[str, Any]] = []
        self.commitments: List[str] = []

        # Per-agent relationship tracking
        self.agent_interactions: Dict[str, List[str]] = defaultdict(list)

    def _allocate_sequence_id(self) -> int:
        """Return the next global sequence id and increment."""
        sid = self._next_sequence_id
        self._next_sequence_id += 1
        return sid

    def set_core_memory(self, key: str, value: str):
        """Set a core memory item (permanent)."""
        self.core_memory[key] = value

    def add_entry(self, entry: MemoryEntry):
        """Add a new memory entry."""
        self.full_history.append(entry)
        self.working_memory.append(entry)

        # Track interactions with other agents
        if entry.agent_id != self.agent_id:
            self.agent_interactions[entry.agent_id].append(entry.content[:200])

        # Check if we need to compress working memory
        self._maybe_compress()

    def add_statement(
        self,
        round_number: int,
        phase: str,
        agent_id: str,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ):
        """Convenience method to add a statement entry."""
        entry = MemoryEntry(
            sequence_id=self._allocate_sequence_id(),
            round_number=round_number,
            phase=phase,
            agent_id=agent_id,
            content=content,
            entry_type="statement",
            importance=importance,
            tags=tags or [],
        )
        self.add_entry(entry)

    def record_concession(
        self,
        round_number: int,
        issue: str,
        description: str,
        made_by: str,
        received_by: str,
    ):
        """Record a concession for tracking."""
        concession = {
            "round": round_number,
            "issue": issue,
            "description": description,
            "made_by": made_by,
            "received_by": received_by,
        }
        if made_by == self.agent_id:
            self.concessions_made.append(concession)
        if received_by == self.agent_id:
            self.concessions_received.append(concession)

    def record_commitment(self, commitment: str):
        """Record a commitment made by this agent."""
        self.commitments.append(commitment)

    def get_context_for_prompt(self, current_round: int) -> str:
        """Build the memory context string to include in prompts."""
        sections = []
        prioritized_summaries = self._prioritize_entries_for_context(
            self.summary_memory[-self.max_summary_entries:],
            current_round,
        )
        prioritized_working = self._prioritize_entries_for_context(
            self.working_memory,
            current_round,
        )

        # Tier 1: Core memory (always included)
        core_text = self._format_core_memory()
        sections.append(("CORE CONTEXT", core_text))

        # Tier 2: Summary of earlier entries
        if prioritized_summaries:
            summary_text = self._format_summary_memory(prioritized_summaries)
            sections.append(("EARLIER ROUNDS SUMMARY", summary_text))

        # Tier 3: Recent detailed exchanges
        working_text = self._format_working_memory(prioritized_working)
        if working_text:
            sections.append(("RECENT EXCHANGES", working_text))

        # Tier 4: Commitments and concessions tracking
        tracking_text = self._format_tracking()
        if tracking_text:
            sections.append(("YOUR COMMITMENTS & CONCESSIONS", tracking_text))

        result = ""
        for header, content in sections:
            if content.strip():
                result += f"\n## {header}\n{content}\n"

        return result

    def get_recent_history_text(self, n_entries: int = 15) -> str:
        """
        Get a formatted text of the most recent N entries.

        FIX for Issue #5: Use sequence_id ordering instead of
        round_number so entries from different phases are
        correctly interleaved.
        """
        return self.get_recent_history_text_limited(
            n_entries=n_entries,
            max_chars_per_entry=300,
        )

    def get_recent_history_text_limited(
        self,
        n_entries: int = 8,
        max_chars_per_entry: int = 160,
    ) -> str:
        """Get recent history with per-entry truncation for token efficiency."""
        if not self.working_memory:
            return "No previous discussion."

        # Take the last n entries by sequence_id (they are already sorted)
        recent = self.working_memory[-n_entries:]

        lines = []
        for entry in recent:
            prefix = f"[{entry.phase} R{entry.round_number}] {entry.agent_id}"
            lines.append(f"{prefix}: {entry.content[:max_chars_per_entry]}")

        return "\n".join(lines) if lines else "No recent discussion."

    def get_compact_context_for_prompt(
        self,
        n_recent_entries: int = 6,
        max_chars_per_entry: int = 120,
        exclude_core_keys: Optional[List[str]] = None,
    ) -> str:
        """Build a compact prompt context that avoids duplicating full draft text."""
        exclude = set(exclude_core_keys or [])
        sections = []

        core_lines = []
        for key, value in self.core_memory.items():
            if key in exclude:
                continue
            core_lines.append(f"**{key}**: {value[:200]}")
        if core_lines:
            sections.append(("CORE CONTEXT", "\n".join(core_lines)))

        if self.summary_memory:
            summary_lines = []
            for entry in self.summary_memory[-3:]:
                summary_lines.append(
                    f"[{entry.phase} R{entry.round_number} summary] {entry.content[:180]}"
                )
            if summary_lines:
                sections.append(("EARLIER ROUNDS SUMMARY", "\n".join(summary_lines)))

        recent_text = self.get_recent_history_text_limited(
            n_entries=n_recent_entries,
            max_chars_per_entry=max_chars_per_entry,
        )
        if recent_text and recent_text != "No previous discussion.":
            sections.append(("RECENT EXCHANGES", recent_text))

        tracking_text = self._format_tracking()
        if tracking_text:
            sections.append(("YOUR COMMITMENTS & CONCESSIONS", tracking_text))

        result = ""
        for header, content in sections:
            if content.strip():
                result += f"\n## {header}\n{content}\n"

        return result

    def _format_core_memory(self) -> str:
        lines = []
        for key, value in self.core_memory.items():
            lines.append(f"**{key}**: {value}")
        return "\n".join(lines)

    def _format_summary_memory(
        self,
        entries: Optional[List[MemoryEntry]] = None,
    ) -> str:
        lines = []
        selected_entries = (
            entries
            if entries is not None
            else self.summary_memory[-self.max_summary_entries:]
        )
        for entry in selected_entries:
            lines.append(f"[{entry.phase} R{entry.round_number} summary] {entry.content}")
        return "\n".join(lines)

    def _format_working_memory(
        self,
        entries: Optional[List[MemoryEntry]] = None,
    ) -> str:
        lines = []
        for entry in entries if entries is not None else self.working_memory:
            speaker = "YOU" if entry.agent_id == self.agent_id else entry.agent_id
            lines.append(
                f"[{entry.phase} R{entry.round_number}] {speaker}: {entry.content[:400]}"
            )
        return "\n".join(lines)

    def _prioritize_entries_for_context(
        self,
        entries: List[MemoryEntry],
        current_round: int,
    ) -> List[MemoryEntry]:
        """Prefer entries from the last three rounds when building prompt context."""
        recent_cutoff = current_round - 3
        recent_entries = [e for e in entries if e.round_number >= recent_cutoff]
        older_entries = [e for e in entries if e.round_number < recent_cutoff]
        return recent_entries + older_entries

    def _format_tracking(self) -> str:
        lines = []
        if self.commitments:
            lines.append("Commitments you've made:")
            for c in self.commitments[-5:]:
                lines.append(f"  - {c}")
        if self.concessions_made:
            lines.append("Concessions you've made:")
            for c in self.concessions_made[-5:]:
                lines.append(f"  - Round {c['round']}: {c['description']}")
        if self.concessions_received:
            lines.append("Concessions you've received:")
            for c in self.concessions_received[-5:]:
                lines.append(
                    f"  - Round {c['round']} from {c['made_by']}: {c['description']}"
                )
        return "\n".join(lines)

    def _maybe_compress(self):
        """
        Compress working memory if it exceeds the size limit.

        FIX for Issue #5: Use sequence_id for ordering, not round_number.
        """
        if len(self.working_memory) <= self.working_memory_size:
            return

        # How many entries to compress
        overflow = len(self.working_memory) - self.working_memory_size
        to_compress = self.working_memory[:overflow]
        self.working_memory = self.working_memory[overflow:]

        # Group by (phase, round_number) and create summaries
        groups: Dict[tuple, List[MemoryEntry]] = defaultdict(list)
        for entry in to_compress:
            key = (entry.phase, entry.round_number)
            groups[key].append(entry)

        for (phase, rn), entries in groups.items():
            summary = self._summarize_group(phase, rn, entries)
            self.summary_memory.append(summary)

        # Trim summary memory if needed
        if len(self.summary_memory) > self.max_summary_entries:
            self.summary_memory = self.summary_memory[-self.max_summary_entries:]

        logger.debug(
            f"Compressed {len(to_compress)} entries for agent {self.agent_id}. "
            f"Working: {len(self.working_memory)}, Summary: {len(self.summary_memory)}"
        )

    def _summarize_group(
        self, phase: str, round_number: int, entries: List[MemoryEntry]
    ) -> MemoryEntry:
        """Create a compressed summary of a group of entries."""
        sorted_entries = sorted(entries, key=lambda e: e.importance, reverse=True)
        top_entries = sorted_entries[:3]

        summary_parts = []
        for e in top_entries:
            summary_parts.append(f"{e.agent_id}: {e.content[:150]}")

        summary_text = " | ".join(summary_parts)

        return MemoryEntry(
            sequence_id=-1,  # summaries don't need a valid sequence_id
            round_number=round_number,
            phase=phase,
            agent_id="SYSTEM",
            content=summary_text,
            entry_type="summary",
            importance=0.7,
        )
