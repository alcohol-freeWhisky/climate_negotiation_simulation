"""
Amendment Processor - Parses and processes text amendments from agent responses.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedAmendment:
    """A parsed amendment from an agent's response."""
    action: str          # "add", "delete", "modify", "support", "oppose", "accept", "pass", "argue"
    original_text: str = ""
    proposed_text: str = ""
    reference: str = ""
    reasoning: str = ""


class AmendmentProcessor:
    """
    Parses agent responses to extract structured amendments.
    Handles the various ways agents might express text changes.
    """

    PATTERNS = {
        "add": [
            r"PROPOSE ADD:\s*(.+?)(?=PROPOSE|SUPPORT|OPPOSE|$)",
            r"(?:We|I) propose (?:to )?add(?:ing)?[:\s]+[\"']?(.+?)[\"']?(?:\.|$)",
            r"ADD:\s*(.+?)(?=\n|$)",
        ],
        "delete": [
            r"PROPOSE DELETE:\s*(.+?)(?=PROPOSE|SUPPORT|OPPOSE|$)",
            r"(?:We|I) propose (?:to )?delet(?:e|ing)[:\s]+[\"']?(.+?)[\"']?(?:\.|$)",
            r"DELETE:\s*(.+?)(?=\n|$)",
        ],
        "modify": [
            r"PROPOSE MODIFY:\s*(.+?)\s*(?:→|➜|->|=>)\s*(.+?)(?=PROPOSE|SUPPORT|OPPOSE|$)",
            r"(?:We|I) propose (?:to )?(?:modify|change|replace|amend)[:\s]+[\"']?(.+?)[\"']?\s*(?:to|with|by)\s*[\"']?(.+?)[\"']?(?:\.|$)",
            r"MODIFY:\s*(.+?)\s*(?:→|➜|->|=>)\s*(.+?)(?=\n|$)",
        ],
        "support": [
            r"SUPPORT:\s*(.+?)(?=PROPOSE|OPPOSE|$)",
            r"(?:We|I) support (?:the )?(?:proposal (?:by|of|from)\s*)?(.+?)(?:\.|$)",
            r"(?:We|I) (?:associate|align) (?:ourselves? )?with\s+(.+?)(?:\.|$)",
        ],
        "oppose": [
            r"OPPOSE:\s*(.+?)(?=PROPOSE|SUPPORT|$)",
            r"OBJECT:\s*(.+?)(?=PROPOSE|SUPPORT|$)",
            r"(?:We|I) (?:oppose|cannot accept|reject|object to)\s+(.+?)(?:\.|$)",
        ],
        "accept": [
            r"^ACCEPT",
            r"(?:We|I) (?:can )?accept (?:the |this )?(?:text|paragraph|proposal)",
        ],
        "pass": [
            r"^PASS",
            r"^RESERVE",
            r"(?:We|I) have no (?:comment|amendment|proposal)",
            r"(?:We|I) (?:reserve|defer)",
        ],
    }

    def __init__(self):
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for action, patterns in self.PATTERNS.items():
            compiled = []
            for p in patterns:
                try:
                    compiled.append(re.compile(p, re.IGNORECASE | re.DOTALL))
                except re.error as exc:
                    logger.error(
                        "Failed to compile regex for action=%s pattern=%r: %s",
                        action, p, exc,
                    )
            self._compiled_patterns[action] = compiled

    def parse_response(self, response: str) -> List[ParsedAmendment]:
        """
        Parse an agent's response to extract all amendments.

        Args:
            response: The raw text response from the agent.

        Returns:
            List of ParsedAmendment objects (never empty).
        """
        if response is None or not isinstance(response, str):
            return []

        cleaned = response.strip()
        if "no textual change" in cleaned.lower():
            return [ParsedAmendment(action="pass")]

        # 1. Try structured format first (most reliable)
        structured = self._parse_structured(cleaned)
        if structured:
            return structured

        # 2. Fall back to pattern matching
        amendments: List[ParsedAmendment] = []
        for action, compiled_patterns in self._compiled_patterns.items():
            for pattern in compiled_patterns:
                for match in pattern.finditer(cleaned):
                    amendment = self._create_amendment(action, match)
                    if amendment:
                        amendments.append(amendment)

        # 3. If still nothing, classify the whole response
        if not amendments:
            amendments.append(self._classify_overall(cleaned))

        return amendments

    def get_primary_action(self, response: str) -> str:
        """
        Infer the primary procedural action for a response.

        This is intentionally stricter than parse_response(): in settings like
        final plenary we care first about the opening verdict (ACCEPT/OPPOSE),
        even if the speaker later proposes amendment text.
        """
        non_empty_lines = [
            raw_line.strip()
            for raw_line in response.splitlines()
            if raw_line.strip()
        ]
        head_lines = non_empty_lines[:6]
        head_text = "\n".join(head_lines)
        head_lower = head_text.lower()

        hard_oppose_markers = [
            "cannot accept",
            "cannot support",
            "cannot agree",
            "cannot join consensus",
            "will not accept",
            "we oppose",
            "we object",
            "object to",
            "strongly oppose",
        ]
        if any(marker in head_lower for marker in hard_oppose_markers):
            return "oppose"

        for line in head_lines:
            line_upper = line.upper()
            line_lower = line.lower()

            if line_upper.startswith("ACCEPT"):
                return "accept"
            if line_upper.startswith(("OPPOSE", "OBJECT")):
                return "oppose"
            if line_upper.startswith(("PASS", "RESERVE")):
                return "pass"
            if line_upper.startswith("PROPOSE MODIFY"):
                return "modify"

            if re.search(r"\b(?:we|i)\s+oppose\b", line_lower):
                return "oppose"
            if re.search(r"\b(?:we|i)\s+(?:can\s+)?accept\b", line_lower):
                return "accept"

        amendments = self.parse_response(response)
        return amendments[0].action if amendments else "argue"

    def summarize_amendments(
        self, agent_amendments: Dict[str, List[ParsedAmendment]]
    ) -> str:
        """Create a human-readable summary of all amendments from all agents."""
        lines: List[str] = []
        for agent_id, amendments in agent_amendments.items():
            lines.append(f"\n{agent_id}:")
            for amd in amendments:
                if amd.action == "add":
                    lines.append(f'  ADD: "{amd.proposed_text}"')
                elif amd.action == "delete":
                    lines.append(f'  DELETE: "{amd.original_text}"')
                elif amd.action == "modify":
                    lines.append(
                        f'  MODIFY: "{amd.original_text}" → "{amd.proposed_text}"'
                    )
                elif amd.action == "support":
                    lines.append(f"  SUPPORT: {amd.reference}")
                elif amd.action == "oppose":
                    lines.append(f"  OPPOSE: {amd.reference}")
                elif amd.action == "accept":
                    lines.append("  ACCEPT current text")
                elif amd.action == "pass":
                    lines.append("  PASS")
                elif amd.action == "argue":
                    lines.append(f"  STATEMENT: {amd.reasoning[:100]}...")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _parse_structured(self, text: str) -> Optional[List[ParsedAmendment]]:
        """
        Try to parse responses that follow the structured format.
        Returns None if nothing was found.
        """
        amendments: List[ParsedAmendment] = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("PROPOSE ADD:"):
                proposed = line[len("PROPOSE ADD:"):].strip()
                amendments.append(ParsedAmendment(action="add", proposed_text=proposed))

            elif line.startswith("PROPOSE DELETE:"):
                original = line[len("PROPOSE DELETE:"):].strip()
                amendments.append(ParsedAmendment(action="delete", original_text=original))

            elif line.startswith("PROPOSE MODIFY:"):
                rest = line[len("PROPOSE MODIFY:"):].strip()
                for arrow in ("→", "➜", "->", "=>"):
                    if arrow in rest:
                        parts = rest.split(arrow, 1)
                        amendments.append(
                            ParsedAmendment(
                                action="modify",
                                original_text=parts[0].strip(),
                                proposed_text=parts[1].strip(),
                            )
                        )
                        break

            elif line.startswith("SUPPORT:"):
                ref = line[len("SUPPORT:"):].strip()
                amendments.append(ParsedAmendment(action="support", reference=ref))

            elif line.upper().startswith(("OPPOSE", "OBJECT")):
                if ":" in line:
                    ref = line.split(":", 1)[1].strip()
                else:
                    ref = line
                amendments.append(ParsedAmendment(action="oppose", reference=ref))

            elif line.upper().startswith("ACCEPT"):
                amendments.append(ParsedAmendment(action="accept"))

            elif line.upper().startswith(("PASS", "RESERVE")):
                amendments.append(ParsedAmendment(action="pass"))

        return amendments if amendments else None

    def _create_amendment(
        self, action: str, match: re.Match
    ) -> Optional[ParsedAmendment]:
        """Create a ParsedAmendment from a regex match."""
        groups = match.groups()
        if not groups:
            return ParsedAmendment(action=action)

        if action == "modify" and len(groups) >= 2:
            return ParsedAmendment(
                action="modify",
                original_text=groups[0].strip(),
                proposed_text=groups[1].strip(),
            )
        elif action == "add":
            return ParsedAmendment(action="add", proposed_text=groups[0].strip())
        elif action == "delete":
            return ParsedAmendment(action="delete", original_text=groups[0].strip())
        elif action in ("support", "oppose"):
            return ParsedAmendment(action=action, reference=groups[0].strip())
        elif action in ("accept", "pass"):
            return ParsedAmendment(action=action)
        return None

    def _classify_overall(self, text: str) -> ParsedAmendment:
        """When no specific amendments are found, classify the overall response."""
        text_lower = text.lower()

        # Check for opposition indicators FIRST (before accept, since
        # "cannot accept" contains "accept")
        oppose_words = [
            "cannot accept", "oppose", "reject", "unacceptable",
            "object", "we object", "strongly object",
        ]
        if any(w in text_lower for w in oppose_words):
            return ParsedAmendment(action="oppose", reasoning=text[:200])

        accept_words = ["accept", "agree", "can live with", "no objection"]
        if any(w in text_lower for w in accept_words):
            return ParsedAmendment(action="accept", reasoning=text[:200])

        pass_words = ["no comment", "reserve", "no further", "pass"]
        if any(w in text_lower for w in pass_words):
            return ParsedAmendment(action="pass")

        return ParsedAmendment(action="argue", reasoning=text[:300])
