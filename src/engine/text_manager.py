"""
Text Manager - Manages the negotiating text, brackets, options, and amendments.
Core component for tracking the evolution of the negotiated document.
"""

import copy
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TextAmendment:
    """Represents a proposed amendment to the text."""
    amendment_id: str
    agent_id: str
    amendment_type: str  # "add", "delete", "modify"
    target_paragraph: int
    original_text: str = ""
    proposed_text: str = ""
    rationale: str = ""
    supporters: List[str] = field(default_factory=list)
    opponents: List[str] = field(default_factory=list)
    status: str = "proposed"  # proposed, accepted, rejected, merged


@dataclass
class TextParagraph:
    """Represents a paragraph in the negotiating text."""
    paragraph_id: int
    text: str
    is_bracketed: bool = False
    is_numbered: bool = False       # Whether original text had a number prefix
    original_number: str = ""       # The original number prefix e.g. "1."
    display_label: str = ""         # External label shown to agents, e.g. "1" or "Preamble 2"
    options: List[str] = field(default_factory=list)
    amendments: List[TextAmendment] = field(default_factory=list)
    status: str = "draft"  # draft, discussed, agreed, disputed
    notes: str = ""


class TextManager:
    """
    Manages the negotiating text throughout the simulation.

    Key features:
    - Parse and track bracketed text
    - Manage multiple options for disputed paragraphs
    - Apply amendments
    - Track text evolution over rounds
    - Generate current text state with brackets/options
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bracket_open = config.get("text", {}).get("bracket_open", "[")
        self.bracket_close = config.get("text", {}).get("bracket_close", "]")
        self.option_prefix = config.get("text", {}).get("option_prefix", "Option")

        self.paragraphs: List[TextParagraph] = []
        self.original_paragraphs: List[TextParagraph] = []
        self.history: List[Dict[str, Any]] = []
        self.amendment_counter = 0

    def load_draft_text(self, text: str):
        """
        Load and parse a draft negotiating text.
        Splits into paragraphs and identifies brackets.
        """
        self.paragraphs = self._parse_text_to_paragraphs(text)
        self.original_paragraphs = copy.deepcopy(self.paragraphs)
        self.history = []
        self.amendment_counter = 0

        # Track amendments against the original draft paragraphs so the
        # negotiation can later distinguish untouched text from revised text
        # even after full-text chair rewrites reset the current paragraph list.
        for para in self.original_paragraphs:
            para.amendments = []

        # Save initial state
        self._save_snapshot("initial_load")
        logger.info(
            f"Loaded draft text: {len(self.paragraphs)} paragraphs, "
            f"{sum(1 for p in self.paragraphs if p.is_bracketed)} bracketed"
        )

    def _parse_text_to_paragraphs(self, text: str) -> List[TextParagraph]:
        """Parse raw text into tracked paragraph objects."""
        raw_paragraphs = self._split_into_paragraphs(text)

        paragraphs: List[TextParagraph] = []
        preamble_counter = 0
        for i, (para_text, is_numbered, original_number) in enumerate(raw_paragraphs):
            para_text = para_text.strip()
            if not para_text:
                continue

            is_bracketed = self.bracket_open in para_text
            options = self._extract_options(para_text)
            if is_numbered and original_number:
                display_label = original_number.rstrip(".")
            else:
                preamble_counter += 1
                display_label = f"Preamble {preamble_counter}"

            paragraphs.append(
                TextParagraph(
                    paragraph_id=i + 1,
                    text=para_text,
                    is_bracketed=is_bracketed,
                    is_numbered=is_numbered,
                    original_number=original_number,
                    display_label=display_label,
                    options=options,
                    status="draft",
                )
            )

        return paragraphs

    def _split_into_paragraphs(self, text: str) -> List[tuple]:
        """
        Split text into paragraphs, preserving both preamble (unnumbered)
        and numbered paragraphs as separate items.

        The chair sometimes returns a complete decision text without blank
        lines between numbered paragraphs.  Splitting only on double newlines
        collapses the document into one giant paragraph, which destroys later
        paragraph-level consultations and bracket-resolution metrics.  We
        therefore split on blank lines first, then further split any block
        containing line-start numbered paragraphs.

        Returns:
            List of (text, is_numbered, original_number) tuples.
        """
        raw_blocks = re.split(r"\n\s*\n", text.strip()) if text.strip() else []
        result = []

        number_pattern = re.compile(r'^(\d+)\.\s*(.*)', re.DOTALL)

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            for part in self._split_numbered_block(block):
                part = part.strip()
                if not part:
                    continue

                match = number_pattern.match(part)
                if match:
                    number_str = match.group(1)
                    body = match.group(2).strip()
                    result.append((body, True, f"{number_str}."))
                else:
                    result.append((part, False, ""))

        return result

    @staticmethod
    def _split_numbered_block(block: str) -> List[str]:
        """
        Split a block when multiple numbered paragraphs start on new lines.

        Lettered subparagraphs such as "(a)" are deliberately left inside the
        preceding numbered paragraph.
        """
        matches = list(re.finditer(r"(?m)^\s*(\d+)\.\s+", block))
        if not matches:
            return [block]

        parts: List[str] = []
        first = matches[0]
        if first.start() > 0:
            preamble = block[:first.start()].strip()
            if preamble:
                parts.append(preamble)

        for idx, match in enumerate(matches):
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(block)
            segment = block[match.start():end].strip()
            if segment:
                parts.append(segment)

        return parts

    def _extract_options(self, text: str) -> List[str]:
        """
        Extract options from bracketed text.

        FIX: Aggressively strip all bracket characters, semicolons,
        and surrounding whitespace from captured option text.
        Uses a character-by-character strip to handle nested/adjacent
        bracket artifacts like ';]\n['.
        """
        options = []
        pattern = (
            rf'{re.escape(self.option_prefix)}\s*(\d+)[:\s]*'
            rf'(.+?)'
            rf'(?={re.escape(self.option_prefix)}\s*\d+|$)'
        )
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            option_text = match.group(2)

            # Strip bracket artifacts from both ends
            dirty_chars = self.bracket_open + self.bracket_close + "; \t\n\r"
            option_text = option_text.strip(dirty_chars)

            # Also remove any remaining lone bracket chars inside
            # that appear at the very start or end after newline splits
            lines = option_text.split("\n")
            cleaned_lines = []
            for line in lines:
                line = line.strip(dirty_chars)
                if line:
                    cleaned_lines.append(line)
            option_text = "\n".join(cleaned_lines)

            if option_text:
                options.append(option_text)
        return options

    def get_full_text(self) -> str:
        """
        Get the full current text with all brackets and options.

        FIX for Issue #3: Use the original number prefix instead of
        unconditionally prepending paragraph_id, which caused
        "1. 1. Decides..." duplication.
        """
        parts = []
        for para in self.paragraphs:
            if para.is_numbered:
                parts.append(f"{para.original_number} {para.text}")
            else:
                parts.append(para.text)
            if para.notes:
                parts.append(f"   [Chair's note: {para.notes}]")
        return "\n\n".join(parts)

    def get_adoption_ready_text(self) -> str:
        """
        Produce a clean endgame text with no brackets.

        Conservative resolution strategy:
        - If a paragraph is only competing bracketed options, choose the
          shortest option.
        - If a sentence contains adjacent bracketed alternatives, choose the
          shortest alternative.
        - If a single bracketed clause is only an optional add-on, delete it.
        """
        parts = []
        for para in self.paragraphs:
            clean_text = self._resolve_paragraph_for_adoption(para.text)
            if not clean_text:
                continue
            if para.is_numbered:
                parts.append(f"{para.original_number} {clean_text}")
            else:
                parts.append(clean_text)
        final_text = "\n\n".join(parts).strip()
        return final_text.replace(self.bracket_open, "").replace(self.bracket_close, "")

    def _resolve_paragraph_for_adoption(self, text: str) -> str:
        """Resolve one paragraph into a clean, adoption-ready form."""
        if "[" not in text and "]" not in text:
            return self._normalize_adoption_text(text)

        tokens = self._tokenize_bracket_text(text)
        output_parts: List[str] = []
        idx = 0

        while idx < len(tokens):
            token_type, token_text = tokens[idx]
            if token_type != "bracket":
                output_parts.append(token_text)
                idx += 1
                continue

            cluster_options = [token_text]
            end_idx = idx
            probe = idx + 1
            while (
                probe + 1 < len(tokens)
                and tokens[probe][0] == "text"
                and tokens[probe][1].strip() == ""
                and tokens[probe + 1][0] == "bracket"
            ):
                cluster_options.append(tokens[probe + 1][1])
                end_idx = probe + 1
                probe += 2

            has_meaningful_before = self._has_meaningful_plain_text(tokens[:idx])
            has_meaningful_after = self._has_meaningful_plain_text(tokens[end_idx + 1:])
            structural_before = self._has_only_structural_plain_text(tokens[:idx])
            structural_after = self._has_only_structural_plain_text(tokens[end_idx + 1:])
            line_structural_before = self._has_only_structural_line_prefix(tokens[:idx])

            if not has_meaningful_before and not has_meaningful_after:
                replacement = self._select_conservative_option(cluster_options)
                if replacement:
                    output_parts.append(replacement)
            elif len(cluster_options) > 1:
                replacement = self._select_conservative_option(cluster_options)
                if replacement:
                    output_parts.append(replacement)
            elif (
                (has_meaningful_before and structural_before and not has_meaningful_after)
                or (has_meaningful_after and structural_after and not has_meaningful_before)
            ):
                replacement = self._select_conservative_option(cluster_options)
                if replacement:
                    output_parts.append(replacement)
            else:
                replacement = self._resolve_inline_bracket(token_text)
                if (
                    not replacement
                    and len(cluster_options) == 1
                    and line_structural_before
                    and has_meaningful_after
                    and self._continuation_requires_leading_phrase(
                        tokens[end_idx + 1:]
                    )
                ):
                    replacement = self._keep_required_leading_phrase(token_text)
                if replacement:
                    output_parts.append(replacement)
            # Else: single optional bracketed clause -> drop it.

            idx = end_idx + 1

        return self._normalize_adoption_text("".join(output_parts))

    def _tokenize_bracket_text(self, text: str) -> List[Tuple[str, str]]:
        """Split text into plain-text and top-level bracket tokens."""
        tokens: List[Tuple[str, str]] = []
        current: List[str] = []
        depth = 0
        mode = "text"

        for char in text:
            if char == self.bracket_open:
                if depth == 0:
                    if current:
                        tokens.append((mode, "".join(current)))
                    current = []
                    mode = "bracket"
                else:
                    current.append(char)
                depth += 1
                continue

            if char == self.bracket_close and depth > 0:
                depth -= 1
                if depth == 0:
                    tokens.append((mode, "".join(current)))
                    current = []
                    mode = "text"
                else:
                    current.append(char)
                continue

            current.append(char)

        if current:
            tokens.append((mode, "".join(current)))

        return tokens

    @staticmethod
    def _has_meaningful_plain_text(tokens: List[Tuple[str, str]]) -> bool:
        """Return whether plain text outside bracket clusters carries content."""
        for token_type, token_text in tokens:
            if token_type != "text":
                continue
            if re.sub(r"[\s;,:.\-()]+", "", token_text):
                return True
        return False

    @staticmethod
    def _has_only_structural_plain_text(tokens: List[Tuple[str, str]]) -> bool:
        """Return whether the surrounding plain text is only a list marker."""
        plain_text = "".join(
            token_text for token_type, token_text in tokens if token_type == "text"
        )
        if not plain_text.strip():
            return False

        normalized = re.sub(
            r"\(\s*[a-zA-Z0-9]+\s*\)|\b\d+\.\s*",
            "",
            plain_text,
        )
        normalized = re.sub(r"[\s;,:.\-()]+", "", normalized)
        return not normalized

    @staticmethod
    def _has_only_structural_line_prefix(tokens: List[Tuple[str, str]]) -> bool:
        """Return whether the current line prefix is only a list marker."""
        plain_text = "".join(
            token_text for token_type, token_text in tokens if token_type == "text"
        )
        line_prefix = plain_text.split("\n")[-1]
        if not line_prefix.strip():
            return False

        normalized = re.sub(
            r"\(\s*[a-zA-Z0-9]+\s*\)|\b\d+\.\s*",
            "",
            line_prefix,
        )
        normalized = re.sub(r"[\s;,:.\-()]+", "", normalized)
        return not normalized

    def _select_conservative_option(self, options: List[str]) -> str:
        """Pick the narrowest available option from a bracket cluster."""
        cleaned_options = []
        for option in options:
            cleaned = re.sub(
                rf"^\s*{re.escape(self.option_prefix)}\s*\d+\s*:\s*",
                "",
                option.strip(),
                flags=re.IGNORECASE,
            )
            if self.bracket_open in cleaned or self.bracket_close in cleaned:
                cleaned = self._resolve_paragraph_for_adoption(cleaned)
            cleaned = self._normalize_adoption_text(cleaned)
            if cleaned:
                cleaned_options.append(cleaned)

        if not cleaned_options:
            return ""

        return min(
            cleaned_options,
            key=lambda item: (
                len(item.split()),
                len(item),
                item.count(","),
            ),
        )

    def _resolve_inline_bracket(self, text: str) -> str:
        """
        Resolve a single inline bracket conservatively.

        Keep only short lexical alternatives that preserve grammar.
        Drop additive or expansive modifiers by default.
        """
        cleaned = text.strip()
        if not cleaned:
            return ""

        if self.bracket_open in cleaned or self.bracket_close in cleaned:
            cleaned = self._resolve_paragraph_for_adoption(cleaned)

        cleaned = self._normalize_adoption_text(cleaned)
        if not cleaned:
            return ""

        slash_options = [
            self._normalize_adoption_text(part)
            for part in re.split(r"\s*/\s*", cleaned)
            if part.strip()
        ]
        if len(slash_options) > 1 and all(
            option and len(option.split()) <= 6 for option in slash_options
        ):
            return min(
                slash_options,
                key=lambda item: (len(item.split()), len(item), item.count(",")),
            )

        or_options = [
            self._normalize_adoption_text(part)
            for part in re.split(r"\s+or\s+", cleaned, flags=re.IGNORECASE)
            if part.strip()
        ]
        if len(or_options) == 2 and all(
            option and len(option.split()) <= 4 for option in or_options
        ):
            return min(
                or_options,
                key=lambda item: (len(item.split()), len(item), item.count(",")),
            )

        first_word = cleaned.split()[0].lower()
        additive_markers = {
            "and",
            "or",
            "as",
            "with",
            "including",
            "via",
            "through",
            "for",
        }
        if first_word in additive_markers:
            return ""

        if cleaned.endswith((".", ";")) and len(cleaned.split()) <= 12:
            return cleaned

        if len(cleaned.split()) == 1:
            return ""

        if len(cleaned.split()) <= 2:
            return cleaned

        return ""

    def _keep_required_leading_phrase(self, text: str) -> str:
        """
        Keep a bracketed phrase when removing it would leave a broken clause.

        This is used for sentence-initial or list-item-initial predicate
        phrases such as "Promote ... with" before an object.
        """
        cleaned = text.strip()
        if not cleaned:
            return ""

        if self.bracket_open in cleaned or self.bracket_close in cleaned:
            cleaned = self._resolve_paragraph_for_adoption(cleaned)

        cleaned = re.sub(
            rf"^\s*{re.escape(self.option_prefix)}\s*\d+\s*:\s*",
            "",
            cleaned.strip(),
            flags=re.IGNORECASE,
        )
        cleaned = self._normalize_adoption_text(cleaned)
        if not cleaned:
            return ""

        first_word = cleaned.split()[0].lower()
        if first_word in {"and", "or", "as", "with", "including", "via", "through", "for"}:
            return ""

        return cleaned

    @staticmethod
    def _continuation_requires_leading_phrase(
        tokens: List[Tuple[str, str]]
    ) -> bool:
        """
        Detect whether following text looks like a dependent continuation.

        If the surviving text starts with a lowercase token, it is usually an
        object or modifier rather than a standalone clause, so dropping the
        bracketed lead-in would create a fragment.
        """
        following_text = "".join(
            token_text for token_type, token_text in tokens if token_type == "text"
        ).lstrip()
        if not following_text:
            return False

        first_char = following_text[0]
        if first_char.islower():
            return True

        first_word_match = re.match(r"([A-Za-z][A-Za-z\-]*)", following_text)
        if not first_word_match:
            return False

        return first_word_match.group(1).islower()

    @staticmethod
    def _normalize_adoption_text(text: str) -> str:
        """Clean spacing and punctuation after bracket resolution."""
        text = text.replace("[", "").replace("]", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n\s+", "\n", text)
        text = re.sub(r",\s*;", ";", text)
        text = re.sub(r",\s*\.", ".", text)
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        text = re.sub(r",(?=[^\s\n])", ", ", text)
        text = re.sub(r";(?=[^\s\n(])", "; ", text)
        text = re.sub(r":(?=[^\s\n(])", ": ", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip(" \n\t;") + (";" if text.strip().endswith(";") else "")

    def get_paragraph(self, paragraph_id: int) -> Optional[TextParagraph]:
        """Get a specific paragraph by ID."""
        for para in self.paragraphs:
            if para.paragraph_id == paragraph_id:
                return para
        return None

    def get_disputed_paragraphs(self) -> List[TextParagraph]:
        """Get all paragraphs that are still disputed/bracketed."""
        return [
            p for p in self.paragraphs
            if p.is_bracketed or p.status == "disputed"
        ]

    def get_agreed_paragraphs(self) -> List[TextParagraph]:
        """Get all paragraphs that have been agreed."""
        return [p for p in self.paragraphs if p.status == "agreed"]

    def get_unchanged_paragraph_ids(self) -> List[int]:
        """
        Return original paragraph IDs that never received any amendments.

        This deliberately references the original draft-paragraph registry so
        the result remains stable after full-text chair rewrites.
        """
        source_paragraphs = self.original_paragraphs or self.paragraphs
        return [
            para.paragraph_id
            for para in source_paragraphs
            if not para.amendments
        ]

    def get_unchanged_paragraphs(self) -> List[TextParagraph]:
        """Return original draft paragraphs that never received amendments."""
        source_paragraphs = self.original_paragraphs or self.paragraphs
        return [
            para
            for para in source_paragraphs
            if not para.amendments
        ]

    def add_amendment(
        self,
        agent_id: str,
        paragraph_id: int,
        amendment_type: str,
        original_text: str = "",
        proposed_text: str = "",
        rationale: str = "",
    ) -> TextAmendment:
        """Add an amendment proposal to a paragraph."""
        self.amendment_counter += 1
        amendment = TextAmendment(
            amendment_id=f"AMD-{self.amendment_counter:04d}",
            agent_id=agent_id,
            amendment_type=amendment_type,
            target_paragraph=paragraph_id,
            original_text=original_text,
            proposed_text=proposed_text,
            rationale=rationale,
        )

        para = self.get_paragraph(paragraph_id)
        if para:
            para.amendments.append(amendment)
            original_para = self._get_original_paragraph(paragraph_id)
            if original_para is not None:
                original_para.amendments.append(copy.deepcopy(amendment))
            if para.status == "draft":
                para.status = "discussed"
            # FIX Issue #6: save snapshot on amendment
            self._save_snapshot(
                f"amendment_{amendment.amendment_id}",
                details={
                    "agent": agent_id,
                    "type": amendment_type,
                    "paragraph": paragraph_id,
                },
            )
            logger.debug(
                f"Amendment {amendment.amendment_id} added to paragraph {paragraph_id} "
                f"by {agent_id}: {amendment_type}"
            )
        else:
            logger.warning(f"Paragraph {paragraph_id} not found for amendment.")

        return amendment

    def apply_chair_revision(
        self, paragraph_id: int, new_text: str, note: str = ""
    ):
        """Apply a Chair's revised text to a paragraph."""
        para = self.get_paragraph(paragraph_id)
        if para:
            old_text = para.text
            para.text = new_text
            para.is_bracketed = self.bracket_open in new_text
            para.notes = note
            para.options = self._extract_options(new_text)

            self._save_snapshot(
                f"chair_revision_para_{paragraph_id}",
                details={"old": old_text, "new": new_text},
            )
            logger.info(f"Chair revised paragraph {paragraph_id}")

    def update_full_text(self, new_text: str, source: str = "chair"):
        """
        Replace the entire text with a new version.
        Used when the Chair provides a complete revised text.
        """
        self._save_snapshot(f"full_update_by_{source}")
        self.paragraphs = self._parse_text_to_paragraphs(new_text)
        self._save_snapshot(f"full_update_applied_by_{source}")

    def mark_paragraph_agreed(self, paragraph_id: int):
        """Mark a paragraph as agreed (remove brackets)."""
        para = self.get_paragraph(paragraph_id)
        if para:
            para.text = para.text.replace(self.bracket_open, "").replace(
                self.bracket_close, ""
            )
            para.is_bracketed = False
            para.status = "agreed"
            para.options = []
            # FIX Issue #6: save snapshot
            self._save_snapshot(
                f"paragraph_agreed_{paragraph_id}",
                details={"paragraph": paragraph_id},
            )
            logger.info(f"Paragraph {paragraph_id} marked as agreed.")

    def mark_paragraph_disputed(self, paragraph_id: int):
        """Mark a paragraph as disputed."""
        para = self.get_paragraph(paragraph_id)
        if para:
            para.status = "disputed"
            # FIX Issue #6: save snapshot
            self._save_snapshot(
                f"paragraph_disputed_{paragraph_id}",
                details={"paragraph": paragraph_id},
            )

    def get_disputed_points_summary(self) -> List[str]:
        """Get a summary list of disputed points."""
        disputed = []
        for para in self.paragraphs:
            if para.is_bracketed or para.status == "disputed":
                text_preview = para.text[:100].replace("\n", " ")
                label = para.display_label or (
                    para.original_number.rstrip(".")
                    if para.is_numbered and para.original_number
                    else f"Preamble {para.paragraph_id}"
                )
                if label.lower().startswith("preamble"):
                    reference = label
                else:
                    reference = f"Paragraph {label}"
                disputed.append(
                    f"{reference}: {text_preview}..."
                )
        return disputed

    def calculate_bracket_resolution_rate(self) -> float:
        """
        Calculate what fraction of originally bracketed paragraphs
        have been resolved.
        """
        if not self.history:
            return 0.0

        initial_snapshot = self.history[0]
        originally_bracketed = initial_snapshot.get("num_bracketed", 0)

        if originally_bracketed == 0:
            return 1.0

        currently_still_bracketed = sum(
            1 for p in self.paragraphs if p.is_bracketed
        )
        resolved = originally_bracketed - currently_still_bracketed

        return max(0.0, min(1.0, resolved / originally_bracketed))

    def _save_snapshot(self, label: str, details: Optional[Dict] = None):
        """Save a snapshot of the current text state."""
        snapshot = {
            "label": label,
            "round": len(self.history),
            "text": self.get_full_text(),
            "num_paragraphs": len(self.paragraphs),
            "num_bracketed": sum(1 for p in self.paragraphs if p.is_bracketed),
            "num_agreed": sum(1 for p in self.paragraphs if p.status == "agreed"),
            "details": details or {},
        }
        self.history.append(snapshot)

    def get_text_evolution(self) -> List[Dict[str, Any]]:
        """Get the history of text changes."""
        return self.history

    def _get_original_paragraph(self, paragraph_id: int) -> Optional[TextParagraph]:
        """Return a paragraph from the original draft registry by its ID."""
        for para in self.original_paragraphs:
            if para.paragraph_id == paragraph_id:
                return para
        return None
