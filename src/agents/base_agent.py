"""
Base Agent - Abstract base class for all negotiation agents.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Iterable, Tuple

from src.memory.negotiation_memory import NegotiationMemory

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for negotiation agents."""

    def __init__(
        self,
        agent_config: Dict[str, Any],
        llm_backend: Any,
        global_config: Dict[str, Any],
    ):
        self.config = agent_config
        self.agent_id = agent_config["agent_id"]
        self.display_name = agent_config["display_name"]
        self.group_category = agent_config.get("group_category", "unknown")
        self.llm = llm_backend
        self.global_config = global_config

        # LLM parameters (with agent-specific overrides)
        llm_overrides = agent_config.get("llm_overrides", {})
        self.temperature = llm_overrides.get(
            "temperature",
            global_config.get("agent_defaults", {}).get("temperature", 0.7),
        )
        self.max_tokens = llm_overrides.get(
            "max_tokens",
            global_config.get("llm", {}).get("max_tokens", 2000),
        )

        # Behavioral parameters
        behavioral = agent_config.get("behavioral_params", {})
        self.stubbornness = behavioral.get("stubbornness", 0.5)
        self.compromise_willingness = behavioral.get("compromise_willingness", 0.5)
        self.coalition_tendency = behavioral.get("coalition_tendency", 0.5)
        self.risk_aversion = behavioral.get("risk_aversion", 0.5)

        # Initialize memory
        # FIX: Use the correct parameter names matching NegotiationMemory.__init__
        self.memory = NegotiationMemory(
            agent_id=self.agent_id,
            working_memory_size=30,
            max_summary_entries=20,
            max_tokens_budget=6000,
        )

        # Track state
        self.rounds_participated = 0
        self.stance_reinforcement_interval = global_config.get(
            "agent_defaults", {}
        ).get("stance_reinforcement_interval", 3)

    @abstractmethod
    def generate_opening_statement(
        self, scenario_context: str, draft_text: str
    ) -> str:
        pass

    @abstractmethod
    def generate_first_reading_response(
        self,
        paragraph_number: int,
        paragraph_text: str,
        other_proposals: List[str],
        scenario_context: str,
        paragraph_label: Optional[str] = None,
    ) -> str:
        pass

    @abstractmethod
    def generate_consultation_response(
        self,
        current_text: str,
        disputed_points: List[str],
        round_number: int,
        max_rounds: int,
        scenario_context: str,
        targeted_focus: Optional[Dict[str, Any]] = None,
    ) -> str:
        pass

    @abstractmethod
    def generate_final_plenary_response(
        self, final_text: str, scenario_context: str
    ) -> str:
        pass

    def needs_stance_reinforcement(self) -> bool:
        """Check if stance reminder should be injected."""
        if self.stance_reinforcement_interval <= 0:
            return False

        return (
            self.rounds_participated > 0
            and self.rounds_participated % self.stance_reinforcement_interval == 0
        )

    def get_stance_summary(self, issue: Optional[str] = None) -> str:
        """Get a summary of this agent's stance for reinforcement."""
        stances = self.config.get("stance", {})
        if issue and issue in stances and isinstance(stances[issue], dict):
            s = stances[issue]
            lines = [f"On {issue}: {s.get('position', '')}"]
            for rl in s.get("red_lines", []):
                lines.append(f"  RED LINE: {rl}")
            return "\n".join(lines)

        lines = []
        for issue_name, s in stances.items():
            if isinstance(s, dict):
                priority = s.get("priority", "medium")
                if priority in ("high", "very_high"):
                    lines.append(f"- {issue_name}: {s.get('position', '')[:100]}")
                    for rl in s.get("red_lines", []):
                        lines.append(f"  RED LINE: {rl}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_issue_name(value: str) -> str:
        """Normalize issue labels for loose matching across scenarios."""
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    @staticmethod
    def _priority_score(priority: str) -> int:
        """Map textual priority labels to a sortable score."""
        return {
            "very_high": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }.get(str(priority).lower(), 1)

    @staticmethod
    def _meaningful_keywords(text: str) -> List[str]:
        """Extract non-trivial keywords for lightweight agenda matching."""
        stop_words = {
            "this", "that", "with", "from", "have", "been", "will", "shall",
            "must", "should", "their", "they", "them", "into", "under",
            "over", "than", "more", "less", "need", "needs", "support",
            "approach", "approaches", "party", "parties", "framework",
            "issue", "issues", "text", "draft", "final", "current",
        }
        return [
            word for word in re.findall(r"[a-z0-9]+", text.lower())
            if len(word) > 3 and word not in stop_words
        ]

    def get_agenda_focus_summary(
        self,
        context_text: str = "",
        salient_issues: Optional[Iterable[str]] = None,
        max_issues: int = 4,
    ) -> str:
        """
        Build a scenario- and paragraph-aware focus brief from the agent's
        general charter.

        The base YAML remains a broad statement of the bloc's worldview. This
        helper narrows that worldview to the issues that appear salient in the
        current scenario or paragraph, so agents do not import unrelated agenda
        fights into every negotiation.
        """
        scored = self._score_relevant_stances(
            context_text=context_text,
            salient_issues=salient_issues,
        )
        if not scored:
            return ""

        selected: List[str] = []
        for score, issue_name, stance in scored:
            if score <= 0:
                continue
            red_lines = stance.get("red_lines", [])
            lines = [
                f"- {issue_name} (priority: {stance.get('priority', 'medium')}): {stance.get('position', '')}"
            ]
            for red_line in red_lines[:2]:
                lines.append(f"  RED LINE: {red_line}")
            selected.append("\n".join(lines))
            if len(selected) >= max_issues:
                break

        if not selected:
            return self.get_stance_summary()

        return "\n".join(selected)

    def build_runtime_briefing(
        self,
        context_text: str = "",
        salient_issues: Optional[Iterable[str]] = None,
        disputed_points: Optional[Iterable[str]] = None,
        scenario_guidance: Optional[Iterable[str]] = None,
        max_issues: int = 3,
    ) -> str:
        """
        Build a scenario-specific issue brief from the agent's general charter.

        This is intentionally runtime-only. It sharpens the bloc's attention for
        the current agenda item without rewriting the base profile itself.
        """
        scored = self._score_relevant_stances(
            context_text=context_text,
            salient_issues=salient_issues,
        )
        if not scored:
            return ""

        top_issues = [
            (issue_name, stance)
            for score, issue_name, stance in scored
            if score > 0
        ][:max_issues]
        if not top_issues:
            return ""

        priority_lines = []
        defend_lines = []
        bridge_lines = []

        for issue_name, stance in top_issues:
            priority = stance.get("priority", "medium")
            position = stance.get("position", "").strip()
            flexibility = stance.get("flexibility", "").strip()
            red_lines = stance.get("red_lines", [])

            if position:
                priority_lines.append(
                    f"- {issue_name}: {position} (priority: {priority})"
                )
            if red_lines:
                defend_lines.append(f"- {issue_name}: {red_lines[0]}")
            if flexibility:
                bridge_lines.append(f"- {issue_name}: {flexibility}")

        sections: List[str] = []
        if priority_lines:
            sections.append(
                "Priority issues for this agenda item:\n" + "\n".join(priority_lines)
            )
        if defend_lines:
            sections.append("Must defend:\n" + "\n".join(defend_lines[:max_issues]))
        if bridge_lines:
            sections.append(
                "Possible landing zone:\n" + "\n".join(bridge_lines[:max_issues])
            )

        dispute_lines = [
            str(point).replace("_", " ").strip()
            for point in (disputed_points or [])
            if str(point).strip()
        ]
        if dispute_lines:
            sections.append(
                "Current text signals to watch:\n"
                + "\n".join(f"- {point}" for point in dispute_lines[:3])
            )

        guidance_lines = [
            str(item).strip()
            for item in (scenario_guidance or [])
            if str(item).strip()
        ]
        if guidance_lines:
            sections.append(
                "Scenario-specific guidance:\n"
                + "\n".join(f"- {line}" for line in guidance_lines[:4])
            )

        return "\n\n".join(sections)

    def _score_relevant_stances(
        self,
        context_text: str = "",
        salient_issues: Optional[Iterable[str]] = None,
    ) -> List[Tuple[int, str, Dict[str, Any]]]:
        """Score stance items for relevance to the current agenda context."""
        stances = self.config.get("stance", {})
        if not stances:
            return []

        normalized_salient = {
            self._normalize_issue_name(issue)
            for issue in (salient_issues or [])
            if issue
        }
        context_lower = context_text.lower()
        scored: List[Tuple[int, str, Dict[str, Any]]] = []

        for issue_name, stance in stances.items():
            if not isinstance(stance, dict):
                continue

            normalized_issue = self._normalize_issue_name(issue_name)
            score = self._priority_score(stance.get("priority", "medium"))

            if normalized_salient and normalized_issue in normalized_salient:
                score += 5

            issue_tokens = set(self._meaningful_keywords(issue_name.replace("_", " ")))
            issue_tokens.update(
                self._meaningful_keywords(stance.get("position", ""))[:6]
            )
            for red_line in stance.get("red_lines", [])[:3]:
                issue_tokens.update(self._meaningful_keywords(red_line)[:4])

            keyword_hits = sum(1 for token in issue_tokens if token in context_lower)
            score += min(keyword_hits, 4)

            if (
                normalized_salient
                and keyword_hits == 0
                and normalized_issue not in normalized_salient
            ):
                # Keep unrelated issues from crowding out agenda-relevant ones.
                score -= 1

            scored.append((score, issue_name, stance))

        scored.sort(
            key=lambda item: (
                item[0],
                self._priority_score(item[2].get("priority", "medium")),
                item[1],
            ),
            reverse=True,
        )
        return scored

    def increment_round(self):
        """Increment round counter."""
        self.rounds_participated += 1
