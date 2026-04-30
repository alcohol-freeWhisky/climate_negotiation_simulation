"""Optional LLM second-pass checks for stance-consistency violations."""

import re
from typing import Any, Dict

from src.llm.llm_backend import LLMBackend


class LLMStanceJudge:
    """Ask an LLM to verify whether a heuristic red-line violation is genuine."""

    def __init__(self, llm_backend: LLMBackend, config: Dict[str, Any]):
        self.llm_backend = llm_backend
        self.config = config

    def verify_violation(
        self,
        agent_id: str,
        agent_display_name: str,
        red_line: str,
        statement_excerpt: str,
        violation_type: str,
    ) -> Dict[str, Any]:
        """Return whether the LLM confirms a genuine red-line breach."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful evaluator of negotiation stance consistency. "
                    "Decide whether the statement genuinely breaches the stated red line. "
                    "Reply with YES or NO first, then one short sentence of reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Assess this possible stance violation.\n\n"
                    f"Agent ID: {agent_id}\n"
                    f"Agent name: {agent_display_name}\n"
                    f"Claimed red line: {red_line}\n"
                    f"Heuristic violation label: {violation_type}\n"
                    f"Statement excerpt: {statement_excerpt}\n\n"
                    "Question: Is this a GENUINE breach of the red line? "
                    "Answer YES or NO. Then give one short sentence of reasoning."
                ),
            },
        ]

        response = self.llm_backend.generate(
            messages=messages,
            temperature=0.1,
            max_tokens=150,
        )
        raw = response.content.strip()
        answer_match = re.search(r"\b(YES|NO)\b", raw, re.IGNORECASE)
        confirmed = bool(answer_match and answer_match.group(1).upper() == "YES")

        reasoning = raw
        if answer_match:
            reasoning = raw[answer_match.end():].strip(" :.-\n\t")
        if not reasoning:
            reasoning = raw

        return {
            "confirmed": confirmed,
            "reasoning": reasoning,
            "raw": raw,
        }
