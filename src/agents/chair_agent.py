"""
Chair Agent - The facilitator/chair of the negotiation.
Unlike negotiation agents, the chair is neutral and manages the process.
"""

import logging
import re
from typing import Dict, Any, List, Optional

from src.llm.llm_backend import LLMBackend, LLMResponse
from src.llm.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class ChairAgent:
    """
    The Chair/Facilitator agent that manages the negotiation process.
    Synthesizes proposals, proposes compromise text, and manages procedure.
    """

    def __init__(self, llm_backend: LLMBackend, global_config: Dict[str, Any]):
        self.llm = llm_backend
        self.global_config = global_config
        self.system_prompt = PromptTemplates.chair_system_prompt()

        # Chair's state
        self.rounds_chaired = 0
        self.synthesis_history: List[Dict[str, Any]] = []

        # LLM params for chair
        self.temperature = 0.6  # Chair should be somewhat deterministic
        self.max_tokens = 2500

        logger.info("Initialized ChairAgent.")

    def _build_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        """Build messages for LLM call."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def present_paragraph(
        self,
        paragraph_number: int,
        paragraph_text: str,
        paragraph_label: Optional[str] = None,
    ) -> str:
        """Present a paragraph for first reading."""
        display_ref = str(paragraph_label or paragraph_number)
        if not display_ref.lower().startswith(("paragraph ", "preamble ")):
            display_ref = f"Paragraph {display_ref}"
        return (
            f"PROCEDURAL NOTE: We now move to {display_ref}.\n\n"
            f"TEXT PRESENTED:\n{paragraph_text}\n\n"
            f"The floor is open for comments and proposals on this paragraph. "
            f"Parties are invited to propose specific textual amendments."
        )

    def synthesize_round(
        self,
        current_text: str,
        proposals: List[Dict[str, str]],
        round_number: int,
        disputed_points: List[str],
        structure_guidance: str = "",
        preserve_verbatim_paragraphs: Optional[List[str]] = None,
    ) -> str:
        """
        Synthesize all proposals from a round into revised text.
        This is the key function where the Chair proposes compromise language.
        """
        prompt = PromptTemplates.chair_synthesis_prompt(
            current_text=current_text,
            all_proposals=proposals,
            round_number=round_number,
            disputed_points=disputed_points,
            structure_guidance=structure_guidance,
            preserve_verbatim_paragraphs=preserve_verbatim_paragraphs,
        )

        messages = self._build_messages(prompt)
        response = self.llm.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Store synthesis
        self.synthesis_history.append(
            {
                "round": round_number,
                "input_proposals": len(proposals),
                "output": response.content,
            }
        )

        self.rounds_chaired += 1
        return response.content

    def revise_for_adoption(
        self,
        candidate_text: str,
        acceptability_map: Dict[str, Any],
        drafting_issues: List[str],
        agent_positions: Dict[str, str],
        scenario_context: str = "",
        preserve_terms: Optional[List[str]] = None,
        structure_guidance: str = "",
        preserve_verbatim_paragraphs: Optional[List[str]] = None,
    ) -> str:
        """
        Produce one last adoption-oriented clean text.

        This is a narrow endgame pass used after debracketing, when the Chair
        needs to fix drafting defects and make minimal bridging edits before
        final plenary.
        """
        prompt = PromptTemplates.chair_finalization_prompt(
            candidate_text=candidate_text,
            acceptability_map=acceptability_map,
            drafting_issues=drafting_issues,
            agent_positions=agent_positions,
            scenario_context=scenario_context,
            preserve_terms=preserve_terms,
            structure_guidance=structure_guidance,
            preserve_verbatim_paragraphs=preserve_verbatim_paragraphs,
        )

        messages = self._build_messages(prompt)
        response = self.llm.generate(
            messages=messages,
            temperature=0.4,
            max_tokens=self.max_tokens,
        )

        self.synthesis_history.append(
            {
                "round": "pre_plenary_finalization",
                "input_proposals": len(agent_positions),
                "output": response.content,
            }
        )
        return response.content

    def assess_convergence(
        self,
        current_text: str,
        agent_positions: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Assess whether the negotiation is converging toward agreement.
        Returns convergence metrics.
        """
        prompt = f"""As Chair, assess the current state of convergence in this negotiation.

## CURRENT TEXT
{current_text}

## PARTY POSITIONS (most recent)
"""
        for agent_id, position in agent_positions.items():
            prompt += f"\n{agent_id}: {position[:300]}\n"

        prompt += """
## TASK
Analyze the positions and provide:
1. CONVERGENCE_SCORE: 0.0 (total deadlock) to 1.0 (full consensus)
2. RESOLVED_ISSUES: List of issues where agreement exists
3. BLOCKING_ISSUES: List of issues blocking agreement
4. SUGGESTED_STRATEGY: How to move forward

Format your response exactly as:
CONVERGENCE_SCORE: [number]
RESOLVED_ISSUES: [list]
BLOCKING_ISSUES: [list]
SUGGESTED_STRATEGY: [text]
"""

        messages = self._build_messages(prompt)
        response = self.llm.generate(
            messages=messages, temperature=0.3, max_tokens=1000
        )

        # Parse the response
        result = self._parse_convergence_assessment(response.content)
        return result

    def _parse_convergence_assessment(self, text: str) -> Dict[str, Any]:
        """Parse the convergence assessment response."""
        result = {
            "convergence_score": 0.5,
            "resolved_issues": [],
            "blocking_issues": [],
            "suggested_strategy": "",
            "raw_text": text,
        }
        header_pattern = re.compile(
            r"^(CONVERGENCE_SCORE|RESOLVED_ISSUES|BLOCKING_ISSUES|SUGGESTED_STRATEGY):\s*(.*)$",
            re.IGNORECASE,
        )
        collecting_blocking_issues = False

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            header_match = header_pattern.match(line)
            if header_match:
                label = header_match.group(1).upper()
                value = header_match.group(2).strip()
                collecting_blocking_issues = False

                if label == "CONVERGENCE_SCORE":
                    try:
                        score = float(value)
                        score = max(0.0, min(1.0, score))
                        result["convergence_score"] = score
                    except ValueError:
                        pass
                elif label == "RESOLVED_ISSUES":
                    result["resolved_issues"] = [
                        i.strip() for i in value.split(",") if i.strip()
                    ]
                elif label == "BLOCKING_ISSUES":
                    result["blocking_issues"] = [
                        i.strip() for i in value.split(",") if i.strip()
                    ]
                    collecting_blocking_issues = not value
                elif label == "SUGGESTED_STRATEGY":
                    result["suggested_strategy"] = value
                continue

            if collecting_blocking_issues and line.startswith(("-", "*")):
                try:
                    issue = line[1:].strip()
                    if issue:
                        result["blocking_issues"].append(issue)
                except IndexError:
                    pass
            else:
                collecting_blocking_issues = False

        return result

    def present_final_text(self, final_text: str) -> str:
        """Present the final text for adoption in plenary."""
        return (
            f"PROCEDURAL NOTE: We have concluded our informal consultations.\n\n"
            f"The Chair presents the following streamlined clean text for adoption by the Parties:\n\n"
            f"FINAL TEXT:\n{final_text}\n\n"
            f"The floor is open. Parties are invited to indicate whether they can "
            f"accept this text. I remind Parties that this text represents a "
            f"carefully balanced compromise."
        )

    def declare_outcome(
        self, accepted: bool, objections: List[Dict[str, str]]
    ) -> str:
        """Declare the outcome of the negotiation."""
        if accepted:
            return (
                "PROCEDURAL NOTE: Seeing no objections, the text is ADOPTED.\n"
                "The Chair thanks all Parties for their constructive engagement.\n"
                "This decision will be forwarded to the CMA/COP for formal adoption."
            )
        else:
            objection_text = "\n".join(
                f"- {o['agent']}: {o['reason']}" for o in objections
            )
            return (
                f"PROCEDURAL NOTE: The text could NOT be adopted due to objections:\n"
                f"{objection_text}\n\n"
                f"The Chair notes the lack of consensus. The issue will be forwarded "
                f"to the next session with the streamlined Chair text and the "
                f"outstanding political differences noted in the record."
            )
