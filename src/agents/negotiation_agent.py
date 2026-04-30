"""
Negotiation Agent - Full implementation of a climate negotiation agent.
Uses LLM to generate contextually appropriate negotiation responses.
"""

import logging
import re
from typing import Dict, Any, List, Optional

from src.agents.base_agent import BaseAgent
from src.llm.llm_backend import LLMBackend, LLMResponse
from src.llm.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class NegotiationAgent(BaseAgent):
    """
    A fully-featured negotiation agent that represents a UNFCCC negotiating group.
    """

    def __init__(
        self,
        agent_config: Dict[str, Any],
        llm_backend: LLMBackend,
        global_config: Dict[str, Any],
    ):
        super().__init__(agent_config, llm_backend, global_config)

        # Build system prompt
        self.system_prompt = PromptTemplates.agent_system_prompt(agent_config)

        # Store initial system message
        self.memory.set_core_memory("identity", self.display_name)
        self.memory.set_core_memory("group_category", self.group_category)

        logger.info(f"Initialized NegotiationAgent: {self.display_name}")

    def _build_messages(
        self,
        user_prompt: str,
        include_memory: bool = True,
        compact_memory: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Build the messages array for LLM call.
        Includes system prompt, memory context, and the current user prompt.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if include_memory:
            if compact_memory:
                memory_context = self.memory.get_compact_context_for_prompt(
                    n_recent_entries=6,
                    max_chars_per_entry=120,
                    exclude_core_keys=["draft_text", "scenario_context"],
                )
            else:
                memory_context = self.memory.get_context_for_prompt(
                    self.rounds_participated
                )
            if memory_context.strip():
                messages.append(
                    {
                        "role": "system",
                        "content": f"## NEGOTIATION CONTEXT & HISTORY\n{memory_context}",
                    }
                )

        messages.append({"role": "user", "content": user_prompt})

        return messages

    @staticmethod
    def _build_consultation_text_excerpt(
        current_text: str,
        max_chars: int = 2200,
    ) -> str:
        """Prioritize unresolved paragraphs when sharing negotiating text."""
        paragraphs = [p.strip() for p in current_text.split("\n\n") if p.strip()]
        if not paragraphs or len(current_text) <= max_chars:
            return current_text

        preamble = [p for p in paragraphs if not re.match(r"^\d+\.", p)][:3]
        bracketed = [p for p in paragraphs if "[" in p or "]" in p]
        numbered = [p for p in paragraphs if re.match(r"^\d+\.", p)]

        selected: List[str] = []
        for paragraph in preamble + bracketed + numbered[:2]:
            if paragraph not in selected:
                selected.append(paragraph)

        excerpt = "\n\n".join(selected) if selected else current_text[:max_chars]
        if len(excerpt) <= max_chars:
            return excerpt

        trimmed = excerpt[:max_chars].rstrip()
        cutoff = trimmed.rfind("\n\n")
        if cutoff > max_chars // 2:
            trimmed = trimmed[:cutoff]
        return trimmed + "\n\n[Text excerpt truncated for consultation focus.]"

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call LLM and return the response content."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        response: LLMResponse = self.llm.generate(
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
        )

        return response.content

    def _get_scenario_salient_issues(self) -> List[str]:
        """Return scenario-scoped salient issues, if configured."""
        scenario = self.global_config.get("scenario", {})
        constraints = scenario.get("scenario_constraints", {})
        issues = constraints.get("salient_issues", [])
        return issues if isinstance(issues, list) else []

    def _get_scenario_runtime_guidance(self) -> List[str]:
        """Return optional runtime-only guidance from the current scenario."""
        scenario = self.global_config.get("scenario", {})
        briefing_cfg = scenario.get("runtime_briefing", {})
        if not isinstance(briefing_cfg, dict):
            return []

        shared = briefing_cfg.get("shared_guidance", [])
        per_agent = briefing_cfg.get("per_agent_guidance", {}).get(self.agent_id, [])

        guidance: List[str] = []
        for item in list(shared or []) + list(per_agent or []):
            if isinstance(item, str) and item.strip():
                guidance.append(item.strip())
        return guidance

    def _build_agenda_focus(
        self,
        *context_parts: Optional[str],
        max_issues: int = 4,
    ) -> str:
        """Build a focused brief from the agent's general charter."""
        context_text = "\n".join(part for part in context_parts if part)
        return self.get_agenda_focus_summary(
            context_text=context_text,
            salient_issues=self._get_scenario_salient_issues(),
            max_issues=max_issues,
        )

    def _build_runtime_briefing(
        self,
        *context_parts: Optional[str],
        disputed_points: Optional[List[str]] = None,
        max_issues: int = 3,
    ) -> str:
        """Generate a scenario-specific issue brief without changing base config."""
        context_text = "\n".join(part for part in context_parts if part)
        return self.build_runtime_briefing(
            context_text=context_text,
            salient_issues=self._get_scenario_salient_issues(),
            disputed_points=disputed_points,
            scenario_guidance=self._get_scenario_runtime_guidance(),
            max_issues=max_issues,
        )

    def generate_opening_statement(
        self, scenario_context: str, draft_text: str
    ) -> str:
        """Generate opening statement for the negotiation."""
        agenda_focus = self._build_agenda_focus(scenario_context, draft_text)
        issue_briefing = self._build_runtime_briefing(
            scenario_context,
            draft_text,
            disputed_points=self.global_config.get("negotiation", {})
            .get("phases", {})
            .get("informal_consultations", {})
            .get("key_dispute_points", []),
        )
        prompt = PromptTemplates.opening_statement_prompt(
            agent_name=self.display_name,
            scenario_context=scenario_context,
            draft_text=draft_text,
            agenda_focus=agenda_focus,
            scenario_briefing=issue_briefing,
        )

        messages = self._build_messages(prompt, include_memory=False)

        # Respect per-phase token cap from config
        phase_max_tokens = (
            self.global_config.get("negotiation", {})
            .get("phases", {})
            .get("opening_statements", {})
            .get("max_tokens_per_agent", self.max_tokens)
        )

        response = self._call_llm(messages, max_tokens=phase_max_tokens)

        self.memory.add_statement(
            round_number=0,
            phase="opening_statements",
            agent_id=self.agent_id,
            content=response,
            importance=0.8,
            tags=["opening_statement"],
        )

        self.increment_round()
        logger.info(f"{self.display_name} delivered opening statement.")
        return response

    def generate_first_reading_response(
        self,
        paragraph_number: int,
        paragraph_text: str,
        other_proposals: List[str],
        scenario_context: str,
        paragraph_label: Optional[str] = None,
    ) -> str:
        """Respond to a paragraph during first reading."""
        display_ref = str(paragraph_label or paragraph_number)
        if not display_ref.lower().startswith(("paragraph ", "preamble ")):
            display_ref = f"Paragraph {display_ref}"
        agenda_focus = self._build_agenda_focus(
            scenario_context,
            paragraph_text,
            "\n".join(other_proposals),
        )
        issue_briefing = self._build_runtime_briefing(
            scenario_context,
            paragraph_text,
            "\n".join(other_proposals),
            disputed_points=[display_ref],
        )
        prompt = PromptTemplates.first_reading_prompt(
            agent_name=self.display_name,
            paragraph_number=paragraph_number,
            paragraph_label=display_ref,
            paragraph_text=paragraph_text,
            other_proposals=other_proposals,
            scenario_context=scenario_context,
            agenda_focus=agenda_focus,
            scenario_briefing=issue_briefing,
        )

        messages = self._build_messages(prompt)

        phase_max_tokens = (
            self.global_config.get("negotiation", {})
            .get("phases", {})
            .get("first_reading", {})
            .get("max_tokens_per_amendment", self.max_tokens)
        )

        response = self._call_llm(messages, max_tokens=phase_max_tokens)

        self.memory.add_statement(
            round_number=paragraph_number,
            phase="first_reading",
            agent_id=self.agent_id,
            content=f"[{display_ref}] {response}",
            importance=0.6,
            tags=["first_reading", f"para_{paragraph_number}"],
        )

        self.increment_round()
        return response

    def generate_consultation_response(
        self,
        current_text: str,
        disputed_points: List[str],
        round_number: int,
        max_rounds: int,
        scenario_context: str,
        targeted_focus: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate response during informal consultations."""
        stance_reminder = None
        if self.needs_stance_reinforcement():
            stance_reminder = self.get_stance_summary()
            logger.debug(
                f"Injecting stance reminder for {self.display_name} "
                f"at round {round_number}"
            )

        recent_history = self.memory.get_recent_history_text_limited(
            n_entries=8,
            max_chars_per_entry=140,
        )
        consultation_text = self._build_consultation_text_excerpt(current_text)
        agenda_focus = self._build_agenda_focus(
            scenario_context,
            consultation_text,
            "\n".join(disputed_points),
            recent_history,
        )
        issue_briefing = self._build_runtime_briefing(
            scenario_context,
            consultation_text,
            recent_history,
            disputed_points=disputed_points,
        )

        prompt = PromptTemplates.informal_consultation_prompt(
            agent_name=self.display_name,
            current_text=consultation_text,
            disputed_points=disputed_points[:5],
            round_number=round_number,
            max_rounds=max_rounds,
            recent_history=recent_history,
            scenario_context=scenario_context,
            agenda_focus=agenda_focus,
            scenario_briefing=issue_briefing,
            stance_reminder=stance_reminder,
            targeted_focus=targeted_focus,
        )

        messages = self._build_messages(prompt, compact_memory=True)

        phase_max_tokens = (
            self.global_config.get("negotiation", {})
            .get("phases", {})
            .get("informal_consultations", {})
            .get("max_tokens_per_statement", self.max_tokens)
        )

        response = self._call_llm(messages, max_tokens=phase_max_tokens)
        response = self._apply_red_line_critic(
            phase="informal_consultations",
            candidate_response=response,
            negotiation_text=current_text,
            scenario_context=scenario_context,
        )

        self.memory.add_statement(
            round_number=round_number,
            phase="informal_consultations",
            agent_id=self.agent_id,
            content=response,
            importance=0.7,
            tags=["consultation", f"round_{round_number}"],
        )

        self.increment_round()
        return response
    
    def generate_final_plenary_response(
        self, final_text: str, scenario_context: str
    ) -> str:
        """Respond in the final plenary."""
        scenario_guardrails = self._scenario_guardrails_text()
        agenda_focus = self._build_agenda_focus(
            scenario_context,
            final_text,
            scenario_guardrails,
        )
        issue_briefing = self._build_runtime_briefing(
            scenario_context,
            final_text,
            scenario_guardrails,
            disputed_points=["final adoption package"],
        )
        prompt = PromptTemplates.final_plenary_prompt(
            agent_name=self.display_name,
            final_text=final_text,
            scenario_context=scenario_context,
            agenda_focus=agenda_focus,
            scenario_briefing=issue_briefing,
            scenario_guardrails=scenario_guardrails,
        )

        messages = self._build_messages(prompt)

        phase_max_tokens = (
            self.global_config.get("negotiation", {})
            .get("phases", {})
            .get("final_plenary", {})
            .get("max_tokens_per_statement", self.max_tokens)
        )

        response = self._call_llm(messages, max_tokens=phase_max_tokens)
        response = self._apply_red_line_critic(
            phase="final_plenary",
            candidate_response=response,
            negotiation_text=final_text,
            scenario_context=scenario_context,
        )

        self.memory.add_statement(
            round_number=1,
            phase="final_plenary",
            agent_id=self.agent_id,
            content=response,
            importance=0.9,
            tags=["final_plenary"],
        )

        self.increment_round()
        return response

    def _scenario_guardrails_text(self) -> str:
        """Return scenario-specific guardrails for this agent, if configured."""
        scenario = self.global_config.get("scenario", {})
        constraints = scenario.get("scenario_constraints", {})

        blocking = (
            constraints.get("agent_blocking_conditions", {})
            .get(self.agent_id, [])
        )
        acceptance = (
            constraints.get("agent_acceptance_conditions", {})
            .get(self.agent_id, [])
        )
        preserve_terms = constraints.get("must_preserve_terms", [])

        lines: List[str] = []
        if blocking:
            lines.append("Blocking conditions:")
            lines.extend(f"- {condition}" for condition in blocking)
        if acceptance:
            lines.append("Acceptance conditions:")
            lines.extend(f"- {condition}" for condition in acceptance)
        if preserve_terms:
            lines.append("Shared terms that should remain in the text when relevant:")
            lines.extend(f"- {term}" for term in preserve_terms[:8])

        return "\n".join(lines)

    def _apply_red_line_critic(
        self,
        phase: str,
        candidate_response: str,
        negotiation_text: str,
        scenario_context: str,
    ) -> str:
        """Optionally revise a response if it violates scenario guardrails."""
        guardrails = self._scenario_guardrails_text()
        if not guardrails.strip():
            return candidate_response

        critic_cfg = (
            self.global_config.get("agent_defaults", {})
            .get("red_line_critic", {})
        )
        if not isinstance(critic_cfg, dict) or not critic_cfg.get("enabled", False):
            return candidate_response

        phases = critic_cfg.get("phases", ["final_plenary"])
        if phase not in phases:
            return candidate_response

        prompt = PromptTemplates.red_line_critic_prompt(
            agent_name=self.display_name,
            phase=phase,
            candidate_response=candidate_response,
            negotiation_text=negotiation_text,
            scenario_context=scenario_context,
            scenario_guardrails=guardrails,
        )
        messages = self._build_messages(prompt)
        critique = self._call_llm(
            messages,
            temperature=0.0,
            max_tokens=critic_cfg.get("max_tokens", 500),
        )

        revised = self._parse_critic_revision(critique, candidate_response)
        if revised != candidate_response:
            logger.info(
                f"Red-line critic revised {self.display_name}'s {phase} response."
            )
        return revised

    @staticmethod
    def _parse_critic_revision(critic_output: str, original_response: str) -> str:
        """Extract a revised response from the red-line critic output."""
        fail = bool(re.search(r"VERDICT:\s*FAIL\b", critic_output, re.IGNORECASE))
        if not fail:
            return original_response

        match = re.search(
            r"REVISED_RESPONSE:\s*(.*)",
            critic_output,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return original_response

        revised = match.group(1).strip()
        return revised or original_response

    def observe_statement(
        self,
        round_number: int,
        phase: str,
        speaker_id: str,
        content: str,
    ):
        """Record another agent's statement in this agent's memory."""
        importance = 0.5
        style = self.config.get("interaction_style", {})
        if speaker_id in style.get("coalition_partners", []):
            importance = 0.7
        elif speaker_id in style.get("typical_adversaries", []):
            importance = 0.8

        self.memory.add_statement(
            round_number=round_number,
            phase=phase,
            agent_id=speaker_id,
            content=content,
            importance=importance,
            tags=["observed", f"from_{speaker_id}"],
        )
