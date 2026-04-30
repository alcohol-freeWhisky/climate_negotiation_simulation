"""
Negotiation Engine - The core orchestrator that runs the simulation.
Coordinates agents, phases, text management, and turn-taking.
"""

import os
import re
import copy
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import yaml

from src.agents.negotiation_agent import NegotiationAgent
from src.agents.chair_agent import ChairAgent
from src.engine.phase_manager import PhaseManager, NegotiationPhase
from src.engine.text_manager import TextManager
from src.engine.turn_manager import TurnManager
from src.engine.amendment_processor import AmendmentProcessor
from src.llm.llm_backend import LLMBackend
from src.llm.prompt_templates import PromptTemplates
from src.llm.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class NegotiationEngine:
    """
    The main orchestrator for the climate negotiation simulation.
    """

    def __init__(self, config: Dict[str, Any], scenario: Dict[str, Any]):
        self.config = self._merge_scenario_overrides(config, scenario)
        self.scenario = scenario
        self.simulation_name = self.config.get("simulation", {}).get(
            "name", "Unnamed_Simulation"
        )

        # Initialize components
        self.llm = LLMBackend(self.config.get("llm", {}))
        self.phase_manager = PhaseManager(self.config)
        self.text_manager = TextManager(self.config.get("negotiation", {}))
        self.turn_manager = TurnManager(self.config.get("negotiation", {}))
        # FIX: Apply random_seed for reproducibility
        seed = self.config.get("simulation", {}).get("random_seed", None)
        if seed is not None:
            self.turn_manager.set_seed(seed)
            import random
            random.seed(seed)
        self.amendment_processor = AmendmentProcessor()
        self.chair = ChairAgent(self.llm, self.config)

        # Initialize agents
        self.agents: Dict[str, NegotiationAgent] = {}
        self._init_agents(scenario)

        # Simulation state
        self.total_rounds = 0
        self.budgeted_rounds = 0
        self.max_total_rounds = self.config.get("simulation", {}).get(
            "max_total_rounds", 30
        )
        self.results: Dict[str, Any] = {
            "simulation_name": self.simulation_name,
            "scenario": scenario.get("scenario_name", "unknown"),
            "seed": self.config.get("llm", {}).get("seed"),
            "start_time": None,
            "end_time": None,
            "phases": {},
            "final_text": "",
            "outcome": "",
            "agent_stats": {},
        }

        # Full log of all interactions
        self.interaction_log: List[Dict[str, Any]] = []

        logger.info(
            f"NegotiationEngine initialized: {self.simulation_name}, "
            f"{len(self.agents)} agents"
        )

    @staticmethod
    def _merge_scenario_overrides(
        config: Dict[str, Any],
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return a merged runtime config without mutating the caller's config.

        Scenario files can narrow the global experiment settings for a specific
        case, for example Article 6.8 using fewer informal-consultation rounds.
        """
        merged = copy.deepcopy(config)
        merged["scenario"] = copy.deepcopy(scenario)

        phase_overrides = scenario.get("phase_overrides", {})
        if isinstance(phase_overrides, dict):
            phases = (
                merged.setdefault("negotiation", {})
                .setdefault("phases", {})
            )
            for phase_name, overrides in phase_overrides.items():
                if isinstance(overrides, dict):
                    phase_cfg = phases.setdefault(phase_name, {})
                    phase_cfg.update(copy.deepcopy(overrides))

        return merged

    def _init_agents(self, scenario: Dict[str, Any]):
        """Initialize all agents specified in the scenario."""
        active_agent_ids = scenario.get("active_agents", [])

        for agent_id in active_agent_ids:
            config_path = f"config/agents/{agent_id.lower()}.yaml"
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    agent_config = yaml.safe_load(f)
            else:
                logger.warning(
                    f"Agent config not found: {config_path}. Using minimal config."
                )
                agent_config = {
                    "agent_id": agent_id,
                    "display_name": agent_id,
                    "group_category": "unknown",
                    "stance": {},
                    "normative_frame": {"primary_principles": []},
                    "interaction_style": {},
                    "behavioral_params": {},
                }

            agent = NegotiationAgent(
                agent_config=agent_config,
                llm_backend=self.llm,
                global_config=self.config,
            )
            self.agents[agent_id] = agent

        # Set up turn manager
        self.turn_manager.set_agents(list(self.agents.keys()))

    def run(self) -> Dict[str, Any]:
        """
        Run the full negotiation simulation.
        Returns the results dictionary.
        """
        self.results["start_time"] = datetime.now().isoformat()
        logger.info(f"{'='*60}")
        logger.info(f"STARTING SIMULATION: {self.simulation_name}")
        logger.info(f"{'='*60}")

        try:
            # Phase 0: Initialization
            self._run_initialization()

            # Phase 1: Opening Statements
            if NegotiationPhase.OPENING_STATEMENTS in self.phase_manager.active_phases:
                self.phase_manager.advance_phase()
                self._run_opening_statements()

            # Phase 1.5: Coalition Caucus
            if NegotiationPhase.COALITION_CAUCUS in self.phase_manager.active_phases:
                self.phase_manager.advance_phase()
                self._run_coalition_caucus()

            # Phase 2: First Reading
            if NegotiationPhase.FIRST_READING in self.phase_manager.active_phases:
                self.phase_manager.advance_phase()
                self._run_first_reading()

            # Phase 3: Informal Consultations
            if (
                NegotiationPhase.INFORMAL_CONSULTATIONS
                in self.phase_manager.active_phases
            ):
                self.phase_manager.advance_phase()
                self._run_informal_consultations()

            # Phase 4: Final Plenary
            if NegotiationPhase.FINAL_PLENARY in self.phase_manager.active_phases:
                self.phase_manager.advance_phase()
                self._run_final_plenary()

            # Conclude
            self.phase_manager.advance_phase()

        except Exception as e:
            logger.error(f"Simulation failed: {e}", exc_info=True)
            self.results["outcome"] = f"FAILED: {str(e)}"

        self.results["end_time"] = datetime.now().isoformat()
        self.results["final_text"] = self.text_manager.get_full_text()
        self.results["total_rounds"] = self.total_rounds
        self.results["budgeted_rounds"] = self.budgeted_rounds
        self.results["llm_stats"] = self.llm.get_stats()
        self.results["interaction_log"] = self.interaction_log

        # Collect agent stats
        for agent_id, agent in self.agents.items():
            self.results["agent_stats"][agent_id] = {
                "rounds_participated": agent.rounds_participated,
                "concessions_made": len(agent.memory.concessions_made),
                "concessions_received": len(agent.memory.concessions_received),
            }

        logger.info(f"{'='*60}")
        logger.info(f"SIMULATION COMPLETE: {self.results.get('outcome', 'Unknown')}")
        logger.info(f"{'='*60}")

        return self.results

    def _run_initialization(self):
        """Phase 0: Load texts and initialize."""
        logger.info("--- Phase 0: Initialization ---")

        # Load draft text
        draft_path = self.scenario.get("draft_text_path", "")
        if os.path.exists(draft_path):
            with open(draft_path, "r") as f:
                draft_text = f.read()
        else:
            draft_text = self.scenario.get(
                "default_draft_text", "No draft text provided."
            )
            logger.warning(f"Draft text file not found: {draft_path}. Using default.")

        self.text_manager.load_draft_text(draft_text)
        self.scenario_context = self.scenario.get("context", "")
        agent_briefings = self._build_agent_runtime_briefings(
            draft_text=self.text_manager.get_full_text()
        )

        # Store draft text in all agents' core memory
        for agent in self.agents.values():
            agent.memory.set_core_memory(
                "draft_text", self.text_manager.get_full_text()[:2000]
            )
            agent.memory.set_core_memory(
                "scenario_context", self.scenario_context[:1000]
            )
            if agent.agent_id in agent_briefings:
                agent.memory.set_core_memory(
                    "scenario_briefing", agent_briefings[agent.agent_id][:2000]
                )

        self.results["phases"]["initialization"] = {
            "draft_text": self.text_manager.get_full_text(),
            "num_paragraphs": len(self.text_manager.paragraphs),
            "num_bracketed": sum(
                1 for p in self.text_manager.paragraphs if p.is_bracketed
            ),
            "agent_briefings": agent_briefings,
        }

    def _build_agent_runtime_briefings(self, draft_text: str) -> Dict[str, str]:
        """
        Build runtime-only scenario briefings for each agent.

        These briefings refine the current agenda item without altering the
        broad agent YAMLs, so swapping scenarios remains low-friction.
        """
        dispute_points = (
            self.config.get("negotiation", {})
            .get("phases", {})
            .get("informal_consultations", {})
            .get("key_dispute_points", [])
        )
        briefings: Dict[str, str] = {}

        for agent_id, agent in self.agents.items():
            build_brief = getattr(agent, "_build_runtime_briefing", None)
            if not callable(build_brief):
                continue

            briefing = build_brief(
                self.scenario_context,
                draft_text,
                disputed_points=dispute_points,
                max_issues=4,
            )
            if briefing:
                briefings[agent_id] = briefing

        return briefings

    def _run_opening_statements(self):
        """Phase 1: Each agent delivers an opening statement."""
        logger.info("--- Phase 1: Opening Statements ---")
        phase_log = []

        speaking_order = self.turn_manager.get_speaking_order(
            round_number=0, phase="opening_statements"
        )

        for agent_id in speaking_order:
            agent = self.agents[agent_id]
            logger.info(f"Opening statement: {agent.display_name}")

            statement = agent.generate_opening_statement(
                scenario_context=self.scenario_context,
                draft_text=self.text_manager.get_full_text(),
            )

            # All other agents observe
            for other_id, other_agent in self.agents.items():
                if other_id != agent_id:
                    other_agent.observe_statement(
                        round_number=0,
                        phase="opening_statements",
                        speaker_id=agent_id,
                        content=statement,
                    )

            entry = {
                "phase": "opening_statements",
                "round": 0,
                "agent": agent_id,
                "content": statement,
                "timestamp": datetime.now().isoformat(),
            }
            self.interaction_log.append(entry)
            phase_log.append(entry)
            self._record_round_progress("opening_statements")

        self.phase_manager.increment_round()
        self.results["phases"]["opening_statements"] = phase_log

    def _run_coalition_caucus(self):
        """Phase 1.5: Add coalition-alignment notes after opening statements."""
        logger.info("--- Phase 1.5: Coalition Caucus ---")
        phase_log = []
        opening_positions = self._get_latest_agent_positions(
            phase="opening_statements"
        )
        use_llm = (
            self.config.get("negotiation", {})
            .get("phases", {})
            .get("coalition_caucus", {})
            .get("use_llm", False)
        )

        for cluster in self._build_coalition_clusters():
            if len(cluster) < 2:
                continue

            for agent_id in cluster:
                note = self._build_coalition_alignment_note(
                    agent_id=agent_id,
                    cluster=cluster,
                    opening_positions=opening_positions,
                    use_llm=use_llm,
                )
                if not note:
                    continue

                agent = self.agents[agent_id]
                agent.memory.add_statement(
                    round_number=0,
                    phase="coalition_caucus",
                    agent_id=agent_id,
                    content=note,
                    importance=0.7,
                    tags=["coalition_caucus", "coalition_alignment"],
                )

                entry = {
                    "phase": "coalition_caucus",
                    "round": 0,
                    "agent": agent_id,
                    "cluster": list(cluster),
                    "content": note,
                    "timestamp": datetime.now().isoformat(),
                }
                self.interaction_log.append(entry)
                phase_log.append(entry)

        self.phase_manager.increment_round()
        self._record_round_progress("coalition_caucus")
        self.results["phases"]["coalition_caucus"] = phase_log

    def _build_coalition_clusters(self) -> List[List[str]]:
        """Group agents by coalition-partner links using transitive closure."""
        agent_ids = sorted(self.agents.keys())
        adjacency = {agent_id: set() for agent_id in agent_ids}

        for agent_id, agent in self.agents.items():
            partners = (
                agent.config.get("interaction_style", {}).get(
                    "coalition_partners",
                    [],
                )
            )
            if not isinstance(partners, list):
                continue

            for partner_id in partners:
                partner = str(partner_id).strip()
                if partner == agent_id or partner not in adjacency:
                    continue
                adjacency[agent_id].add(partner)
                adjacency[partner].add(agent_id)

        clusters: List[List[str]] = []
        visited = set()
        for agent_id in agent_ids:
            if agent_id in visited:
                continue

            stack = [agent_id]
            cluster = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                neighbors = sorted(adjacency[current] - visited, reverse=True)
                stack.extend(neighbors)

            clusters.append(sorted(cluster))

        return clusters

    def _build_coalition_alignment_note(
        self,
        agent_id: str,
        cluster: List[str],
        opening_positions: Dict[str, str],
        use_llm: bool = False,
    ) -> str:
        """Generate an internal coalition-alignment note for one agent."""
        allies = [member for member in cluster if member != agent_id]
        if not allies:
            return ""

        if use_llm:
            note = self._build_coalition_alignment_note_with_llm(
                agent_id=agent_id,
                allies=allies,
                opening_positions=opening_positions,
            )
            if note:
                return note

        return self._build_coalition_alignment_note_in_memory(
            agent_id=agent_id,
            allies=allies,
            opening_positions=opening_positions,
        )

    def _build_coalition_alignment_note_with_llm(
        self,
        agent_id: str,
        allies: List[str],
        opening_positions: Dict[str, str],
    ) -> str:
        """Optionally distill a coalition note with one short LLM call."""
        agent = self.agents[agent_id]
        build_messages = getattr(agent, "_build_messages", None)
        call_llm = getattr(agent, "_call_llm", None)
        if not callable(build_messages) or not callable(call_llm):
            return ""

        allies_statements = [
            {
                "agent": ally_id,
                "content": opening_positions.get(ally_id, ""),
            }
            for ally_id in allies
            if opening_positions.get(ally_id, "").strip()
        ]
        if not allies_statements:
            return ""

        prompt = PromptTemplates.coalition_caucus_alignment_prompt(
            agent.config,
            allies_statements,
        )
        messages = build_messages(
            prompt,
            include_memory=True,
            compact_memory=True,
        )
        response = call_llm(
            messages,
            max_tokens=min(getattr(agent, "max_tokens", 200), 200),
        )
        response = str(response).strip()
        if not response:
            return ""
        if response.startswith("COALITION_ALIGNMENT_NOTE:"):
            return response
        return f"COALITION_ALIGNMENT_NOTE: {response}"

    def _build_coalition_alignment_note_in_memory(
        self,
        agent_id: str,
        allies: List[str],
        opening_positions: Dict[str, str],
    ) -> str:
        """Build a deterministic coalition note from opening statements only."""
        my_statement = opening_positions.get(agent_id, "")
        coalition_statements = [
            opening_positions.get(member, "")
            for member in [agent_id] + allies
            if opening_positions.get(member, "").strip()
        ]
        ally_statements = [
            opening_positions.get(ally_id, "")
            for ally_id in allies
            if opening_positions.get(ally_id, "").strip()
        ]
        if not ally_statements:
            return ""

        shared_priorities = self._collect_keyword_signals(
            coalition_statements,
            min_mentions=max(2, (len(coalition_statements) + 1) // 2),
        )
        my_keywords = set(self._extract_alignment_keywords(my_statement))
        ally_keywords = self._collect_keyword_signals(
            ally_statements,
            min_mentions=1,
        )
        ally_emphasis = [
            keyword for keyword in ally_keywords if keyword not in my_keywords
        ][:4]

        if shared_priorities:
            alignment_text = (
                "The coalition broadly aligns with your opening on "
                f"{', '.join(shared_priorities)}."
            )
        elif ally_emphasis:
            alignment_text = (
                "Your allies are closest to your opening on procedural direction, "
                f"with extra emphasis on {', '.join(ally_emphasis)}."
            )
        else:
            alignment_text = (
                "Your allies' opening statements are politically adjacent to your own, "
                "but their priorities are expressed with different wording."
            )

        details = []
        for ally_id in allies:
            ally_statement = opening_positions.get(ally_id, "").strip()
            if not ally_statement:
                continue
            details.append(
                f"{ally_id}: {self._truncate_for_note(ally_statement, max_chars=140)}"
            )

        detail_text = (
            "; ".join(details)
            if details
            else "No allied opening statements were logged."
        )
        note_parts = [
            f"COALITION_ALIGNMENT_NOTE: Coalition partners in this caucus: {', '.join(allies)}.",
            alignment_text,
            f"Allies' opening positions: {detail_text}",
        ]
        if ally_emphasis:
            note_parts.append(
                "Track these additional allied emphases in first reading: "
                f"{', '.join(ally_emphasis)}."
            )
        return " ".join(part for part in note_parts if part).strip()

    @staticmethod
    def _truncate_for_note(text: str, max_chars: int = 160) -> str:
        """Truncate text for compact internal notes without splitting badly."""
        collapsed = re.sub(r"\s+", " ", str(text)).strip()
        if len(collapsed) <= max_chars:
            return collapsed
        truncated = collapsed[:max_chars].rsplit(" ", 1)[0].rstrip(",;:")
        return f"{truncated}..."

    @staticmethod
    def _extract_alignment_keywords(text: str) -> List[str]:
        """Extract lightweight thematic keywords from a statement."""
        stop_words = {
            "this", "that", "with", "from", "have", "been", "will", "shall",
            "must", "should", "their", "they", "them", "into", "under",
            "over", "than", "more", "less", "need", "needs", "support",
            "approach", "approaches", "party", "parties", "framework",
            "group", "chair", "agenda", "current", "draft", "text",
            "opening", "statement", "statements", "would", "could",
            "where", "which", "about", "remain", "ensuring", "ensure",
            "through", "those", "these", "being", "while", "such",
        }
        keywords = []
        for word in re.findall(r"[a-z0-9_'-]+", str(text).lower()):
            normalized = word.strip("_'-")
            if len(normalized) < 4 or normalized in stop_words:
                continue
            keywords.append(normalized)
        return list(dict.fromkeys(keywords))

    def _collect_keyword_signals(
        self,
        statements: List[str],
        min_mentions: int = 1,
        max_terms: int = 4,
    ) -> List[str]:
        """Collect the most repeated keywords across a list of statements."""
        frequencies: Dict[str, int] = {}
        for statement in statements:
            for keyword in set(self._extract_alignment_keywords(statement)):
                frequencies[keyword] = frequencies.get(keyword, 0) + 1

        ranked = sorted(
            frequencies.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            keyword
            for keyword, count in ranked
            if count >= min_mentions
        ][:max_terms]

    def _run_first_reading(self):
        """
        Phase 2: Go through text paragraph by paragraph.

        In realistic COP-style process, first reading is mainly for tabling
        additions, deletions, and modifications paragraph by paragraph. The
        actual narrowing of options belongs in consultations, not in an early
        chair clean-text pass.

        We therefore keep first reading as a dispute-mapping stage whenever
        informal consultations are enabled. Only if consultations are disabled
        do we fall back to a chair synthesis that updates the text directly.
        """
        logger.info("--- Phase 2: First Reading ---")
        phase_log = []

        # Collect all proposals across all paragraphs for chair synthesis
        all_round_proposals: List[Dict[str, str]] = []

        for para in self.text_manager.paragraphs:
            display_label = self._paragraph_display_label(para)
            logger.info(f"First reading: {display_label}")

            # Chair presents the paragraph
            chair_presentation = self.chair.present_paragraph(
                para.paragraph_id,
                para.text,
                paragraph_label=display_label,
            )
            self.interaction_log.append(
                {
                    "phase": "first_reading",
                    "round": para.paragraph_id,
                    "agent": "CHAIR",
                    "content": chair_presentation,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Each agent responds
            speaking_order = self.turn_manager.get_speaking_order(
                round_number=para.paragraph_id, phase="first_reading"
            )
            paragraph_proposals = []
            paragraph_actions: List[str] = []

            for agent_id in speaking_order:
                agent = self.agents[agent_id]

                response = agent.generate_first_reading_response(
                    paragraph_number=para.paragraph_id,
                    paragraph_text=para.text,
                    other_proposals=paragraph_proposals,
                    scenario_context=self.scenario_context,
                    paragraph_label=display_label,
                )

                # Parse amendments
                amendments = self.amendment_processor.parse_response(response)
                primary_action = self.amendment_processor.get_primary_action(response)
                paragraph_actions.append(primary_action)

                # Apply amendments with per-agent cap (single loop, no duplication)
                max_amendments = (
                    self.config.get("negotiation", {})
                    .get("phases", {})
                    .get("first_reading", {})
                    .get("max_amendments_per_agent", 999)
                )
                amendment_count = 0
                for amd in amendments:
                    if amd.action in ("add", "delete", "modify"):
                        if amendment_count >= max_amendments:
                            logger.debug(
                                f"Agent {agent_id} hit max amendments "
                                f"({max_amendments}) for paragraph "
                                f"{para.paragraph_id}"
                            )
                            break
                        self.text_manager.add_amendment(
                            agent_id=agent_id,
                            paragraph_id=para.paragraph_id,
                            amendment_type=amd.action,
                            original_text=amd.original_text,
                            proposed_text=amd.proposed_text,
                        )
                        amendment_count += 1

                paragraph_proposals.append(
                    f"{agent.display_name}: {response[:200]}"
                )
                all_round_proposals.append(
                    {"agent": agent_id, "content": response}
                )

                # Others observe
                for other_id, other_agent in self.agents.items():
                    if other_id != agent_id:
                        other_agent.observe_statement(
                            round_number=para.paragraph_id,
                            phase="first_reading",
                            speaker_id=agent_id,
                            content=response,
                        )

                entry = {
                    "phase": "first_reading",
                    "round": para.paragraph_id,
                    "agent": agent_id,
                    "content": response,
                    "amendments": [
                        {
                            "action": a.action,
                            "text": a.proposed_text or a.original_text,
                        }
                        for a in amendments
                    ],
                    "timestamp": datetime.now().isoformat(),
                }
                self.interaction_log.append(entry)
                phase_log.append(entry)

            # Mark paragraph as discussed
            if self._first_reading_has_live_disagreement(paragraph_actions, para.amendments):
                para.status = "disputed"
            else:
                para.status = "agreed"

            self._record_round_progress("first_reading")
            self.phase_manager.increment_round()

        # ---------------------------------------------------------------
        # FIX Issue #4: Chair synthesis after first reading so that
        # amendments are applied to the text even when informal
        # consultations are disabled.
        # ---------------------------------------------------------------
        disputed_points = self._get_disputed_points()
        self.phase_manager.set_phase_data(
            "first_reading_disputed_points",
            disputed_points,
        )
        consultations_enabled = self._phase_enabled("informal_consultations")
        if all_round_proposals and disputed_points:
            if consultations_enabled:
                chair_note = (
                    "PROCEDURAL NOTE: Positions and proposed textual changes "
                    "have been recorded. Outstanding paragraphs will be taken "
                    "up in informal consultations; no final compromise text is "
                    "adopted at first reading."
                )
                self.interaction_log.append(
                    {
                        "phase": "first_reading",
                        "round": "synthesis",
                        "agent": "CHAIR",
                        "content": chair_note,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            else:
                logger.info("Chair synthesizing first-reading amendments...")
                chair_synthesis = self.chair.synthesize_round(
                    current_text=self.text_manager.get_full_text(),
                    proposals=all_round_proposals,
                    round_number=0,
                    disputed_points=disputed_points,
                    preserve_verbatim_paragraphs=(
                        self._get_preserve_verbatim_paragraph_texts()
                    ),
                )
                revised_text = self._extract_revised_text(chair_synthesis)
                if revised_text:
                    self.text_manager.update_full_text(
                        revised_text, source="chair_first_reading"
                    )
                self.interaction_log.append(
                    {
                        "phase": "first_reading",
                        "round": "synthesis",
                        "agent": "CHAIR",
                        "content": chair_synthesis,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        self.phase_manager.set_phase_data("all_paragraphs_read", True)
        self.results["phases"]["first_reading"] = phase_log

    def _run_informal_consultations(self):
        """Phase 3: Multi-round informal consultations on disputed text."""
        logger.info("--- Phase 3: Informal Consultations ---")
        phase_log = []
        max_rounds = self.phase_manager.get_max_rounds()
        patience = (
            self.config.get("negotiation", {})
            .get("phases", {})
            .get("informal_consultations", {})
            .get("patience", 3)
        )
        rounds_without_progress = 0
        last_convergence = 0.0
        rounds_completed = 0  # FIX: Initialize before loop

        for round_num in range(1, max_rounds + 1):
            logger.info(
                f"Informal consultation round {round_num}/{max_rounds}"
            )
            rounds_completed = round_num

            current_text = self.text_manager.get_full_text()
            disputed_points = self._get_disputed_points()

            if not disputed_points:
                logger.info("No more disputed points. Moving to final plenary.")
                break

            speaking_order = self.turn_manager.get_speaking_order(
                round_number=round_num, phase="consultation"
            )
            targeted_focus = self._select_targeted_consultation_focus(
                round_number=round_num,
                max_rounds=max_rounds,
                rounds_without_progress=rounds_without_progress,
            )

            round_proposals = []

            for agent_id in speaking_order:
                agent = self.agents[agent_id]

                response = agent.generate_consultation_response(
                    current_text=current_text,
                    disputed_points=disputed_points,
                    round_number=round_num,
                    max_rounds=max_rounds,
                    scenario_context=self.scenario_context,
                    targeted_focus=targeted_focus,
                )

                amendments = self.amendment_processor.parse_response(response)

                action = amendments[0].action if amendments else "argue"
                self.turn_manager.record_speaking(agent_id, round_num, action)

                round_proposals.append(
                    {"agent": agent_id, "content": response}
                )

                for other_id, other_agent in self.agents.items():
                    if other_id != agent_id:
                        other_agent.observe_statement(
                            round_number=round_num,
                            phase="informal_consultations",
                            speaker_id=agent_id,
                            content=response,
                        )

                entry = {
                    "phase": "informal_consultations",
                    "round": round_num,
                    "agent": agent_id,
                    "content": response,
                    "amendments": [
                        {
                            "action": a.action,
                            "text": a.proposed_text or a.reasoning[:100],
                        }
                        for a in amendments
                    ],
                    "timestamp": datetime.now().isoformat(),
                }
                self.interaction_log.append(entry)
                phase_log.append(entry)

            # Chair synthesizes proposals
            logger.info(f"Chair synthesizing round {round_num} proposals...")
            chair_synthesis = self.chair.synthesize_round(
                current_text=current_text,
                proposals=round_proposals,
                round_number=round_num,
                disputed_points=disputed_points,
                structure_guidance=self._build_structure_guidance(current_text),
                preserve_verbatim_paragraphs=(
                    self._get_preserve_verbatim_paragraph_texts()
                ),
            )

            revised_text = self._extract_revised_text(chair_synthesis)
            if revised_text:
                revised_text = self._stabilize_revised_text_structure(
                    current_text=current_text,
                    revised_text=revised_text,
                )
            agent_positions = {
                p["agent"]: p["content"] for p in round_proposals
            }
            candidate_text = revised_text or current_text
            round_acceptability_map = self._build_endgame_acceptability_map(
                agent_positions,
                candidate_text=candidate_text,
            )
            likely_objector_count = len(
                round_acceptability_map.get("likely_object", [])
            )
            substantive_blocker_count = self._count_substantive_paragraph_blockers(
                round_acceptability_map.get("paragraph_blockers", {})
            )
            if likely_objector_count and substantive_blocker_count == 0:
                substantive_blocker_count = 1
            convergence = self.chair.assess_convergence(
                current_text=candidate_text,
                agent_positions=agent_positions,
            )
            convergence_score = convergence.get("convergence_score", 0.0)
            preserved_brackets = False

            if revised_text:
                if self._should_preserve_brackets(
                    current_text=current_text,
                    revised_text=revised_text,
                    round_proposals=round_proposals,
                    convergence=convergence,
                ):
                    preserved_brackets = True
                    logger.info(
                        "Preserving bracketed text because clean chair package "
                        "still appears politically blocked."
                    )
                else:
                    self.text_manager.update_full_text(revised_text, source="chair")

            if preserved_brackets:
                threshold = (
                    self.config.get("negotiation", {})
                    .get("phases", {})
                    .get("informal_consultations", {})
                    .get("convergence_threshold", 0.8)
                )
                convergence_score = min(convergence_score, max(0.0, threshold - 0.01))

            self.interaction_log.append(
                {
                    "phase": "informal_consultations",
                    "round": round_num,
                    "agent": "CHAIR",
                    "content": chair_synthesis,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            phase_log.append(
                {
                    "phase": "informal_consultations",
                    "round": round_num,
                    "agent": "CHAIR",
                    "content": chair_synthesis,
                }
            )

            logger.info(
                f"Round {round_num} convergence: {convergence_score:.2f}, "
                f"Blocking issues: {convergence.get('blocking_issues', [])}, "
                f"Likely objectors: {likely_objector_count}, "
                f"operative blocker clusters: {substantive_blocker_count}"
            )

            if convergence_score <= last_convergence:
                rounds_without_progress += 1
            else:
                rounds_without_progress = 0
            last_convergence = convergence_score

            if rounds_without_progress >= patience:
                logger.info(
                    f"No progress for {patience} rounds. Chair escalating."
                )
                self._chair_escalation(round_num, disputed_points)
                rounds_without_progress = 0

            self.phase_manager.increment_round()
            self._record_round_progress("informal_consultations")
            self.phase_manager.set_phase_data(
                "latest_consultation_acceptability_map",
                round_acceptability_map,
            )
            self.phase_manager.set_phase_data(
                "consultation_rounds_without_progress",
                rounds_without_progress,
            )

            if self.phase_manager.should_advance(
                convergence_score,
                blocker_count=likely_objector_count,
                open_paragraphs=substantive_blocker_count,
            ):
                logger.info(
                    f"Phase advancement triggered at round {round_num}."
                )
                break

            if self.budgeted_rounds >= self.max_total_rounds:
                logger.warning("Max total rounds reached.")
                break

            if self.turn_manager.check_all_passed():
                logger.info("All agents passing. Moving forward.")
                break

        self.results["phases"]["informal_consultations"] = {
            "rounds_completed": rounds_completed,  # FIX: Always defined
            "final_convergence": last_convergence,
            "latest_acceptability_map": self.phase_manager.get_phase_data(
                "latest_consultation_acceptability_map",
                {},
            ),
            "log": phase_log,
        }

    def _select_targeted_consultation_focus(
        self,
        round_number: int,
        max_rounds: int,
        rounds_without_progress: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Narrow late-stage consultations to the most blocked operative paragraph.

        Real negotiations often move from broad issue-trading to very targeted
        drafting huddles late in the process. This helper keeps that behavior
        generic by using the evolving acceptability map rather than any
        scenario-specific clause template.
        """
        latest_map = self.phase_manager.get_phase_data(
            "latest_consultation_acceptability_map",
            {},
        )
        paragraph_blockers = latest_map.get("paragraph_blockers", {})
        overloaded_paragraphs = latest_map.get("overloaded_paragraphs", {})
        if not paragraph_blockers:
            return None

        late_round_start = max(2, max_rounds - 1)
        if round_number < late_round_start and rounds_without_progress < 1:
            return None

        likely_objectors = set(latest_map.get("likely_object", []))
        conditional_acceptors = set(latest_map.get("conditional_accept", []))

        candidates: List[Tuple[Tuple[int, int, int, int, int], str, Dict[str, Any]]] = []
        for paragraph_ref, details in paragraph_blockers.items():
            if paragraph_ref != "general" and not re.match(
                r"^\d+(?:\([a-z]\))?$",
                str(paragraph_ref),
            ):
                continue

            hard_objectors = list(details.get("objectors", []))
            if likely_objectors:
                hard_objectors = [
                    agent_id
                    for agent_id in hard_objectors
                    if agent_id in likely_objectors
                ]
            soft_objectors = [
                agent_id
                for agent_id in details.get("conditional_acceptors", [])
                if agent_id in conditional_acceptors
            ]
            themes = [
                theme
                for theme in details.get("themes", [])
                if theme != "general_acceptability"
            ]
            overloaded = paragraph_ref in overloaded_paragraphs
            if not hard_objectors and not soft_objectors:
                continue

            score = (
                len(hard_objectors),
                0 if overloaded else 1,
                len(soft_objectors),
                len(themes),
                0 if paragraph_ref == "general" else 1,
            )
            candidates.append((score, paragraph_ref, details))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        _, paragraph_ref, details = candidates[0]
        objectors = list(details.get("objectors", []))
        if likely_objectors:
            objectors = [
                agent_id
                for agent_id in objectors
                if agent_id in likely_objectors
            ]
        conditional_parties = [
            agent_id
            for agent_id in details.get("conditional_acceptors", [])
            if agent_id in conditional_acceptors
        ]
        supporters = [
            agent_id
            for agent_id in latest_map.get("likely_accept", [])
            if agent_id not in objectors and agent_id not in conditional_parties
        ]
        if not supporters:
            supporters = [
                agent_id
                for agent_id in latest_map.get("uncertain", [])
                if agent_id not in objectors and agent_id not in conditional_parties
            ][:3]

        return {
            "paragraph_ref": paragraph_ref,
            "objectors": objectors,
            "conditional_acceptors": conditional_parties,
            "supporters": supporters[:4],
            "themes": details.get("themes", []),
            "overloaded": paragraph_ref in overloaded_paragraphs,
            "overload_details": overloaded_paragraphs.get(paragraph_ref, {}),
            "resolution_mode": (
                overloaded_paragraphs.get(paragraph_ref, {}).get(
                    "recommended_resolution",
                    "merge",
                )
                if paragraph_ref in overloaded_paragraphs
                else "merge"
            ),
        }

    def _get_disputed_points(self) -> List[str]:
        """
        Return current text disputes plus scenario-specific issue labels.

        Scenario key dispute points are useful guidance, but they should not
        keep consultations alive after all textual brackets have been resolved.
        """
        disputed_points = self.text_manager.get_disputed_points_summary()
        if not disputed_points:
            phase_manager = getattr(self, "phase_manager", None)
            if phase_manager is not None:
                disputed_points = phase_manager.get_phase_data(
                    "first_reading_disputed_points",
                    [],
                )
            else:
                disputed_points = []
        if not disputed_points:
            return []

        phase_cfg = (
            self.config.get("negotiation", {})
            .get("phases", {})
            .get("informal_consultations", {})
        )
        for issue in phase_cfg.get("key_dispute_points", []):
            label = str(issue).replace("_", " ")
            disputed_points.append(f"Scenario issue: {label}")

        return disputed_points

    def _preserve_unchanged_paragraphs_enabled(self) -> bool:
        """Return whether untouched original paragraphs should be locked verbatim."""
        config = getattr(self, "config", {}) or {}
        negotiation_text = config.get("negotiation", {}).get("text", {})
        if "preserve_unchanged_paragraphs" in negotiation_text:
            return bool(negotiation_text["preserve_unchanged_paragraphs"])
        return bool(
            config.get("text", {}).get(
                "preserve_unchanged_paragraphs",
                True,
            )
        )

    def _get_preserve_verbatim_paragraph_texts(self) -> List[str]:
        """Return untouched original paragraph texts for chair drafting prompts."""
        if not self._preserve_unchanged_paragraphs_enabled():
            return []

        text_manager = getattr(self, "text_manager", None)
        if text_manager is None or not hasattr(
            text_manager,
            "get_unchanged_paragraphs",
        ):
            return []

        return [
            paragraph.text
            for paragraph in text_manager.get_unchanged_paragraphs()
            if getattr(paragraph, "text", "").strip()
        ]

    def _phase_enabled(self, phase_name: str) -> bool:
        """Check whether a phase is enabled in the current runtime config."""
        default_enabled = False if phase_name == "coalition_caucus" else True
        return (
            self.config.get("negotiation", {})
            .get("phases", {})
            .get(phase_name, {})
            .get("enabled", default_enabled)
        )

    def _should_preserve_brackets(
        self,
        current_text: str,
        revised_text: str,
        round_proposals: List[Dict[str, str]],
        convergence: Dict[str, Any],
    ) -> bool:
        """
        Prevent the chair from debracketing a still-contested package.

        A fully clean text should not be introduced while the current round
        still contains hard objections or the Chair's own convergence
        assessment reports unresolved blockers.
        """
        if "[" not in current_text or "]" not in current_text:
            return False
        if "[" in revised_text or "]" in revised_text:
            return False

        likely_objectors = [
            proposal["agent"]
            for proposal in round_proposals
            if self._signals_hard_objection(proposal.get("content", ""))
        ]
        blocking_issues = convergence.get("blocking_issues", [])
        threshold = (
            self.config.get("negotiation", {})
            .get("phases", {})
            .get("informal_consultations", {})
            .get("min_convergence_for_clean_text", 0.95)
        )
        convergence_score = convergence.get("convergence_score", 0.0)

        return bool(likely_objectors or blocking_issues or convergence_score < threshold)

    @staticmethod
    def _signals_hard_objection(response: str) -> bool:
        """Heuristic for whether an intervention still signals a hard block."""
        response_lower = response.lower()
        markers = [
            "cannot accept",
            "we oppose",
            "we object",
            "object to",
            "red line",
            "non-negotiable",
            "cannot support",
            "must not",
        ]
        return any(marker in response_lower for marker in markers)

    def _chair_escalation(self, round_num: int, disputed_points: List[str]):
        """Chair escalation when progress stalls."""
        preserve_section = PromptTemplates.preserve_verbatim_section(
            self._get_preserve_verbatim_paragraph_texts()
        )
        prompt = f"""As Chair, progress has stalled for several rounds.
You need to propose a STRONG COMPROMISE to break the deadlock.

## CURRENT TEXT
{self.text_manager.get_full_text()}
{preserve_section}

## DISPUTED POINTS
{chr(10).join(f"- {dp}" for dp in disputed_points)}

## ESCALATION STRATEGIES
Consider using one or more of these approaches:
1. Remove the most contentious language entirely and replace with general reference
2. Use "constructive ambiguity" - language that each side can interpret favorably
3. Move specific commitments to a decision text or work programme
4. Use "chapeau" or preambular language to acknowledge principles
5. Create a footnote or interpretive note
6. Propose a phased approach: agree on framework now, details later
7. Use "shall endeavor" or "should, as appropriate" for softer obligations

Additional drafting discipline:
- Prefer narrower, more procedural language over expansive or innovative formulations
- Avoid sharp or polarizing terms unless they are clearly indispensable
- Do not introduce new beneficiaries, institutions, finance categories, or obligations unless they are already well supported
- If two ideas cannot both survive, delete the more contested detail rather than preserving both
- In endgame drafting, produce clean text without square brackets

Propose a complete revised text that has a realistic chance of acceptance.
"""
        messages = [
            {"role": "system", "content": self.chair.system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self.llm.generate(
            messages=messages,
            temperature=0.5,
            max_tokens=2500,
        )

        revised = self._extract_revised_text(response.content)
        if revised:
            current_text = self.text_manager.get_full_text()
            if "[" in current_text and "]" in current_text and "[" not in revised and "]" not in revised:
                logger.info(
                    "Chair escalation produced a fully clean text while "
                    "brackets remained in the underlying negotiation text. "
                    "Keeping the bracketed package."
                )
            else:
                self.text_manager.update_full_text(
                    revised, source="chair_escalation"
                )

        self.interaction_log.append(
            {
                "phase": "informal_consultations",
                "round": round_num,
                "agent": "CHAIR_ESCALATION",
                "content": response.content,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self.turn_manager.reset_passes()

    def _run_final_plenary(self):
        """
        Phase 4: Final plenary for adoption.

        FIX for Issue #2: Use the parsed action directly from
        AmendmentProcessor which now recognises OPPOSE / OBJECT.
        Any action that is not explicitly "accept" is treated as
        a potential objection.
        """
        logger.info("--- Phase 4: Final Plenary ---")
        phase_log = []

        final_text = self._finalize_text_for_plenary()

        # Chair presents final text
        chair_presentation = self.chair.present_final_text(final_text)
        self.interaction_log.append(
            {
                "phase": "final_plenary",
                "round": 0,
                "agent": "CHAIR",
                "content": chair_presentation,
                "timestamp": datetime.now().isoformat(),
            }
        )

        speaking_order = self.turn_manager.get_speaking_order(
            round_number=0, phase="final_plenary"
        )

        objections = []
        acceptances = []

        for agent_id in speaking_order:
            agent = self.agents[agent_id]

            response = agent.generate_final_plenary_response(
                final_text=final_text,
                scenario_context=self.scenario_context,
            )

            amendments = self.amendment_processor.parse_response(response)
            primary_action = self.amendment_processor.get_primary_action(response)

            # Only explicit "accept" counts as acceptance.
            # Explicit objections and last-minute text changes block
            # consensus. "pass" at final plenary is unusual but not an
            # objection.
            if primary_action == "accept":
                acceptances.append(agent_id)
            elif primary_action in ("oppose",):
                objections.append(
                    {"agent": agent_id, "reason": response[:300]}
                )
            elif primary_action in ("add", "delete", "modify"):
                objections.append(
                    {
                        "agent": agent_id,
                        "reason": (
                            f"Requested {primary_action} amendment: "
                            f"{response[:300]}"
                        ),
                    }
                )
            else:
                # "argue", "pass", etc. – not a clear accept
                # Log but don't count as either
                logger.warning(
                    f"Agent {agent_id} gave ambiguous final plenary "
                    f"response (action={primary_action}). "
                    f"Not counting as accept or objection."
                )

            entry = {
                "phase": "final_plenary",
                "round": 0,
                "agent": agent_id,
                "content": response,
                "action": primary_action,
                "timestamp": datetime.now().isoformat(),
            }
            self.interaction_log.append(entry)
            phase_log.append(entry)

        # Determine outcome
        require_consensus = (
            self.config.get("negotiation", {})
            .get("phases", {})
            .get("final_plenary", {})
            .get("require_consensus", True)
        )

        if require_consensus:
            adopted = len(objections) == 0 and len(acceptances) > 0
        else:
            adopted = len(acceptances) > len(objections)
        if adopted and require_consensus:
            consensus_status = "consensus"
        elif adopted:
            consensus_status = "adopted_without_consensus"
        else:
            consensus_status = "no_consensus"

        outcome_text = self.chair.declare_outcome(adopted, objections)
        self.interaction_log.append(
            {
                "phase": "final_plenary",
                "round": 0,
                "agent": "CHAIR",
                "content": outcome_text,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self.results["outcome"] = "ADOPTED" if adopted else "NOT_ADOPTED"
        self.results["consensus_status"] = consensus_status
        self.results["acceptances"] = acceptances
        self.results["objections"] = objections
        self.results["phases"]["final_plenary"] = {
            "adopted": adopted,
            "consensus_status": consensus_status,
            "acceptances": len(acceptances),
            "objections": len(objections),
            "log": phase_log,
        }

        self.phase_manager.increment_round()
        self._record_round_progress("final_plenary")

        logger.info(
            f"Final Plenary outcome: {'ADOPTED' if adopted else 'NOT ADOPTED'} "
            f"({len(acceptances)} accept, {len(objections)} object)"
        )

    def _get_latest_agent_positions(
        self,
        phase: str = "informal_consultations",
    ) -> Dict[str, str]:
        """Return the latest logged intervention per agent for a given phase."""
        positions: Dict[str, str] = {}
        known_agents = set(getattr(self, "agents", {}).keys())
        if not known_agents:
            return positions

        for entry in reversed(getattr(self, "interaction_log", [])):
            if entry.get("phase") != phase:
                continue
            agent_id = entry.get("agent")
            if agent_id not in known_agents or agent_id in positions:
                continue
            positions[agent_id] = entry.get("content", "")
            if len(positions) == len(known_agents):
                break

        return positions

    @staticmethod
    def _phase_counts_toward_round_budget(phase: str) -> bool:
        """
        Return whether a phase consumes the scarce endgame round budget.

        Opening statements and paragraph-by-paragraph first reading are useful
        agenda-setting steps, but they should not eat into the consultation
        budget that is needed for late-stage convergence.
        """
        return phase in {"informal_consultations", "final_plenary"}

    def _record_round_progress(self, phase: str):
        """Update bookkeeping for overall and budgeted round counts."""
        self.total_rounds += 1
        if self._phase_counts_toward_round_budget(phase):
            self.budgeted_rounds += 1

    @staticmethod
    def _paragraph_display_label(paragraph: Any) -> str:
        """
        Return the human-facing label for a text item.

        Real negotiations distinguish preambular paragraphs from operative
        paragraphs. Using the original operative numbering in first reading and
        endgame diagnostics makes later paragraph-specific bargaining much less
        error-prone than exposing only internal list indices.
        """
        display_label = str(getattr(paragraph, "display_label", "") or "").strip()
        if display_label:
            return display_label

        original_number = str(getattr(paragraph, "original_number", "") or "").strip()
        if original_number:
            return original_number.rstrip(".")

        paragraph_id = getattr(paragraph, "paragraph_id", "")
        if getattr(paragraph, "is_numbered", False):
            return str(paragraph_id)
        return f"Preamble {paragraph_id}"

    @staticmethod
    def _count_substantive_paragraph_blockers(
        paragraph_blockers: Dict[str, Dict[str, Any]],
    ) -> int:
        """
        Count blocker clusters that still attach to operative paragraphs.

        The goal is not to insist on zero ambiguity before plenary, but to
        avoid moving forward when multiple operative paragraphs still anchor
        incompatible coalitions.
        """
        substantive_refs = [
            ref
            for ref in paragraph_blockers
            if ref == "general" or re.match(r"^\d+(?:\([a-z]\))?$", str(ref))
        ]
        return len(substantive_refs)

    @staticmethod
    def _first_reading_has_live_disagreement(
        paragraph_actions: List[str],
        paragraph_amendments: List[Any],
    ) -> bool:
        """
        Decide whether a paragraph remains politically live after first reading.

        First reading should surface contested insertions, deletions,
        modifications, and objections. A paragraph is only "agreed" at this
        stage when no one is still pushing textual change or opposition.
        """
        if paragraph_amendments:
            return True
        live_actions = {"add", "delete", "modify", "oppose", "argue"}
        return any(action in live_actions for action in paragraph_actions)

    def _build_endgame_acceptability_map(
        self,
        agent_positions: Dict[str, str],
        candidate_text: str = "",
    ) -> Dict[str, Any]:
        """
        Build a lightweight acceptability map from the latest party signals.

        This heuristic does not try to predict the exact legal outcome. It
        simply gives the Chair a compact picture of which blocs appear ready
        to accept, likely to object, or still uncertain.
        """
        result = {
            "likely_accept": [],
            "conditional_accept": [],
            "likely_object": [],
            "uncertain": [],
            "signals": [],
            "blocker_tags_by_agent": {},
            "blocker_themes": {},
            "paragraph_blockers": {},
            "overloaded_paragraphs": {},
        }

        accept_markers = [
            "accept",
            "can accept",
            "could accept",
            "support the compromise",
            "able to support",
            "ready to support",
            "join consensus",
            "support as a basis",
            "accept this as a foundation",
            "can live with",
            "prepared to support",
        ]
        bridge_markers = [
            "we welcome",
            "we appreciate",
            "we remain open to",
            "we can work with",
            "we could work with",
            "we can go along with",
            "we could go along with",
        ]
        explicit_object_markers = [
            "cannot accept",
            "cannot accept in its current form",
            "we oppose",
            "oppose the text",
            "must oppose",
            "cannot support",
            "cannot join consensus",
            "we object",
            "would have to oppose",
            "red line",
            "unacceptable",
        ]
        hard_textual_demand_markers = [
            "must explicitly",
            "must include",
            "must retain",
            "must preserve",
            "must read",
            "we insist on",
            "we strongly maintain our position",
            "matter of principle",
            "matter of survival",
        ]

        for agent_id, statement in agent_positions.items():
            lowered = statement.lower()
            snippet = self._summarize_acceptability_signal(statement)
            structured_fields = self._extract_structured_bridge_fields(statement)
            blocker_tags = self._extract_blocker_tags(statement)
            paragraph_refs = self._extract_paragraph_references(statement)
            scenario_tags, matched_conditions = self._extract_scenario_condition_tags(
                agent_id=agent_id,
                statement=statement,
                condition_key="agent_blocking_conditions",
            )
            if scenario_tags:
                blocker_tags = list(dict.fromkeys(blocker_tags + scenario_tags))
            hard_condition_tags = self._extract_hard_condition_tags(
                statement,
                blocker_tags,
            )
            if hard_condition_tags:
                blocker_tags = list(
                    dict.fromkeys(
                        [tag for tag in blocker_tags if tag != "general_acceptability"]
                        + hard_condition_tags
                    )
                ) or blocker_tags

            has_accept_anchor = any(marker in lowered for marker in accept_markers)
            has_bridge_anchor = any(marker in lowered for marker in bridge_markers)
            has_explicit_object = any(
                marker in lowered for marker in explicit_object_markers
            )
            has_hard_textual_demand = any(
                marker in lowered for marker in hard_textual_demand_markers
            )

            conditional_status = None
            structured_assessment = None
            if candidate_text:
                structured_assessment = self._assess_structured_bridge_fields(
                    statement=statement,
                    candidate_text=candidate_text,
                    fallback_tags=blocker_tags,
                    fallback_refs=paragraph_refs,
                )
                if structured_assessment:
                    blocker_tags = list(
                        dict.fromkeys(
                            blocker_tags
                            + structured_assessment.get("floor_tags", [])
                            + structured_assessment.get("preferred_tags", [])
                        )
                    ) or blocker_tags
                    paragraph_refs = (
                        structured_assessment.get("paragraph_refs") or paragraph_refs
                    )
                conditional_status = self._classify_conditional_signal(
                    statement=statement,
                    candidate_text=candidate_text,
                    tags=blocker_tags,
                    paragraph_refs=paragraph_refs,
                    has_acceptance_anchor=has_accept_anchor,
                    has_bridge_anchor=has_bridge_anchor,
                )

            missing_tags: List[str] = []
            satisfied_tags: List[str] = []
            if candidate_text and scenario_tags and matched_conditions:
                missing_tags = self._missing_candidate_tags(
                    candidate_text=candidate_text,
                    tags=scenario_tags,
                )
                satisfied_tags = [
                    tag for tag in scenario_tags if tag not in missing_tags
                ]

            if has_explicit_object:
                result["likely_object"].append(agent_id)
                result["blocker_tags_by_agent"][agent_id] = blocker_tags
                for tag in blocker_tags:
                    result["blocker_themes"].setdefault(tag, []).append(agent_id)
                self._record_paragraph_blockers(
                    result["paragraph_blockers"],
                    paragraph_refs,
                    agent_id,
                    blocker_tags,
                )
                result["signals"].append(f"{agent_id}: likely object - {snippet}")
            elif structured_assessment:
                structured_status = structured_assessment.get("status")
                structured_signal = structured_assessment.get(
                    "reason",
                    "structured bridge signal",
                )
                if structured_status == "likely_object":
                    result["likely_object"].append(agent_id)
                    result["blocker_tags_by_agent"][agent_id] = blocker_tags
                    for tag in blocker_tags:
                        result["blocker_themes"].setdefault(tag, []).append(agent_id)
                    self._record_paragraph_blockers(
                        result["paragraph_blockers"],
                        paragraph_refs,
                        agent_id,
                        blocker_tags,
                    )
                    result["signals"].append(
                        f"{agent_id}: likely object ({structured_signal}) - {snippet}"
                    )
                elif structured_status == "conditional_accept":
                    result["conditional_accept"].append(agent_id)
                    result["blocker_tags_by_agent"][agent_id] = blocker_tags
                    for tag in blocker_tags:
                        result["blocker_themes"].setdefault(tag, []).append(agent_id)
                    self._record_paragraph_blockers(
                        result["paragraph_blockers"],
                        paragraph_refs,
                        agent_id,
                        blocker_tags,
                        bucket_key="conditional_acceptors",
                    )
                    result["signals"].append(
                        f"{agent_id}: conditional accept ({structured_signal}) - {snippet}"
                    )
                elif structured_status == "likely_accept":
                    result["likely_accept"].append(agent_id)
                    result["signals"].append(
                        f"{agent_id}: likely accept ({structured_signal}) - {snippet}"
                    )
                else:
                    result["uncertain"].append(agent_id)
                    result["signals"].append(
                        f"{agent_id}: uncertain ({structured_signal}) - {snippet}"
                    )
            elif (
                candidate_text
                and conditional_status == "likely_object"
            ):
                result["likely_object"].append(agent_id)
                result["blocker_tags_by_agent"][agent_id] = blocker_tags
                for tag in blocker_tags:
                    result["blocker_themes"].setdefault(tag, []).append(agent_id)
                self._record_paragraph_blockers(
                    result["paragraph_blockers"],
                    paragraph_refs,
                    agent_id,
                    blocker_tags,
                )
                result["signals"].append(
                    f"{agent_id}: likely object (conditional demand unmet) - {snippet}"
                )
            elif (
                candidate_text
                and scenario_tags
                and matched_conditions
                and missing_tags
            ):
                if (
                    has_accept_anchor
                    or has_bridge_anchor
                    or conditional_status == "conditional_accept"
                ):
                    result["conditional_accept"].append(agent_id)
                    result["blocker_tags_by_agent"][agent_id] = blocker_tags
                    for tag in blocker_tags:
                        result["blocker_themes"].setdefault(tag, []).append(agent_id)
                    self._record_paragraph_blockers(
                        result["paragraph_blockers"],
                        paragraph_refs,
                        agent_id,
                        blocker_tags,
                        bucket_key="conditional_acceptors",
                    )
                    result["signals"].append(
                        f"{agent_id}: conditional accept (scenario conditions not yet fully met: {', '.join(missing_tags)}) - {snippet}"
                    )
                elif missing_tags and not satisfied_tags:
                    result["likely_object"].append(agent_id)
                    result["blocker_tags_by_agent"][agent_id] = blocker_tags
                    for tag in blocker_tags:
                        result["blocker_themes"].setdefault(tag, []).append(agent_id)
                    self._record_paragraph_blockers(
                        result["paragraph_blockers"],
                        paragraph_refs,
                        agent_id,
                        blocker_tags,
                    )
                    result["signals"].append(
                        f"{agent_id}: likely object (scenario blocking conditions unmet: {', '.join(missing_tags)}) - {snippet}"
                    )
                else:
                    result["uncertain"].append(agent_id)
                    result["signals"].append(f"{agent_id}: uncertain - {snippet}")
            elif (
                candidate_text
                and conditional_status == "conditional_accept"
            ):
                result["conditional_accept"].append(agent_id)
                result["blocker_tags_by_agent"][agent_id] = blocker_tags
                for tag in blocker_tags:
                    result["blocker_themes"].setdefault(tag, []).append(agent_id)
                self._record_paragraph_blockers(
                    result["paragraph_blockers"],
                    paragraph_refs,
                    agent_id,
                    blocker_tags,
                    bucket_key="conditional_acceptors",
                )
                result["signals"].append(
                    f"{agent_id}: conditional accept - {snippet}"
                )
            elif (
                not candidate_text
                and has_hard_textual_demand
                and not has_accept_anchor
            ):
                result["likely_object"].append(agent_id)
                result["blocker_tags_by_agent"][agent_id] = blocker_tags
                for tag in blocker_tags:
                    result["blocker_themes"].setdefault(tag, []).append(agent_id)
                self._record_paragraph_blockers(
                    result["paragraph_blockers"],
                    paragraph_refs,
                    agent_id,
                    blocker_tags,
                )
                result["signals"].append(
                    f"{agent_id}: likely object (hard textual demand) - {snippet}"
                )
            elif any(marker in lowered for marker in accept_markers):
                result["likely_accept"].append(agent_id)
                result["signals"].append(f"{agent_id}: likely accept - {snippet}")
            else:
                result["uncertain"].append(agent_id)
                result["signals"].append(f"{agent_id}: uncertain - {snippet}")

        result["overloaded_paragraphs"] = self._detect_overloaded_paragraphs(
            result["paragraph_blockers"]
        )
        return result

    @staticmethod
    def _extract_structured_bridge_fields(statement: str) -> Dict[str, Any]:
        """
        Parse late-stage bridge fields when agents use the structured format.

        This lets the engine distinguish minimum adoption conditions from
        preferred improvements instead of treating the whole intervention as a
        single hard objection.
        """
        labels = {
            "ADOPTION FLOOR": "adoption_floor",
            "PREFERRED IMPROVEMENT": "preferred_improvement",
            "CAN ACCEPT WITHOUT PREFERRED IMPROVEMENT": "can_accept_without_preferred",
            "BRIDGE TEXT": "bridge_text",
            "RESOLUTION MODE VIEW": "resolution_mode_view",
        }
        pattern = re.compile(
            r"(?ims)^\s*(ADOPTION FLOOR|PREFERRED IMPROVEMENT|CAN ACCEPT WITHOUT PREFERRED IMPROVEMENT|BRIDGE TEXT|RESOLUTION MODE VIEW):\s*(.*?)\s*(?=^\s*(?:ADOPTION FLOOR|PREFERRED IMPROVEMENT|CAN ACCEPT WITHOUT PREFERRED IMPROVEMENT|BRIDGE TEXT|RESOLUTION MODE VIEW):|\Z)"
        )
        fields: Dict[str, Any] = {
            "present": False,
            "adoption_floor": "",
            "preferred_improvement": "",
            "can_accept_without_preferred": None,
            "bridge_text": "",
            "resolution_mode_view": "",
        }

        for label, content in pattern.findall(statement):
            key = labels[label.upper()]
            cleaned = " ".join(content.strip().split())
            if cleaned:
                fields["present"] = True
                fields[key] = cleaned

        value = fields.get("can_accept_without_preferred")
        if isinstance(value, str) and value:
            lowered = value.lower()
            if lowered.startswith("yes") or lowered == "y":
                fields["can_accept_without_preferred"] = True
            elif lowered.startswith("no") or lowered == "n":
                fields["can_accept_without_preferred"] = False
            else:
                fields["can_accept_without_preferred"] = None

        return fields

    def _assess_structured_bridge_fields(
        self,
        statement: str,
        candidate_text: str,
        fallback_tags: List[str],
        fallback_refs: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Assess whether structured late-stage fields are satisfied."""
        fields = self._extract_structured_bridge_fields(statement)
        if not fields.get("present"):
            return None

        floor_text = fields.get("adoption_floor", "") or statement
        preferred_text = fields.get("preferred_improvement", "")
        floor_tags = self._extract_hard_condition_tags(floor_text, fallback_tags)
        preferred_tags = self._extract_hard_condition_tags(preferred_text, floor_tags)
        floor_refs = self._extract_paragraph_references(floor_text) or fallback_refs or [
            "general"
        ]
        preferred_refs = (
            self._extract_paragraph_references(preferred_text) or floor_refs
        )

        floor_missing = self._structured_segment_missing_tags(
            candidate_text=candidate_text,
            paragraph_refs=floor_refs,
            tags=floor_tags,
            requirement_text=floor_text,
        )
        preferred_missing = self._structured_segment_missing_tags(
            candidate_text=candidate_text,
            paragraph_refs=preferred_refs,
            tags=preferred_tags,
            requirement_text=preferred_text,
        ) if preferred_text else []
        can_accept_without_preferred = fields.get("can_accept_without_preferred")

        if floor_missing:
            reason = f"adoption floor unmet: {', '.join(floor_missing)}"
            status = "likely_object"
        elif preferred_text and can_accept_without_preferred is False and preferred_missing:
            reason = (
                "preferred improvement marked as indispensable and still unmet: "
                + ", ".join(preferred_missing)
            )
            status = "likely_object"
        elif preferred_text and preferred_missing:
            reason = (
                "adoption floor met but preferred improvement still open: "
                + ", ".join(preferred_missing)
            )
            status = "conditional_accept"
        elif can_accept_without_preferred is False:
            reason = "structured floor and indispensable improvement appear met"
            status = "likely_accept"
        else:
            reason = "structured adoption floor appears met"
            status = "likely_accept"

        return {
            "status": status,
            "reason": reason,
            "paragraph_refs": list(dict.fromkeys(floor_refs + preferred_refs)),
            "floor_tags": floor_tags,
            "preferred_tags": preferred_tags if preferred_text else [],
            "can_accept_without_preferred": can_accept_without_preferred,
        }

    def _structured_segment_missing_tags(
        self,
        candidate_text: str,
        paragraph_refs: List[str],
        tags: List[str],
        requirement_text: str = "",
    ) -> List[str]:
        """
        Return missing structured requirements across one or more references.

        For structured late-stage fields, theme tags alone are too coarse. We
        therefore also check whether the candidate text retains the narrower
        textual anchors from the requirement itself, such as references to the
        most vulnerable, SIDS, CBDR, or specific finance qualifiers.
        """
        if not tags and not requirement_text:
            return []

        refs = paragraph_refs or ["general"]
        combined_missing: List[str] = []
        for ref in refs:
            segment = self._extract_candidate_segment(candidate_text, ref)
            missing = self._missing_candidate_tags(segment, tags)
            if requirement_text and self._structured_requirement_keyword_ratio(
                requirement_text=requirement_text,
                candidate_segment=segment,
            ) < 0.45:
                if "textual_specificity" not in missing:
                    missing.append("textual_specificity")
            if not missing:
                return []
            for tag in missing:
                if tag not in combined_missing:
                    combined_missing.append(tag)
        return combined_missing

    @classmethod
    def _structured_requirement_keyword_ratio(
        cls,
        requirement_text: str,
        candidate_segment: str,
    ) -> float:
        """Measure how much of a structured requirement is still visible."""
        if not requirement_text.strip():
            return 1.0

        stop_words = {
            "paragraph",
            "subparagraph",
            "adoption",
            "floor",
            "preferred",
            "improvement",
            "bridge",
            "text",
            "retain",
            "retained",
            "remain",
            "remains",
            "must",
            "should",
            "could",
            "would",
            "option",
            "first",
            "second",
            "current",
            "explicit",
            "reference",
            "references",
            "clause",
            "clauses",
            "language",
            "wording",
            "support",
            "provide",
            "provided",
            "providedby",
            "prioritize",
            "including",
            "linked",
            "paragraphs",
        }
        keywords = [
            keyword
            for keyword in cls._meaningful_condition_keywords(requirement_text)
            if keyword not in stop_words
        ]
        if not keywords:
            return 1.0

        candidate_lower = candidate_segment.lower()
        hits = sum(1 for keyword in keywords if keyword in candidate_lower)
        return hits / len(keywords)

    @staticmethod
    def _blocker_tag_markers() -> Dict[str, List[str]]:
        """Return generic keyword families for blocker themes."""
        return {
            "principles": [
                "cbdr",
                "common but differentiated",
                "equity",
                "principles of the convention",
                "differentiated responsibilities",
                "national circumstances",
            ],
            "status": [
                "equal footing",
                "equal status",
                "status of",
                "distinct and robust",
                "distinct standing",
                "stand-alone",
                "standalone",
                "not subordinate",
                "subordinate",
                "subordination",
                "subsidiary to",
                "hierarchy",
                "marginalized",
                "integral part",
            ],
            "support": [
                "support for developing countr",
                "means of implementation",
                "finance",
                "capacity-building",
                "capacity building",
                "technology transfer",
                "developing country",
                "vulnerable countr",
                "adaptation",
            ],
            "reporting": [
                "reporting",
                "review",
                "tracking",
                "transparency",
                "modalities",
                "flexibility",
            ],
            "integrity": [
                "environmental integrity",
                "double counting",
                "accounting",
                "not duplicate",
                "non-duplication",
                "complementarity",
                "coherence",
                "transparency",
            ],
            "governance": [
                "governance",
                "institutional arrangement",
                "secretariat",
                "sbsta",
                "body",
                "committee",
                "coordination",
            ],
            "timing": [
                "timeline",
                "timing",
                "session",
                "deadline",
                "work programme",
                "work program",
                "phased approach",
            ],
        }

    @staticmethod
    def _extract_blocker_tags(statement: str) -> List[str]:
        """
        Map objection language to broad negotiation themes.

        These tags stay at the workflow level. They are meant to tell the Chair
        what kind of bridge is needed, not to encode scenario-specific answers.
        """
        lowered = statement.lower()
        marker_map = NegotiationEngine._blocker_tag_markers()

        tags: List[str] = []
        for tag, markers in marker_map.items():
            if any(marker in lowered for marker in markers):
                tags.append(tag)

        if not tags:
            tags.append("general_acceptability")

        return tags

    @staticmethod
    def _extract_paragraph_references(statement: str) -> List[str]:
        """Extract paragraph references like 'paragraph 2(d)' or 'paragraph 4'."""
        refs = [
            match.group(1).strip().lower()
            for match in re.finditer(
                r"\bparagraph\s+(\d+(?:\([a-z]\))?)(?!\w)",
                statement,
                re.IGNORECASE,
            )
        ]
        seen = set()
        ordered: List[str] = []
        for ref in refs:
            if ref not in seen:
                ordered.append(ref)
                seen.add(ref)
        return ordered

    def _extract_scenario_condition_tags(
        self,
        agent_id: str,
        statement: str,
        condition_key: str,
    ) -> Tuple[List[str], List[str]]:
        """
        Return generic theme tags from scenario conditions that are active here.

        This keeps bloc base profiles general. Scenario files already encode the
        officially relevant blocking and acceptance conditions, so endgame
        diagnostics should consult those conditions rather than guessing solely
        from ad hoc keywords in one intervention.
        """
        config = getattr(self, "config", {}) or {}
        scenario = getattr(self, "scenario", None) or config.get("scenario", {})
        constraints = scenario.get("scenario_constraints", {})
        conditions = constraints.get(condition_key, {}).get(agent_id, [])

        tags: List[str] = []
        matched_conditions: List[str] = []
        for condition in conditions:
            if not self._condition_is_salient_in_statement(statement, condition):
                continue
            matched_conditions.append(condition)
            for tag in self._extract_blocker_tags(condition):
                if tag != "general_acceptability" and tag not in tags:
                    tags.append(tag)

        return tags, matched_conditions

    @staticmethod
    def _meaningful_condition_keywords(text: str) -> List[str]:
        """Extract non-trivial keywords from scenario conditions."""
        stop_words = {
            "must",
            "will",
            "shall",
            "should",
            "the",
            "and",
            "for",
            "not",
            "with",
            "that",
            "this",
            "from",
            "have",
            "been",
            "are",
            "was",
            "were",
            "any",
            "new",
            "all",
            "our",
            "their",
            "its",
            "also",
            "final",
            "text",
            "omits",
            "omitted",
            "allows",
            "allow",
            "where",
            "issue",
            "issues",
            "party",
            "parties",
            "approach",
            "approaches",
            "framework",
            "referred",
            "relevant",
        }
        return [
            word
            for word in re.findall(r"\w+", text.lower())
            if len(word) > 3 and word not in stop_words
        ]

    @classmethod
    def _condition_is_salient_in_statement(cls, statement: str, condition: str) -> bool:
        """Check whether a scenario condition is clearly active in a statement."""
        statement_lower = statement.lower()
        condition_lower = condition.lower()
        if condition_lower in statement_lower:
            return True

        keywords = cls._meaningful_condition_keywords(condition_lower)
        if not keywords:
            return False

        hits = sum(1 for keyword in keywords if keyword in statement_lower)
        min_hits = 2 if len(keywords) >= 4 else 1
        required_ratio = 0.30
        return hits >= min_hits and (hits / len(keywords)) >= required_ratio

    @staticmethod
    def _record_paragraph_blockers(
        paragraph_blockers: Dict[str, Dict[str, Any]],
        paragraph_refs: List[str],
        agent_id: str,
        blocker_tags: List[str],
        bucket_key: str = "objectors",
    ):
        """Accumulate paragraph-specific objector information."""
        refs = paragraph_refs or ["general"]
        for ref in refs:
            bucket = paragraph_blockers.setdefault(
                ref,
                {
                    "objectors": [],
                    "conditional_acceptors": [],
                    "themes": [],
                },
            )
            pressure_bucket = bucket.setdefault(bucket_key, [])
            if agent_id not in pressure_bucket:
                pressure_bucket.append(agent_id)
            for tag in blocker_tags:
                if tag not in bucket["themes"]:
                    bucket["themes"].append(tag)

    @staticmethod
    def _detect_overloaded_paragraphs(
        paragraph_blockers: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Identify paragraphs carrying too many different political functions.

        Late-stage texts often fail when one operative paragraph becomes a
        catch-all for support, integrity, status, timing, and governance
        demands. These paragraphs should be simplified or split, not merely
        loaded with more compromise phrases.
        """
        overloaded: Dict[str, Dict[str, Any]] = {}
        for paragraph_ref, details in paragraph_blockers.items():
            if paragraph_ref == "general":
                continue

            themes = [
                theme
                for theme in details.get("themes", [])
                if theme != "general_acceptability"
            ]
            hard_objectors = details.get("objectors", [])
            conditional_acceptors = details.get("conditional_acceptors", [])
            total_pressure = len(hard_objectors) + len(conditional_acceptors)

            is_overloaded = len(themes) >= 3 or (
                len(themes) >= 2 and total_pressure >= 4
            )
            if not is_overloaded:
                continue

            overloaded[paragraph_ref] = {
                "themes": themes,
                "hard_objectors": list(hard_objectors),
                "conditional_acceptors": list(conditional_acceptors),
                "reason": (
                    f"{len(themes)} themes across {total_pressure} pressure signals"
                ),
                "recommended_resolution": NegotiationEngine._recommend_resolution_mode(
                    themes
                ),
            }

        return overloaded

    @staticmethod
    def _recommend_resolution_mode(themes: List[str]) -> str:
        """
        Suggest whether a late-stage bridge should merge, split, or relocate.

        Split: one paragraph is mixing different political functions.
        Relocate: one overflow issue is better moved into procedural text.
        Merge: tensions are narrow enough for a single-phrase bridge.
        """
        theme_set = set(themes)
        political = {"support", "principles", "status"}
        technical = {"integrity", "reporting", "governance", "timing"}

        if len(theme_set) >= 4 or (theme_set & political and theme_set & technical):
            return "split"
        if len(theme_set) >= 3 or (
            theme_set & {"reporting", "timing", "governance"}
            and theme_set & {"support", "principles"}
        ):
            return "relocate"
        return "merge"

    def _classify_conditional_signal(
        self,
        statement: str,
        candidate_text: str,
        tags: List[str],
        paragraph_refs: List[str],
        has_acceptance_anchor: bool = False,
        has_bridge_anchor: bool = False,
    ) -> Optional[str]:
        """
        Distinguish between hard adoption blockers and bargaining maxima.

        Many late-stage interventions sound demanding while still signaling
        willingness to land the package if a narrow bridge is found. The goal
        here is to classify those statements as conditional accept rather than
        immediate objection unless the language clearly signals a veto.
        """
        lowered = statement.lower()
        hard_retention_markers = [
            "oppose any deletion",
            "must be retained",
            "are red lines",
            "non-negotiable",
            "fundamental principles",
        ]
        hard_condition_markers = [
            "ultimate test is whether",
            "must deliver",
            "must be explicitly",
            "must remain",
            "must retain",
            "must preserve",
            "must include",
            "depends on whether",
            "provided that",
            "only if",
            "prepared to accept the text if",
            "we are prepared to accept the text if",
            "matter of survival",
            "matter of principle",
        ]
        if not (
            any(marker in lowered for marker in hard_retention_markers)
            or any(marker in lowered for marker in hard_condition_markers)
            or (
                (has_acceptance_anchor or has_bridge_anchor)
                and any(token in lowered for token in (" however", " but ", "provided that"))
            )
        ):
            return None

        condition_segment = self._extract_condition_segment(statement)
        condition_tags = self._extract_hard_condition_tags(statement, tags)
        refs = self._extract_paragraph_references(condition_segment) or paragraph_refs or [
            "general"
        ]
        missing_any = any(
            bool(
                self._missing_candidate_tags(
                    candidate_text=self._extract_candidate_segment(
                        candidate_text,
                        ref,
                    ),
                    tags=condition_tags,
                )
            )
            for ref in refs
        )
        if not missing_any:
            return None

        if any(marker in lowered for marker in hard_retention_markers):
            return "likely_object"
        if has_acceptance_anchor or has_bridge_anchor:
            return "conditional_accept"
        return "conditional_accept"

    def _signals_conditional_block(
        self,
        statement: str,
        candidate_text: str,
        tags: List[str],
        paragraph_refs: List[str],
    ) -> bool:
        """
        Detect cases where a party has not said 'cannot accept' yet, but is
        clearly signaling that deletion of specific language would become a block.
        """
        return (
            self._classify_conditional_signal(
                statement=statement,
                candidate_text=candidate_text,
                tags=tags,
                paragraph_refs=paragraph_refs,
                has_acceptance_anchor=any(
                    marker in statement.lower()
                    for marker in [
                        "accept",
                        "can accept",
                        "could accept",
                        "prepared to support",
                    ]
                ),
                has_bridge_anchor=any(
                    marker in statement.lower()
                    for marker in [
                        "we welcome",
                        "we appreciate",
                        "we remain open to",
                        "we can work with",
                    ]
                ),
            )
            == "likely_object"
        )

    @staticmethod
    def _extract_condition_segment(statement: str) -> str:
        """Return the most demanding clause of a conditional intervention."""
        lowered = statement.lower()
        split_markers = [
            "ultimate test is whether",
            "depends on whether",
            "provided that",
            "only if",
            "however",
            " but ",
        ]
        for marker in split_markers:
            index = lowered.find(marker)
            if index == -1:
                continue
            start = index if marker.strip() != "but" else index + len(marker)
            segment = statement[start:].strip(" ,:;-")
            if segment:
                return segment
        return statement

    def _extract_hard_condition_tags(
        self,
        statement: str,
        fallback_tags: List[str],
    ) -> List[str]:
        """Infer the theme tags attached to the hardest condition in a statement."""
        condition_segment = self._extract_condition_segment(statement)
        condition_tags = [
            tag
            for tag in self._extract_blocker_tags(condition_segment)
            if tag != "general_acceptability"
        ]
        if condition_tags:
            return condition_tags
        return [
            tag
            for tag in fallback_tags
            if tag != "general_acceptability"
        ]

    def _candidate_text_satisfies_tags(
        self,
        candidate_text: str,
        tags: List[str],
    ) -> bool:
        """Check whether the candidate text still contains signal language for the requested themes."""
        return any(
            self._candidate_text_supports_tag(candidate_text, tag)
            for tag in tags
        )

    @staticmethod
    def _candidate_tag_evidence() -> Dict[str, List[str]]:
        """Return generic textual evidence for each blocker theme."""
        return {
            "principles": [
                "equity",
                "common but differentiated",
                "respective capabilities",
                "national circumstances",
                "principles of the convention",
            ],
            "support": [
                "developing country",
                "vulnerable",
                "finance",
                "technology transfer",
                "capacity-building",
                "capacity building",
                "means of implementation",
                "adaptation",
            ],
            "reporting": [
                "reporting",
                "review",
                "tracking",
                "transparency",
                "flexibility",
            ],
            "integrity": [
                "environmental integrity",
                "double counting",
                "robust transparency",
                "transparency",
                "accounting",
            ],
            "status": [
                "equal footing",
                "equal status",
                "integral part",
                "distinct",
                "standalone",
                "stand-alone",
                "not subordinate",
                "without subordinating",
            ],
            "governance": [
                "institutional",
                "sbsta",
                "secretariat",
                "committee",
                "governance",
            ],
            "timing": [
                "session",
                "work programme",
                "deadline",
                "timeline",
            ],
        }

    def _candidate_text_supports_tag(self, candidate_text: str, tag: str) -> bool:
        """Return whether the current candidate text still supports a theme."""
        evidence = self._candidate_tag_evidence().get(tag)
        if not evidence:
            return False
        lowered = candidate_text.lower()
        return any(token in lowered for token in evidence)

    def _missing_candidate_tags(
        self,
        candidate_text: str,
        tags: List[str],
    ) -> List[str]:
        """Return scenario-relevant tags that currently lack textual support."""
        ordered_tags = list(dict.fromkeys(tags))
        return [
            tag
            for tag in ordered_tags
            if self._candidate_tag_evidence().get(tag)
            and not self._candidate_text_supports_tag(candidate_text, tag)
        ]

    @staticmethod
    def _extract_candidate_segment(candidate_text: str, paragraph_ref: str) -> str:
        """Extract a paragraph or subparagraph segment from candidate clean text."""
        if not candidate_text.strip() or paragraph_ref == "general":
            return candidate_text

        match = re.fullmatch(r"(\d+)(?:\(([a-z])\))?", paragraph_ref)
        if not match:
            return candidate_text

        para_num = match.group(1)
        subpara = match.group(2)
        pattern = re.compile(
            rf"(?ms)^\s*{re.escape(para_num)}\.\s+(.*?)(?=^\s*\d+\.\s+|\Z)"
        )
        para_match = pattern.search(candidate_text)
        if not para_match:
            return candidate_text
        para_block = para_match.group(1).strip()
        if not subpara:
            return para_block

        sub_pattern = re.compile(
            rf"(?ms)^\s*\({re.escape(subpara)}\)\s+(.*?)(?=^\s*\([a-z]\)\s+|\Z)"
        )
        sub_match = sub_pattern.search(para_block)
        return sub_match.group(1).strip() if sub_match else para_block

    @staticmethod
    def _summarize_acceptability_signal(statement: str) -> str:
        """Return a compact one-line summary of an intervention."""
        condensed = " ".join(statement.strip().split())
        if not condensed:
            return "no signal captured"
        return condensed[:180]

    def _detect_final_text_issues(self, text: str) -> List[str]:
        """
        Detect obvious drafting defects in the final clean text.

        The goal is not to fully parse legal English. We only catch high-value
        defects that reliably break adoption quality, such as dangling
        conjunctions and fragmentary list items.
        """
        issues: List[str] = []
        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            if re.search(r"\b(?:and|or),\s*$", line, re.IGNORECASE):
                issues.append(
                    f"Line {line_no} ends with a dangling conjunction."
                )

            if re.match(r"^\([A-Za-z0-9]+\)\s+[a-z]", line):
                issues.append(
                    f"Line {line_no} starts a list item with a lowercase word and may be a fragment."
                )

            if (
                re.match(
                    r"^(Acknowledging|Recognizing|Noting|Emphasizing|Recalling)\b",
                    line,
                    re.IGNORECASE,
                )
                and re.search(r"\band other (approaches|mechanisms)\b", line, re.IGNORECASE)
            ):
                relation_markers = (
                    "can complement",
                    "complement",
                    "alongside",
                    "integral part",
                    "equal importance",
                    "coherence with",
                    "relationship with",
                    "relationship to",
                    "together with",
                )
                if not any(marker in line.lower() for marker in relation_markers):
                    issues.append(
                        f"Line {line_no} mentions other approaches or mechanisms but may be missing a relation phrase."
                    )

        return issues

    @staticmethod
    def _acceptability_scorecard(
        acceptability_map: Dict[str, Any],
        drafting_issues: List[str],
    ) -> Dict[str, int]:
        """Summarize the pre-plenary risk profile in a compact scorecard."""
        return {
            "hard_objectors": len(acceptability_map.get("likely_object", [])),
            "conditional_accepts": len(
                acceptability_map.get("conditional_accept", [])
            ),
            "overloaded_paragraphs": len(
                acceptability_map.get("overloaded_paragraphs", {})
            ),
            "drafting_issues": len(drafting_issues),
        }

    @classmethod
    def _repair_outperforms_baseline(
        cls,
        baseline_map: Dict[str, Any],
        baseline_issues: List[str],
        repaired_map: Dict[str, Any],
        repaired_issues: List[str],
    ) -> bool:
        """
        Compare a repaired package against the baseline.

        Hard objectors matter most, then conditional acceptances that still
        need work, then overloaded paragraphs, then pure drafting defects.
        """
        baseline_score = cls._acceptability_scorecard(
            baseline_map,
            baseline_issues,
        )
        repaired_score = cls._acceptability_scorecard(
            repaired_map,
            repaired_issues,
        )
        baseline_tuple = (
            baseline_score["hard_objectors"],
            baseline_score["conditional_accepts"],
            baseline_score["overloaded_paragraphs"],
            baseline_score["drafting_issues"],
        )
        repaired_tuple = (
            repaired_score["hard_objectors"],
            repaired_score["conditional_accepts"],
            repaired_score["overloaded_paragraphs"],
            repaired_score["drafting_issues"],
        )
        return repaired_tuple < baseline_tuple

    def _should_attempt_pre_plenary_repair(
        self,
        drafting_issues: List[str],
        acceptability_map: Dict[str, Any],
    ) -> bool:
        """Decide whether the Chair should do one final clean-text repair pass."""
        if drafting_issues:
            return True
        return bool(
            acceptability_map.get("likely_object")
            or acceptability_map.get("conditional_accept")
            or acceptability_map.get("overloaded_paragraphs")
        )

    def _finalize_text_for_plenary(self) -> str:
        """
        Convert the negotiated package into a clean adoption text.

        Endgame practice should remove brackets rather than carry them into
        the formal adoption moment. If consensus is still missing, the text
        should become narrower and more procedural, not more expansive.
        """
        if not hasattr(self, "results") or self.results is None:
            self.results = {}

        current_text = self.text_manager.get_full_text()
        clean_text = self.text_manager.get_adoption_ready_text()

        if not clean_text:
            clean_text = current_text.replace("[", "").replace("]", "")

        latest_positions = self._get_latest_agent_positions()
        if len(latest_positions) < len(getattr(self, "agents", {})):
            for agent_id, statement in self._get_latest_agent_positions(
                phase="first_reading"
            ).items():
                latest_positions.setdefault(agent_id, statement)
        acceptability_map = self._build_endgame_acceptability_map(
            latest_positions,
            candidate_text=clean_text,
        )
        drafting_issues = self._detect_final_text_issues(clean_text)
        baseline_scorecard = self._acceptability_scorecard(
            acceptability_map,
            drafting_issues,
        )

        self.results["pre_plenary_checks"] = {
            "drafting_issues": drafting_issues,
            "acceptability_map": acceptability_map,
            "scorecard": baseline_scorecard,
            "accepted_scorecard": baseline_scorecard,
            "repair_applied": False,
        }

        if (
            getattr(self, "chair", None) is not None
            and self._should_attempt_pre_plenary_repair(
                drafting_issues=drafting_issues,
                acceptability_map=acceptability_map,
            )
        ):
            scenario_config = getattr(self, "scenario", {}) or {}
            logger.info(
                "Running final Chair repair pass before plenary "
                f"(issues={len(drafting_issues)}, likely_objectors={len(acceptability_map.get('likely_object', []))})."
            )
            repair_response = self.chair.revise_for_adoption(
                candidate_text=clean_text,
                acceptability_map=acceptability_map,
                drafting_issues=drafting_issues,
                agent_positions=latest_positions,
                scenario_context=scenario_config.get("context", ""),
                preserve_terms=(
                    scenario_config.get("scenario_constraints", {}).get(
                        "must_preserve_terms", []
                    )
                ),
                structure_guidance=self._build_structure_guidance(current_text),
                preserve_verbatim_paragraphs=(
                    self._get_preserve_verbatim_paragraph_texts()
                ),
            )
            repaired_text = self._extract_revised_text(repair_response) or repair_response.strip()
            repaired_text = self._stabilize_revised_text_structure(
                current_text=current_text,
                revised_text=repaired_text,
                clean_only=True,
            )
            repaired_text = repaired_text.replace("[", "").replace("]", "").strip()
            repaired_issues = self._detect_final_text_issues(repaired_text)
            repaired_acceptability_map = self._build_endgame_acceptability_map(
                latest_positions,
                candidate_text=repaired_text,
            )
            repaired_scorecard = self._acceptability_scorecard(
                repaired_acceptability_map,
                repaired_issues,
            )
            self.results["pre_plenary_checks"][
                "post_repair_issues"
            ] = repaired_issues
            self.results["pre_plenary_checks"][
                "post_repair_acceptability_map"
            ] = repaired_acceptability_map
            self.results["pre_plenary_checks"][
                "post_repair_scorecard"
            ] = repaired_scorecard

            if repaired_text and self._repair_outperforms_baseline(
                baseline_map=acceptability_map,
                baseline_issues=drafting_issues,
                repaired_map=repaired_acceptability_map,
                repaired_issues=repaired_issues,
            ):
                clean_text = repaired_text
                self.results["pre_plenary_checks"]["repair_applied"] = True
                self.results["pre_plenary_checks"]["accepted_scorecard"] = (
                    repaired_scorecard
                )
            else:
                self.results["pre_plenary_checks"]["repair_applied"] = False
                self.results["pre_plenary_checks"]["accepted_scorecard"] = (
                    baseline_scorecard
                )

        if clean_text != current_text:
            self.text_manager.update_full_text(
                clean_text,
                source="pre_plenary_cleanup",
            )

        return self.text_manager.get_full_text()

    @staticmethod
    def _extract_leading_preamble_paragraphs(text: str) -> List[str]:
        """Return leading unnumbered paragraphs before the first operative paragraph."""
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text.strip())
            if paragraph.strip()
        ]
        preamble: List[str] = []
        for paragraph in paragraphs:
            if re.match(r"^\d+\.\s", paragraph):
                break
            preamble.append(paragraph)
        return preamble

    def _extract_drafting_anchors(self, text: str) -> List[str]:
        """Collect explicit legal and procedural anchors already present in the text."""
        anchors: List[str] = []

        for match in re.findall(r"Article\s+\d+(?:\.\d+)?", text):
            if match not in anchors:
                anchors.append(match)

        phrase_patterns = [
            r"Subsidiary Body for Scientific and Technological Advice",
            r"Conference of the Parties serving as the meeting of the Parties to the Paris Agreement",
            r"submission portal",
            r"report(?:ing)? requirements",
            r"work programme",
            r"institutional arrangements",
            r"report on progress",
        ]
        for pattern in phrase_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                phrase = match.group(0)
                if phrase.lower() not in {anchor.lower() for anchor in anchors}:
                    anchors.append(phrase)

        return anchors[:8]

    def _build_structure_guidance(self, text: str) -> str:
        """Build compact structural guidance for Chair drafting steps."""
        if not text.strip():
            return ""

        sections: List[str] = []
        preamble = self._extract_leading_preamble_paragraphs(text)
        if preamble:
            sections.append(
                "Leading preambular paragraphs currently in the text:\n"
                + "\n".join(
                    f"- {' '.join(paragraph.split())[:160]}"
                    for paragraph in preamble[:4]
                )
            )

        numbered_paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text.strip())
            if re.match(r"^\d+\.\s", paragraph.strip())
        ]
        if numbered_paragraphs:
            sections.append(
                "Current operative paragraph skeleton:\n"
                + "\n".join(
                    f"- {' '.join(paragraph.split())[:160]}"
                    for paragraph in numbered_paragraphs[:6]
                )
            )

        anchors = self._extract_drafting_anchors(text)
        if anchors:
            sections.append(
                "Explicit legal or procedural anchors already present:\n"
                + "\n".join(f"- {anchor}" for anchor in anchors)
            )

        return "\n\n".join(sections)

    def _stabilize_revised_text_structure(
        self,
        current_text: str,
        revised_text: str,
        clean_only: bool = False,
    ) -> str:
        """
        Preserve uncontested leading structure when the Chair rewrites the package.

        This is intentionally conservative: it only reinstates leading
        preambular paragraphs when the revised text drops them entirely.
        """
        if not current_text.strip() or not revised_text.strip():
            return revised_text

        current_preamble = self._extract_leading_preamble_paragraphs(current_text)
        revised_preamble = self._extract_leading_preamble_paragraphs(revised_text)

        if not current_preamble or revised_preamble:
            return revised_text

        if clean_only:
            current_preamble = [
                paragraph
                for paragraph in current_preamble
                if "[" not in paragraph and "]" not in paragraph
            ]

        if not current_preamble:
            return revised_text

        return "\n\n".join(current_preamble + [revised_text.strip()])

    def _extract_revised_text(self, chair_response: str) -> Optional[str]:
        """
        Extract the revised text section from the chair's response.

        Strategy:
        1. Prefer an explicit REVISED TEXT / FINAL TEXT section and stop at
           the next known chair section.
        2. If the model omitted the section heading, start only at a line that
           looks like actual decision text.  This avoids saving procedural
           notes as final negotiating text.
        """
        SECTION_HEADERS = {
            "REVISED TEXT",
            "FINAL TEXT",
            "TEXT FOR ADOPTION",
            "CLEAN TEXT",
            "PROGRESS SUMMARY",
            "PROGRESS",
            "REMAINING ISSUES",
            "REMAINING",
            "CHAIR'S NOTE",
            "CHAIR",
            "SUMMARY",
            "NOTE",
            "PROCEDURAL NOTE",
            "TEXT PRESENTED",
            "CHAIR'S PROPOSAL",
            "ACCEPTANCE MAP",
            "CONSENSUS CHECK",
            "NEXT STEPS",
        }
        TEXT_HEADERS = {
            "REVISED TEXT",
            "FINAL TEXT",
            "TEXT FOR ADOPTION",
            "CLEAN TEXT",
        }

        NOISE_TOKENS = {
            "done", "done.", "none", "none.", "end", "end.",
            "n/a", "n/a.", "---", "***", "...", "```",
        }

        def header_label(line: str) -> str:
            stripped = line.strip().strip("*").strip()
            stripped = re.sub(r"^#+\s*", "", stripped)
            if ":" in stripped:
                stripped = stripped.split(":", 1)[0]
            return stripped.strip().upper()

        def same_line_after_header(line: str) -> str:
            if ":" not in line:
                return ""
            return line.split(":", 1)[1].strip().strip("*").strip()

        def clean_lines(lines: List[str]) -> str:
            cleaned: List[str] = []
            for line in lines:
                stripped = line.strip()
                if stripped in ("```", "```text"):
                    continue
                # Treaty decisions should not contain markdown styling.
                cleaned.append(self._strip_drafting_markup(line).rstrip())

            text = "\n".join(cleaned).strip()
            return self._strip_trailing_noise(text, NOISE_TOKENS).strip()

        lines = chair_response.splitlines()

        # Explicit section extraction.
        capturing = False
        text_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            label = header_label(stripped) if stripped else ""

            if label in TEXT_HEADERS:
                capturing = True
                inline = same_line_after_header(line)
                if inline:
                    text_lines.append(inline)
                continue

            if capturing and label in SECTION_HEADERS:
                break

            if capturing:
                text_lines.append(line)

        extracted = clean_lines(text_lines)
        if len(extracted) > 5:
            return extracted

        # Conservative fallback: start only when a line looks like decision text.
        start_pattern = re.compile(
            r"^\s*(?:The Conference\b|Recalling\b|Recognizing\b|Acknowledging\b|\d+\.)",
            re.IGNORECASE,
        )
        text_lines = []
        capturing = False
        for line in lines:
            stripped = line.strip()
            label = header_label(stripped) if stripped else ""
            if label in SECTION_HEADERS and label not in TEXT_HEADERS:
                if capturing:
                    break
                continue
            if not capturing and start_pattern.search(stripped):
                capturing = True
            if capturing:
                text_lines.append(line)

        extracted = clean_lines(text_lines)
        if len(extracted) > 5:
            return extracted

        return None

    @staticmethod
    def _strip_drafting_markup(text: str) -> str:
        """Remove lightweight markdown emphasis that sometimes leaks into draft text."""
        cleaned = text.replace("**", "")
        emphasis_pattern = re.compile(r"(?<!\w)[_*]([^_*]+?)[_*](?!\w)")
        previous = None
        while cleaned != previous:
            previous = cleaned
            cleaned = emphasis_pattern.sub(r"\1", cleaned)
        return cleaned

    @staticmethod
    def _strip_trailing_noise(text: str, noise_tokens: set) -> str:
        """Remove trailing lines that are clearly not negotiation text."""
        lines = text.split("\n")
        while lines:
            last = lines[-1].strip().lower()
            if last in noise_tokens or not last:
                lines.pop()
            else:
                break
        return "\n".join(lines)

    def save_results(
        self,
        output_dir: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        """Save simulation results to files."""
        out_dir = output_dir or self.config.get("simulation", {}).get(
            "output_dir", "outputs/"
        )
        os.makedirs(out_dir, exist_ok=True)

        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.simulation_name}_{timestamp}"

        results_path = os.path.join(out_dir, f"{base_name}_results.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        text_path = os.path.join(out_dir, f"{base_name}_final_text.txt")
        with open(text_path, "w") as f:
            f.write(self.results.get("final_text", ""))

        log_path = os.path.join(out_dir, f"{base_name}_interaction_log.json")
        with open(log_path, "w") as f:
            json.dump(self.interaction_log, f, indent=2, default=str)

        evolution_path = os.path.join(
            out_dir, f"{base_name}_text_evolution.json"
        )
        with open(evolution_path, "w") as f:
            json.dump(self.text_manager.get_text_evolution(), f, indent=2)

        logger.info(f"Results saved to {out_dir}/{base_name}_*")
        return results_path
