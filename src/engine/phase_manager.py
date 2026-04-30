"""
Phase Manager - Manages the progression through negotiation phases.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class NegotiationPhase(Enum):
    """Enumeration of negotiation phases."""
    INITIALIZATION = "initialization"
    OPENING_STATEMENTS = "opening_statements"
    COALITION_CAUCUS = "coalition_caucus"
    FIRST_READING = "first_reading"
    INFORMAL_CONSULTATIONS = "informal_consultations"
    FINAL_PLENARY = "final_plenary"
    CONCLUDED = "concluded"


class PhaseManager:
    """
    Manages the progression through negotiation phases.
    Determines when to advance to the next phase and
    tracks phase-specific state.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.phase_config = config.get("negotiation", {}).get("phases", {})

        # Phase sequence
        self.phase_sequence = [
            NegotiationPhase.INITIALIZATION,
            NegotiationPhase.OPENING_STATEMENTS,
            NegotiationPhase.COALITION_CAUCUS,
            NegotiationPhase.FIRST_READING,
            NegotiationPhase.INFORMAL_CONSULTATIONS,
            NegotiationPhase.FINAL_PLENARY,
            NegotiationPhase.CONCLUDED,
        ]

        # Filter out disabled phases
        self.active_phases = [NegotiationPhase.INITIALIZATION]
        for phase in self.phase_sequence[1:-1]:
            phase_name = phase.value
            default_enabled = False if phase == NegotiationPhase.COALITION_CAUCUS else True
            if self.phase_config.get(phase_name, {}).get("enabled", default_enabled):
                self.active_phases.append(phase)
        self.active_phases.append(NegotiationPhase.CONCLUDED)

        self.current_phase_index = 0
        self.current_phase = self.active_phases[0]

        # Phase-specific state
        self.phase_rounds: Dict[str, int] = {}
        self.phase_data: Dict[str, Any] = {}

        logger.info(
            f"PhaseManager initialized with phases: "
            f"{[p.value for p in self.active_phases]}"
        )

    @property
    def current_phase_name(self) -> str:
        return self.current_phase.value

    def advance_phase(self) -> Optional[NegotiationPhase]:
        """
        Advance to the next phase.
        Returns the new phase or None if negotiation is concluded.
        """
        if self.current_phase_index < len(self.active_phases) - 1:
            self.current_phase_index += 1
            self.current_phase = self.active_phases[self.current_phase_index]
            self.phase_rounds[self.current_phase.value] = 0
            logger.info(f"Advanced to phase: {self.current_phase.value}")
            return self.current_phase
        else:
            logger.info("Negotiation has concluded.")
            return None

    def increment_round(self):
        """Increment the round counter for the current phase."""
        phase_name = self.current_phase.value
        self.phase_rounds[phase_name] = self.phase_rounds.get(phase_name, 0) + 1

    def get_current_round(self) -> int:
        """Get the current round number for the current phase."""
        return self.phase_rounds.get(self.current_phase.value, 0)

    def get_max_rounds(self) -> int:
        """Get the maximum rounds for the current phase."""
        phase_name = self.current_phase.value
        return self.phase_config.get(phase_name, {}).get("max_rounds", 10)

    def should_advance(
        self,
        convergence_score: float = 0.0,
        blocker_count: int = 0,
        open_paragraphs: int = 0,
    ) -> bool:
        """
        Determine if the current phase should end and advance to next.
        """
        phase_name = self.current_phase.value
        current_round = self.get_current_round()
        max_rounds = self.get_max_rounds()

        # Always advance from initialization
        if self.current_phase == NegotiationPhase.INITIALIZATION:
            return True

        # Opening statements: one round only
        if self.current_phase == NegotiationPhase.OPENING_STATEMENTS:
            return current_round >= 1

        # Coalition caucus: one round only
        if self.current_phase == NegotiationPhase.COALITION_CAUCUS:
            return current_round >= 1

        # First reading: after all paragraphs are read
        if self.current_phase == NegotiationPhase.FIRST_READING:
            return self.phase_data.get("all_paragraphs_read", False)

        # Informal consultations: convergence or max rounds
        if self.current_phase == NegotiationPhase.INFORMAL_CONSULTATIONS:
            phase_cfg = self.phase_config.get("informal_consultations", {})
            threshold = phase_cfg.get("convergence_threshold", 0.8)
            max_likely_objectors = phase_cfg.get(
                "max_likely_objectors_for_advance", 1
            )
            max_open_paragraphs = phase_cfg.get(
                "max_open_paragraphs_for_advance", 1
            )

            if (
                convergence_score >= threshold
                and blocker_count <= max_likely_objectors
                and open_paragraphs <= max_open_paragraphs
            ):
                logger.info(
                    "Convergence threshold reached with manageable remaining "
                    f"blockers: {convergence_score:.2f} >= {threshold}, "
                    f"likely_objectors={blocker_count}, "
                    f"open_paragraphs={open_paragraphs}"
                )
                return True
            if convergence_score >= threshold:
                logger.info(
                    "Convergence score is high, but consultations continue "
                    f"because likely_objectors={blocker_count} "
                    f"(max {max_likely_objectors}) and open_paragraphs={open_paragraphs} "
                    f"(max {max_open_paragraphs})."
                )
            if current_round >= max_rounds:
                logger.info(f"Max rounds reached for informal consultations.")
                return True
            return False

        # Final plenary: one round
        if self.current_phase == NegotiationPhase.FINAL_PLENARY:
            return current_round >= 1

        return False

    def set_phase_data(self, key: str, value: Any):
        """Set phase-specific data."""
        self.phase_data[key] = value

    def get_phase_data(self, key: str, default: Any = None) -> Any:
        """Get phase-specific data."""
        return self.phase_data.get(key, default)

    def is_concluded(self) -> bool:
        """Check if the negotiation has concluded."""
        return self.current_phase == NegotiationPhase.CONCLUDED

    def get_status(self) -> Dict[str, Any]:
        """Get current status summary."""
        return {
            "current_phase": self.current_phase.value,
            "phase_index": self.current_phase_index,
            "total_phases": len(self.active_phases),
            "current_round": self.get_current_round(),
            "max_rounds": self.get_max_rounds(),
            "phase_rounds": dict(self.phase_rounds),
        }
