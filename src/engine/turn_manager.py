"""
Turn Manager - Manages speaking order and turn-taking in negotiations.
"""

import random
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class TurnManager:
    """
    Manages the speaking order for negotiation rounds.

    FIX: Respects random_seed for reproducibility and reads per-phase
    speaking_order config.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.randomize = config.get("turns", {}).get("randomize_order", True)
        self.allow_right_of_reply = config.get("turns", {}).get(
            "allow_right_of_reply", True
        )
        self.max_consecutive_passes = config.get("turns", {}).get(
            "max_consecutive_passes", 3
        )

        # Read per-phase speaking order preferences
        phases_config = config.get("phases", {})
        self.opening_order = (
            phases_config.get("opening_statements", {})
            .get("speaking_order", "alphabetical")
        )

        self.agents: List[str] = []
        self.speaking_history: List[Dict[str, Any]] = []
        self.passes: Dict[str, int] = {}
        self.right_of_reply_queue: List[str] = []

        # Internal RNG for reproducible shuffling.
        # Seeded externally via set_seed().
        self._rng = random.Random()

    def set_seed(self, seed: int):
        """
        Set the random seed for reproducible speaking orders.
        Should be called once during simulation initialization.
        """
        self._rng = random.Random(seed)
        logger.debug(f"TurnManager seed set to {seed}")

    def set_agents(self, agent_ids: List[str]):
        """Set the list of participating agents."""
        self.agents = list(agent_ids)
        self.passes = {a: 0 for a in agent_ids}

    def get_speaking_order(
        self,
        round_number: int,
        phase: str = "consultation",
    ) -> List[str]:
        """Get the speaking order for a given round."""
        if not self.agents:
            logger.warning("No agents set. Returning empty speaking order.")
            return []

        if phase == "opening_statements":
            return self._opening_order(round_number)

        elif phase == "first_reading":
            offset = round_number % len(self.agents)
            base = sorted(self.agents)
            order = base[offset:] + base[:offset]
            return order

        elif phase == "consultation":
            order = self._priority_order(round_number)

            if self.right_of_reply_queue:
                reply_order = []
                for agent in self.right_of_reply_queue:
                    if agent in order:
                        order.remove(agent)
                    reply_order.append(agent)
                order = reply_order + order
                self.right_of_reply_queue = []

            return order

        elif phase == "final_plenary":
            return sorted(self.agents)

        return list(self.agents)

    def _opening_order(self, round_number: int) -> List[str]:
        """
        Determine speaking order for opening statements.
        Respects the opening_statements.speaking_order config.
        Only applies randomisation when both the config is set to
        random AND randomize_order is True.
        """
        if self.opening_order == "random" and self.randomize:
            order = list(self.agents)
            self._rng.shuffle(order)
            return order

        # Default: alphabetical, with optional light shuffle
        order = sorted(self.agents)
        if self.randomize and self.opening_order != "alphabetical":
            # Light shuffle: occasionally swap adjacent elements
            for i in range(len(order) - 1):
                if self._rng.random() < 0.3:
                    order[i], order[i + 1] = order[i + 1], order[i]

        return order

    def _priority_order(self, round_number: int) -> List[str]:
        """Create a priority-based speaking order."""
        recent_cutoff = max(0, round_number - 3)
        recent_counts = {a: 0 for a in self.agents}
        for entry in self.speaking_history:
            if entry["round"] >= recent_cutoff:
                agent = entry["agent"]
                if agent in recent_counts:
                    recent_counts[agent] += 1

        scored = []
        for agent in self.agents:
            score = recent_counts[agent] + self._rng.uniform(0, 0.5)
            scored.append((agent, score))

        scored.sort(key=lambda x: x[1])
        return [agent for agent, _ in scored]

    def record_speaking(self, agent_id: str, round_number: int, action: str):
        """Record that an agent spoke."""
        self.speaking_history.append(
            {
                "agent": agent_id,
                "round": round_number,
                "action": action,
            }
        )

        if action == "pass":
            self.passes[agent_id] = self.passes.get(agent_id, 0) + 1
        else:
            self.passes[agent_id] = 0

    def request_right_of_reply(self, agent_id: str):
        """An agent requests the right of reply."""
        if agent_id not in self.agents:
            logger.warning(
                "Ignoring right of reply request from unknown agent %s",
                agent_id,
            )
            return

        if self.allow_right_of_reply and agent_id not in self.right_of_reply_queue:
            self.right_of_reply_queue.append(agent_id)
            logger.debug(f"{agent_id} requested right of reply.")

    def check_all_passed(self) -> bool:
        """Check if all agents have consecutively passed."""
        if not self.agents:
            return True
        return all(
            self.passes.get(a, 0) >= self.max_consecutive_passes
            for a in self.agents
        )

    def reset_passes(self):
        """Reset pass counters."""
        self.passes = {a: 0 for a in self.agents}
