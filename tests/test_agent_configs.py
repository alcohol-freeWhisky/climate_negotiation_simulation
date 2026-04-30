"""Configuration integrity tests for base agent profiles."""

from pathlib import Path
import unittest

import yaml


class TestAgentConfigIntegrity(unittest.TestCase):
    """Ensure base agent YAMLs stay general and internally consistent."""

    @classmethod
    def setUpClass(cls):
        cls.agent_dir = Path("config/agents")
        cls.agent_paths = sorted(cls.agent_dir.glob("*.yaml"))
        cls.agent_configs = [
            yaml.safe_load(path.read_text(encoding="utf-8"))
            for path in cls.agent_paths
        ]
        cls.agent_ids = {
            config["agent_id"]
            for config in cls.agent_configs
        }

    def test_coalition_and_adversary_ids_are_known(self):
        """Relationship metadata should refer to known ids or deliberate externals."""
        allowed_external_ids = {"AILAC"}

        for config in self.agent_configs:
            style = config.get("interaction_style", {})
            for key in ("coalition_partners", "typical_adversaries"):
                for related_id in style.get(key, []):
                    self.assertIn(
                        related_id,
                        self.agent_ids | allowed_external_ids,
                        msg=(
                            f"{config['agent_id']} has unknown {key} id: "
                            f"{related_id}"
                        ),
                    )

    def test_base_agent_profiles_do_not_hardcode_article_6_8(self):
        """
        Base agent charters should remain general.

        Scenario-specific refinement belongs in the scenario file and runtime
        prompting, not in the broad coalition YAMLs.
        """
        forbidden_markers = (
            "Article 6.8",
            "Article 6.9",
            "4/CMA.3",
            "SBSTA work programme under Article 6.8",
        )

        for path in self.agent_paths:
            text = path.read_text(encoding="utf-8")
            for marker in forbidden_markers:
                self.assertNotIn(
                    marker,
                    text,
                    msg=f"{path.name} should stay general, but contains {marker!r}.",
                )

    def test_base_agent_profiles_do_not_embed_scenario_runtime_guidance(self):
        """
        Scenario-specific briefing belongs in scenario files or runtime logic,
        not in the reusable bloc charters.
        """
        forbidden_keys = {
            "runtime_briefing",
            "scenario_runtime_guidance",
            "per_agent_guidance",
        }

        for config in self.agent_configs:
            self.assertTrue(
                forbidden_keys.isdisjoint(config.keys()),
                msg=(
                    f"{config['agent_id']} should not embed scenario guidance in its "
                    "base YAML."
                ),
            )


if __name__ == "__main__":
    unittest.main()
