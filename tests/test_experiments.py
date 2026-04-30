"""Tests for experiment utility helpers."""

import os
import tempfile
import unittest

from experiments.analyze_results import find_result_files
from experiments.run_simulation import (
    apply_runtime_seed,
    apply_runtime_scenario_overrides,
    build_run_output_dir,
    initialize_runtime_seed,
    slugify_path_component,
)


class TestRunSimulationOutputPaths(unittest.TestCase):
    """Test per-run output path helpers."""

    def test_slugify_path_component(self):
        self.assertEqual(
            slugify_path_component("COP Negotiation Simulation"),
            "COP_Negotiation_Simulation",
        )
        self.assertEqual(
            slugify_path_component("  Article 6.8 / DeepSeek  "),
            "Article_6.8_DeepSeek",
        )

    def test_build_run_output_dir(self):
        run_dir = build_run_output_dir(
            base_output_dir="outputs",
            simulation_name="COP Negotiation Simulation",
            timestamp="20260410_150000",
        )

        self.assertEqual(
            run_dir,
            os.path.join(
                "outputs",
                "COP_Negotiation_Simulation",
                "20260410_150000",
            ),
        )


class TestRunSimulationScenarioOverrides(unittest.TestCase):
    """Test CLI overrides applied on top of the default scenario."""

    def test_apply_runtime_scenario_overrides_prefers_cli_paths(self):
        scenario = {
            "draft_text_path": "data/draft_texts/article6_8_draft.txt",
            "reference_text_path": "data/final_texts/article6_8_final.txt",
        }

        updated = apply_runtime_scenario_overrides(
            scenario,
            draft_text_path="data/draft_texts/custom_draft.txt",
            reference_text_path="data/final_texts/custom_final.txt",
        )

        self.assertEqual(
            updated["draft_text_path"],
            "data/draft_texts/custom_draft.txt",
        )
        self.assertEqual(
            updated["reference_text_path"],
            "data/final_texts/custom_final.txt",
        )
        self.assertEqual(
            scenario["draft_text_path"],
            "data/draft_texts/article6_8_draft.txt",
        )

    def test_apply_runtime_scenario_overrides_keeps_defaults_when_absent(self):
        scenario = {
            "draft_text_path": "data/draft_texts/article6_8_draft.txt",
            "reference_text_path": "data/final_texts/article6_8_final.txt",
        }

        updated = apply_runtime_scenario_overrides(scenario)

        self.assertEqual(updated, scenario)


class TestRunSimulationSeedHandling(unittest.TestCase):
    """Test CLI seed plumbing helpers."""

    def test_apply_runtime_seed_sets_llm_seed_when_provided(self):
        config = {
            "llm": {"provider": "deepseek"},
            "simulation": {"random_seed": 42},
        }

        updated = apply_runtime_seed(config, seed=123)

        self.assertEqual(updated["llm"]["seed"], 123)
        self.assertEqual(updated["simulation"]["random_seed"], 123)
        self.assertIsNone(config["llm"].get("seed"))

    def test_apply_runtime_seed_leaves_llm_seed_unset_when_absent(self):
        config = {
            "llm": {"provider": "deepseek"},
            "simulation": {"random_seed": 42},
        }

        updated = apply_runtime_seed(config)

        self.assertIsNone(updated["llm"].get("seed"))
        self.assertEqual(updated["simulation"]["random_seed"], 42)

    def test_output_helpers_remain_deterministic_with_seed(self):
        initialize_runtime_seed(7)
        slug_one = slugify_path_component("COP Negotiation Simulation")
        run_dir_one = build_run_output_dir(
            base_output_dir="outputs",
            simulation_name="COP Negotiation Simulation",
            timestamp="20260410_150000",
        )

        initialize_runtime_seed(7)
        slug_two = slugify_path_component("COP Negotiation Simulation")
        run_dir_two = build_run_output_dir(
            base_output_dir="outputs",
            simulation_name="COP Negotiation Simulation",
            timestamp="20260410_150000",
        )

        self.assertEqual(slug_one, slug_two)
        self.assertEqual(run_dir_one, run_dir_two)


class TestAnalyzeResultsDiscovery(unittest.TestCase):
    """Test recursive discovery of result files."""

    def test_find_result_files_recurses_into_run_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(
                tmpdir,
                "COP_Negotiation_Simulation",
                "20260410_150000",
            )
            os.makedirs(nested)
            result_path = os.path.join(
                nested,
                "COP_Negotiation_Simulation_20260410_150000_results.json",
            )
            ignored_path = os.path.join(nested, "evaluation_20260410_150000.json")

            with open(result_path, "w", encoding="utf-8") as f:
                f.write("{}")
            with open(ignored_path, "w", encoding="utf-8") as f:
                f.write("{}")

            self.assertEqual(find_result_files(tmpdir), [result_path])


if __name__ == "__main__":
    unittest.main()
