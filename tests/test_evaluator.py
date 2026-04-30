"""Tests for NegotiationEvaluator."""

import unittest

from src.evaluation.evaluator import NegotiationEvaluator
from src.llm.llm_backend import LLMResponse


class MockLLMBackend:
    """Minimal mock backend for stance-judge tests."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = []

    def generate(
        self,
        messages,
        temperature=None,
        max_tokens=None,
        stop=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
            }
        )
        return LLMResponse(content=self.response_text, model="mock-model")


class TestNegotiationEvaluator(unittest.TestCase):
    """Test evaluator behavior around scenario-scoped stance checks."""

    @staticmethod
    def _heuristic_violation_case():
        interaction_log = [
            {
                "agent": "EU",
                "phase": "final_plenary",
                "content": (
                    "The EU says we can accept text without "
                    "environmental integrity."
                ),
            }
        ]
        agent_configs = {
            "EU": {
                "display_name": "European Union",
                "stance": {
                    "market_mechanisms": {
                        "red_lines": ["Must ensure environmental integrity"],
                    },
                },
            }
        }
        scenario = {
            "scenario_constraints": {
                "salient_issues": ["market_mechanisms"],
            }
        }
        return interaction_log, agent_configs, scenario

    def test_stance_consistency_uses_scenario_conditions_first(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        interaction_log = [
            {
                "agent": "AOSIS",
                "content": (
                    "On behalf of AOSIS, ACCEPT. We accept because the text "
                    "does not substitute for finance and protects vulnerable countries."
                ),
            }
        ]
        agent_configs = {
            "AOSIS": {
                "stance": {
                    "mitigation": {
                        "red_lines": ["Must include 1.5C reference"],
                    },
                }
            }
        }
        scenario = {
            "scenario_constraints": {
                "agent_blocking_conditions": {
                    "AOSIS": [
                        "Final text allows non-market approaches to substitute for finance.",
                    ],
                }
            }
        }

        result = evaluator._evaluate_stance_consistency(
            interaction_log, agent_configs, scenario
        )

        self.assertEqual(
            result["AOSIS"]["condition_source"],
            "scenario_blocking_condition",
        )
        self.assertEqual(result["AOSIS"]["total_violations"], 0)
        self.assertEqual(result["AOSIS"]["consistency_score"], 1.0)

    def test_stance_consistency_falls_back_to_salient_issues(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        interaction_log = [
            {
                "agent": "EU",
                "content": "The EU says we can accept text without environmental integrity.",
            }
        ]
        agent_configs = {
            "EU": {
                "stance": {
                    "market_mechanisms": {
                        "red_lines": ["Must ensure environmental integrity"],
                    },
                    "mitigation": {
                        "red_lines": ["Must reference 1.5C goal"],
                    },
                }
            }
        }
        scenario = {
            "scenario_constraints": {
                "salient_issues": ["market_mechanisms"],
            }
        }

        result = evaluator._evaluate_stance_consistency(
            interaction_log, agent_configs, scenario
        )

        self.assertEqual(result["EU"]["condition_source"], "agent_red_line")
        self.assertEqual(result["EU"]["red_lines_count"], 1)
        self.assertGreater(result["EU"]["total_violations"], 0)

    def test_stance_consistency_ignores_bracketed_acceptance_in_nonfinal_phase(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        interaction_log = [
            {
                "agent": "AOSIS",
                "phase": "first_reading",
                "content": (
                    "On behalf of AOSIS, we ACCEPT the current text of Paragraph 7, "
                    "including the bracketed options. We note our strong preference "
                    "for the option protecting vulnerable countries."
                ),
            }
        ]
        agent_configs = {
            "AOSIS": {
                "stance": {
                    "adaptation": {
                        "red_lines": [
                            "Must protect vulnerable countries and adaptation benefits."
                        ],
                    },
                },
            }
        }
        scenario = {
            "scenario_constraints": {
                "salient_issues": ["adaptation"],
            }
        }

        result = evaluator._evaluate_stance_consistency(
            interaction_log, agent_configs, scenario
        )

        self.assertEqual(result["AOSIS"]["total_violations"], 0)
        self.assertEqual(result["AOSIS"]["consistency_score"], 1.0)

    def test_stance_consistency_llm_judge_disabled_keeps_heuristic_behavior(self):
        interaction_log, agent_configs, scenario = self._heuristic_violation_case()
        mock_backend = MockLLMBackend("YES. This is clearly a breach.")
        evaluator = NegotiationEvaluator(
            {"evaluation": {"metrics": [], "llm_judge_enabled": False}},
            llm_backend=mock_backend,
        )

        result = evaluator._evaluate_stance_consistency(
            interaction_log, agent_configs, scenario
        )

        self.assertEqual(result["EU"]["total_violations"], 1)
        self.assertEqual(result["EU"]["consistency_score"], 0.0)
        self.assertNotIn("llm_confirmed_violations", result["EU"])
        self.assertNotIn("consistency_score_llm_verified", result["EU"])
        self.assertEqual(mock_backend.calls, [])

    def test_stance_consistency_llm_judge_warns_and_skips_without_backend(self):
        interaction_log, agent_configs, scenario = self._heuristic_violation_case()
        evaluator = NegotiationEvaluator(
            {"evaluation": {"metrics": [], "llm_judge_enabled": True}}
        )

        with self.assertLogs("src.evaluation.evaluator", level="WARNING") as logs:
            result = evaluator._evaluate_stance_consistency(
                interaction_log, agent_configs, scenario
            )

        self.assertEqual(result["EU"]["total_violations"], 1)
        self.assertNotIn("llm_confirmed_violations", result["EU"])
        self.assertTrue(
            any("Skipping LLM stance-judge pass" in message for message in logs.output)
        )

    def test_stance_consistency_llm_judge_all_no_clears_confirmed_violations(self):
        interaction_log, agent_configs, scenario = self._heuristic_violation_case()
        mock_backend = MockLLMBackend("NO. The statement is not a genuine breach.")
        evaluator = NegotiationEvaluator(
            {"evaluation": {"metrics": [], "llm_judge_enabled": True}},
            llm_backend=mock_backend,
        )

        result = evaluator._evaluate_stance_consistency(
            interaction_log, agent_configs, scenario
        )

        self.assertEqual(result["EU"]["total_violations"], 1)
        self.assertEqual(result["EU"]["heuristic_violations"], result["EU"]["violations"])
        self.assertEqual(result["EU"]["llm_confirmed_violations"], [])
        self.assertEqual(result["EU"]["consistency_score_llm_verified"], 1.0)
        self.assertEqual(len(mock_backend.calls), 1)

    def test_stance_consistency_llm_judge_all_yes_keeps_heuristic_violations(self):
        interaction_log, agent_configs, scenario = self._heuristic_violation_case()
        mock_backend = MockLLMBackend("YES. The statement directly abandons the red line.")
        evaluator = NegotiationEvaluator(
            {"evaluation": {"metrics": [], "llm_judge_enabled": True}},
            llm_backend=mock_backend,
        )

        result = evaluator._evaluate_stance_consistency(
            interaction_log, agent_configs, scenario
        )

        self.assertEqual(result["EU"]["heuristic_violations"], result["EU"]["violations"])
        self.assertEqual(
            result["EU"]["llm_confirmed_violations"],
            result["EU"]["heuristic_violations"],
        )
        self.assertEqual(
            result["EU"]["consistency_score_llm_verified"],
            result["EU"]["consistency_score"],
        )

    def test_key_clause_match_supports_aliases(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        result = evaluator._check_key_clauses(
            text="Paragraph 3 includes reporting and transparency of activities and outcomes.",
            key_clauses=[
                {
                    "clause": "reporting requirements",
                    "expected": True,
                    "aliases": [
                        "reporting and transparency",
                        "transparency of activities and outcomes",
                    ],
                }
            ],
        )

        self.assertEqual(result["matched"], 1)
        self.assertEqual(
            result["details"][0]["matched_by"],
            "exact_phrase",
        )

    def test_extract_numeric_refs_normalizes_plural_article_paragraph_format(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})

        refs = evaluator._extract_numeric_references(
            "The relationship is addressed in Article 6, paragraphs 2 and 4."
        )

        self.assertEqual(refs, ["6.2", "6.4"])

    def test_extract_numeric_refs_normalizes_singular_article_paragraph_format(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})

        refs = evaluator._extract_numeric_references(
            "Reporting is addressed in Article 13, paragraph 7."
        )

        self.assertEqual(refs, ["13.7"])

    def test_key_clause_match_accepts_article_paragraph_format_for_numeric_refs(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        result = evaluator._check_key_clauses(
            text=(
                "The draft clarifies the relationship to Article 6, paragraphs 2 "
                "and 4 under the Paris Agreement."
            ),
            key_clauses=[
                {
                    "clause": "relationship to Article 6.2 and 6.4",
                    "expected": True,
                }
            ],
        )

        self.assertEqual(result["matched"], 1)
        self.assertEqual(
            result["details"][0]["matched_by"],
            "keyword_overlap",
        )

    def test_key_clause_match_requires_numeric_references_when_present(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        result = evaluator._check_key_clauses(
            text="The text mentions linkages with other relevant approaches under Article 6.",
            key_clauses=[
                {
                    "clause": "relationship to Article 6.2 and 6.4",
                    "expected": True,
                }
            ],
        )

        self.assertEqual(result["matched"], 0)
        self.assertEqual(
            result["details"][0]["matched_by"],
            "numeric_refs",
        )

    def test_key_clause_match_alias_bypasses_numeric_reference_gate(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        result = evaluator._check_key_clauses(
            text="The draft addresses the relationship with other relevant approaches.",
            key_clauses=[
                {
                    "clause": "relationship to Article 6.2 and 6.4",
                    "expected": True,
                    "aliases": [
                        "relationship with other relevant approaches",
                    ],
                }
            ],
        )

        self.assertEqual(result["matched"], 1)
        self.assertEqual(
            result["details"][0]["matched_by"],
            "exact_phrase",
        )

    def test_key_clause_match_marks_bracketed_only_mentions(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        result = evaluator._check_key_clauses(
            text=(
                "1. Agreed operative paragraph.\n\n"
                "[2. Includes reporting and transparency requirements.]"
            ),
            key_clauses=[
                {
                    "clause": "reporting requirements",
                    "expected": True,
                    "aliases": [
                        "reporting and transparency",
                    ],
                }
            ],
        )

        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["matched_anywhere"], 1)
        self.assertTrue(result["details"][0]["found_only_in_brackets"])
        self.assertEqual(result["details"][0]["bracket_status"], "bracketed_only")

    def test_acceptability_uses_final_plenary_actions(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        interaction_log = [
            {"phase": "final_plenary", "agent": "EU", "action": "accept"},
            {"phase": "final_plenary", "agent": "AOSIS", "action": "accept"},
            {"phase": "final_plenary", "agent": "LMDC", "action": "oppose"},
            {"phase": "final_plenary", "agent": "CHAIR", "content": "procedural"},
        ]

        result = evaluator._evaluate_acceptability(interaction_log)

        self.assertEqual(result["accept_count"], 2)
        self.assertEqual(result["oppose_count"], 1)
        self.assertFalse(result["consensus_possible"])
        self.assertEqual(result["blocking_agents"], ["LMDC"])

    def test_summary_scores_include_text_quality_score(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        summary = evaluator._compute_summary_scores(
            {
                "rouge_l": {"rougeL_f": 0.8},
                "bertscore": {"f1": 0.8},
                "key_clause_match": {"accuracy": 0.8},
                "structural_similarity": {"average_similarity": 0.8},
                "bracket_resolution_rate": {"resolution_rate": 0.2},
                "stance_consistency": {"EU": {"consistency_score": 0.4}},
                "process_realism": {"realism_score": 0.6},
                "acceptability": {"acceptability_score": 0.0},
            }
        )

        self.assertIn("text_quality_score", summary)
        self.assertEqual(summary["text_quality_score"], 0.8)
        self.assertNotEqual(summary["text_quality_score"], summary["overall_score"])

    def test_evaluate_surfaces_political_outcome(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        result = evaluator.evaluate(
            simulated_text="",
            reference_text="",
            interaction_log=[
                {"phase": "final_plenary", "agent": "EU", "action": "accept"},
                {"phase": "final_plenary", "agent": "AOSIS", "action": "accept"},
            ],
            agent_configs={},
        )

        self.assertIn("political_outcome", result)
        self.assertIn("adopted", result["political_outcome"])
        self.assertTrue(result["political_outcome"]["adopted"])

    def test_text_quality_score_is_unchanged_by_acceptability(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        base_results = {
            "rouge_l": {"rougeL_f": 0.73},
            "bertscore": {"f1": 0.77},
            "key_clause_match": {"accuracy": 0.81},
            "structural_similarity": {"average_similarity": 0.69},
            "bracket_resolution_rate": {"resolution_rate": 0.5},
            "stance_consistency": {"EU": {"consistency_score": 0.7}},
            "process_realism": {"realism_score": 0.6},
        }

        adopted_summary = evaluator._compute_summary_scores(
            {
                **base_results,
                "acceptability": {
                    "accept_count": 5,
                    "oppose_count": 0,
                    "modify_count": 0,
                    "acceptability_score": 1.0,
                },
            }
        )
        blocked_summary = evaluator._compute_summary_scores(
            {
                **base_results,
                "acceptability": {
                    "accept_count": 2,
                    "oppose_count": 2,
                    "modify_count": 1,
                    "acceptability_score": 0.0,
                },
            }
        )

        self.assertEqual(
            adopted_summary["text_quality_score"],
            blocked_summary["text_quality_score"],
        )
        self.assertNotEqual(
            adopted_summary["overall_score"],
            blocked_summary["overall_score"],
        )

    def test_generate_report_includes_text_quality_score_line(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        report = evaluator.generate_report(
            {
                "overall_score": 0.514,
                "summary_scores": {
                    "text_quality_score": 0.8,
                    "reference_alignment_score": 0.8,
                    "negotiation_quality_score": 0.28,
                },
            }
        )

        self.assertIn("Text quality score: 0.800", report)

    def test_infer_plenary_action_overrides_support_when_content_blocks(self):
        evaluator = NegotiationEvaluator({"evaluation": {"metrics": []}})
        action = evaluator._infer_plenary_action(
            {
                "phase": "final_plenary",
                "agent": "AFRICAN_GROUP",
                "action": "support",
                "content": (
                    "We support the second option. However, we cannot accept "
                    "the text until paragraph 6 is resolved."
                ),
            }
        )

        self.assertEqual(action, "oppose")

    def test_reassuring_statement_is_generic_not_topic_specific(self):
        self.assertTrue(
            NegotiationEvaluator._is_reassuring_statement(
                statement=(
                    "We can support this compromise because the text retains "
                    "benefit-sharing safeguards and protects local participation."
                ),
                condition=(
                    "Final text omits benefit-sharing safeguards for affected communities."
                ),
            )
        )

    def test_reassuring_statement_rejects_missing_keyword_pattern(self):
        self.assertFalse(
            NegotiationEvaluator._is_reassuring_statement(
                statement=(
                    "We can accept the package even without benefit-sharing safeguards."
                ),
                condition=(
                    "Final text omits benefit-sharing safeguards for affected communities."
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
