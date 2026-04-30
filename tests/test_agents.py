"""Tests for negotiation agents."""

import unittest
from unittest.mock import MagicMock, patch
from src.agents.negotiation_agent import NegotiationAgent
from src.llm.llm_backend import LLMResponse
from src.llm.prompt_templates import PromptTemplates


class TestNegotiationAgent(unittest.TestCase):
    """Test cases for NegotiationAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent_config = {
            "agent_id": "TEST_AGENT",
            "display_name": "Test Agent",
            "group_category": "test",
            "hard_facts": {
                "population": {"value": 100, "unit": "millions"},
            },
            "normative_frame": {
                "primary_principles": ["Test principle"],
                "fairness_weights": {
                    "historical_responsibility": 0.5,
                    "current_capability": 0.3,
                    "future_needs": 0.2,
                },
                "key_phrases": ["test phrase"],
            },
            "stance": {
                "mitigation": {
                    "position": "Test position",
                    "red_lines": ["Never concede on X"],
                    "flexibility": "Some flexibility",
                    "priority": "high",
                },
                "adaptation": {
                    "position": "Prioritize resilience and support for vulnerable countries",
                    "red_lines": ["Do not omit adaptation support"],
                    "flexibility": "Moderate",
                    "priority": "very_high",
                },
                "transparency": {
                    "position": "Common reporting with flexibility",
                    "red_lines": ["No intrusive review without support"],
                    "flexibility": "Moderate",
                    "priority": "medium",
                },
            },
            "behavioral_params": {
                "stubbornness": 0.5,
                "compromise_willingness": 0.5,
                "coalition_tendency": 0.7,
                "risk_aversion": 0.8,
                "time_discount": 0.2,
                "leadership_tendency": 0.9,
                "procedural_strictness": 0.3,
                "epistemic_trust": {
                    "ipcc": 0.95,
                    "industry": 0.2,
                    "civil_society": 0.8,
                    "other_parties": 0.5,
                },
            },
            "interaction_style": {
                "typical_opening": "On behalf of Test Agent",
                "coalition_partners": [],
                "typical_adversaries": [],
                "negotiation_tactics": ["test_tactic"],
                "language_patterns": ["We propose"],
            },
            "llm_overrides": {
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        }

        self.global_config = {
            "llm": {"temperature": 0.7, "max_tokens": 2000},
            "agent_defaults": {
                "temperature": 0.7,
                "stance_reinforcement_interval": 3,
            },
            "scenario": {
                "runtime_briefing": {
                    "shared_guidance": [
                        "Stay within the current agenda item.",
                    ],
                    "per_agent_guidance": {
                        "TEST_AGENT": [
                            "If the text becomes overly broad, narrow it before adding detail.",
                        ]
                    },
                },
                "scenario_constraints": {
                    "salient_issues": ["adaptation", "transparency"],
                }
            },
        }

        # Mock LLM backend
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = LLMResponse(
            content="On behalf of Test Agent, we propose to...",
            model="test-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            finish_reason="stop",
        )

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )
        self.assertEqual(agent.agent_id, "TEST_AGENT")
        self.assertEqual(agent.display_name, "Test Agent")
        self.assertEqual(agent.stubbornness, 0.5)
        self.assertIsNotNone(agent.system_prompt)

    def test_opening_statement(self):
        """Test opening statement generation."""
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )
        statement = agent.generate_opening_statement(
            scenario_context="Test context",
            draft_text="Test draft text",
        )
        self.assertIsInstance(statement, str)
        self.assertTrue(len(statement) > 0)
        self.mock_llm.generate.assert_called_once()

    def test_agent_system_prompt_includes_extended_config_sections(self):
        """System prompt should surface fairness, tendencies, and epistemic trust."""
        prompt = PromptTemplates.agent_system_prompt(self.agent_config)

        self.assertIn("FAIRNESS FRAMING", prompt)
        self.assertIn("Historical responsibility: HIGH (0.50)", prompt)
        self.assertIn("Current capability: MODERATE (0.30)", prompt)
        self.assertIn("Future needs / vulnerability: LOW (0.20)", prompt)

        self.assertIn("BEHAVIORAL TENDENCIES:", prompt)
        self.assertIn(
            "Coalition tendency: HIGH (0.70). You strongly prefer speaking as part of a coalition and frequently reference allied blocs",
            prompt,
        )
        self.assertIn(
            "Leadership tendency: VERY HIGH (0.90). You often take initiative to propose bridging text and drive the agenda",
            prompt,
        )
        self.assertIn(
            "Procedural strictness: MODERATE (0.30). You are flexible about procedural form when it helps outcomes",
            prompt,
        )
        self.assertIn(
            "Risk aversion: VERY HIGH (0.80). You prefer conservative, well-defined commitments over ambitious ones",
            prompt,
        )
        self.assertIn(
            "Time discount: LOW (0.20). You take a long-term perspective and are willing to defer benefits",
            prompt,
        )

        self.assertIn("EPISTEMIC TRUST", prompt)
        self.assertIn("IPCC and scientific assessments: VERY HIGH (0.95)", prompt)
        self.assertIn("Industry reports: LOW (0.20)", prompt)
        self.assertIn("Civil society: VERY HIGH (0.80)", prompt)
        self.assertIn("Statements from other parties: HIGH (0.50)", prompt)

    def test_consultation_response(self):
        """Test consultation response generation."""
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )
        response = agent.generate_consultation_response(
            current_text="Test text [with brackets]",
            disputed_points=["Point 1", "Point 2"],
            round_number=1,
            max_rounds=10,
            scenario_context="Test context",
        )
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_first_reading_memory_uses_paragraph_number(self):
        """First-reading memory should track the paragraph, not total turns taken."""
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )
        agent.rounds_participated = 12

        agent.generate_first_reading_response(
            paragraph_number=4,
            paragraph_text="4. Test paragraph text.",
            other_proposals=["Alternative drafting"],
            scenario_context="Test context",
        )

        self.assertEqual(agent.memory.working_memory[-1].round_number, 4)

    def test_consultation_response_uses_compact_memory_context(self):
        """Consultations should avoid duplicating full draft text in memory context."""
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )
        agent.memory.set_core_memory("draft_text", "VERY LONG DRAFT TEXT")
        agent.memory.set_core_memory("scenario_context", "VERY LONG SCENARIO CONTEXT")
        agent.memory.add_statement(
            round_number=1,
            phase="informal_consultations",
            agent_id="OTHER_AGENT",
            content="A long intervention about compromise options.",
            importance=0.5,
        )

        agent.generate_consultation_response(
            current_text="1. Agreed text.\n\n2. [Bracketed option remains here.]",
            disputed_points=["Paragraph 2: [Bracketed option remains here.]"],
            round_number=2,
            max_rounds=10,
            scenario_context="Scenario context in the user prompt",
        )

        messages = self.mock_llm.generate.call_args.kwargs["messages"]
        system_context = "\n".join(
            message["content"]
            for message in messages
            if message["role"] == "system"
        )
        user_prompt = messages[-1]["content"]

        self.assertNotIn("**draft_text**: VERY LONG DRAFT TEXT", system_context)
        self.assertIn("CURRENT NEGOTIATING TEXT EXCERPT", user_prompt)
        self.assertIn("RECENT DISCUSSION SNAPSHOT", user_prompt)

    def test_consultation_response_includes_targeted_focus_when_supplied(self):
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )

        agent.generate_consultation_response(
            current_text="2. Current compromise text.",
            disputed_points=["Paragraph 2(c): unresolved support language"],
            round_number=9,
            max_rounds=10,
            scenario_context="Test context",
            targeted_focus={
                "paragraph_ref": "2(c)",
                "objectors": ["AOSIS", "LMDC"],
                "supporters": ["EU", "EIG"],
                "themes": ["support", "status"],
                "resolution_mode": "split",
                "overloaded": True,
                "overload_details": {
                    "reason": "2 themes across 4 pressure signals",
                    "recommended_resolution": "split",
                },
            },
        )

        messages = self.mock_llm.generate.call_args.kwargs["messages"]
        user_prompt = messages[-1]["content"]

        self.assertIn("LATE-STAGE TARGETED BRIDGE", user_prompt)
        self.assertIn("Target paragraph: 2(c)", user_prompt)
        self.assertIn("Likely objectors on this paragraph: AOSIS, LMDC", user_prompt)
        self.assertIn("ADOPTION FLOOR", user_prompt)
        self.assertIn("PREFERRED IMPROVEMENT", user_prompt)
        self.assertIn("Suggested resolution mode: split", user_prompt)

    def test_stance_reinforcement(self):
        """Test that stance reinforcement triggers at correct intervals."""
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )
        # Not needed at round 0
        self.assertFalse(agent.needs_stance_reinforcement())

        # Simulate rounds
        for _ in range(3):
            agent.increment_round()

        # Should need reinforcement at round 3
        self.assertTrue(agent.needs_stance_reinforcement())

    def test_observe_statement(self):
        """Test that observing statements adds to memory."""
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )
        agent.observe_statement(
            round_number=1,
            phase="consultation",
            speaker_id="OTHER_AGENT",
            content="Other agent says something",
        )
        self.assertEqual(len(agent.memory.working_memory),1)
        self.assertEqual(
            agent.memory.working_memory[0].agent_id, "OTHER_AGENT"
        )

    def test_get_stance_summary(self):
        """Test stance summary generation."""
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )
        summary = agent.get_stance_summary("mitigation")
        self.assertIn("Test position", summary)
        self.assertIn("RED LINE", summary)

    def test_agenda_focus_summary_prioritizes_salient_issues(self):
        """Runtime focus should narrow the general charter to current agenda issues."""
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )
        summary = agent.get_agenda_focus_summary(
            context_text="The paragraph concerns adaptation support and reporting requirements.",
            salient_issues=["adaptation", "transparency"],
            max_issues=2,
        )

        self.assertIn("adaptation", summary.lower())
        self.assertIn("transparency", summary.lower())
        self.assertNotIn("mitigation", summary.lower())

    def test_runtime_briefing_uses_general_charter_and_runtime_guidance(self):
        """Scenario briefings should refine the current agenda without rewriting identity."""
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )
        briefing = agent._build_runtime_briefing(
            "The text focuses on adaptation support and transparency.",
            disputed_points=["reporting requirements", "support for vulnerable countries"],
        )

        self.assertIn("Priority issues for this agenda item", briefing)
        self.assertIn("adaptation", briefing.lower())
        self.assertIn("transparency", briefing.lower())
        self.assertIn("support for vulnerable countries", briefing.lower())
        self.assertIn("stay within the current agenda item", briefing.lower())
        self.assertIn("narrow it before adding detail", briefing.lower())

    def test_final_plenary_response(self):
        """Test final plenary response generation."""
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )
        response = agent.generate_final_plenary_response(
            final_text="Final text for adoption",
            scenario_context="Test context",
        )
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_final_plenary_memory_uses_round_one(self):
        """Final plenary memory should always use round 1 for the phase."""
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )
        agent.rounds_participated = 12

        agent.generate_final_plenary_response(
            final_text="Final text for adoption",
            scenario_context="Test context",
        )

        self.assertEqual(agent.memory.working_memory[-1].round_number, 1)

    def test_opening_prompt_includes_runtime_briefing(self):
        """Opening prompts should include runtime-only scenario briefing text."""
        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, self.global_config
        )
        agent.generate_opening_statement(
            scenario_context="Context on adaptation and transparency.",
            draft_text="Draft text on support for vulnerable countries.",
        )

        messages = self.mock_llm.generate.call_args.kwargs["messages"]
        user_prompt = messages[-1]["content"]

        self.assertIn("SCENARIO-SPECIFIC ISSUE BRIEFING", user_prompt)
        self.assertIn("Possible landing zone", user_prompt)

    def test_final_plenary_red_line_critic_can_revise_response(self):
        """Test optional red-line critic revises unsafe acceptance."""
        global_config = {
            **self.global_config,
            "agent_defaults": {
                **self.global_config["agent_defaults"],
                "red_line_critic": {
                    "enabled": True,
                    "phases": ["final_plenary"],
                    "max_tokens": 200,
                },
            },
            "scenario": {
                "scenario_constraints": {
                    "agent_blocking_conditions": {
                        "TEST_AGENT": [
                            "Final text violates X."
                        ]
                    },
                    "agent_acceptance_conditions": {
                        "TEST_AGENT": [
                            "Final text must protect X."
                        ]
                    },
                }
            },
        }
        self.mock_llm.generate.side_effect = [
            LLMResponse(
                content="ACCEPT We can accept the text.",
                model="test-model",
                usage={"prompt_tokens": 100, "completion_tokens": 20},
                finish_reason="stop",
            ),
            LLMResponse(
                content=(
                    "VERDICT: FAIL\n"
                    "VIOLATED_CONDITION: Final text violates X.\n"
                    "REVISED_RESPONSE:\n"
                    "OPPOSE The text violates our red line on X."
                ),
                model="test-model",
                usage={"prompt_tokens": 100, "completion_tokens": 40},
                finish_reason="stop",
            ),
        ]

        agent = NegotiationAgent(
            self.agent_config, self.mock_llm, global_config
        )
        response = agent.generate_final_plenary_response(
            final_text="Final text violates X.",
            scenario_context="Test context",
        )

        self.assertTrue(response.startswith("OPPOSE"))
        self.assertEqual(self.mock_llm.generate.call_count, 2)


class TestChairAgent(unittest.TestCase):
    """Test cases for ChairAgent."""

    def setUp(self):
        """Set up test fixtures."""
        from src.agents.chair_agent import ChairAgent

        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = LLMResponse(
            content=(
                "REVISED TEXT:\nThe parties agree to...\n\n"
                "PROGRESS SUMMARY:\nSome progress made.\n\n"
                "REMAINING ISSUES:\nFinance still disputed."
            ),
            model="test-model",
            usage={"prompt_tokens": 200, "completion_tokens": 100},
            finish_reason="stop",
        )

        self.global_config = {
            "llm": {"temperature": 0.7, "max_tokens": 2000},
        }

        self.chair = ChairAgent(self.mock_llm, self.global_config)

    def test_chair_initialization(self):
        """Test chair initializes correctly."""
        self.assertIsNotNone(self.chair.system_prompt)
        self.assertEqual(self.chair.rounds_chaired, 0)

    def test_chair_system_prompt_includes_generic_unfccc_drafting_rules(self):
        """Chair prompt should include generic rules about sub-items and core procedural elements."""
        prompt = self.chair.system_prompt

        self.assertIn(
            "When the negotiating text contains enumerated sub-items under an operative paragraph (e.g., (a), (b), (c), (d)), you must preserve the sub-item structure in your revised text. You may reword individual sub-items but you must not merge or collapse them into a single sentence or paragraph. If parties have agreed to delete a sub-item, you may remove it, but do not consolidate remaining items.",
            prompt,
        )
        self.assertIn(
            "In UNFCCC decisions, it is standard practice to include provisions on: (i) a work programme or mandate, (ii) a reporting or progress-review clause, and (iii) an invitation for submissions by Parties. Ensure the final text retains all three elements unless the negotiating parties explicitly agree to remove one.",
            prompt,
        )

    def test_present_paragraph(self):
        """Test paragraph presentation."""
        result = self.chair.present_paragraph(1, "Test paragraph text")
        self.assertIn("Paragraph 1", result)
        self.assertIn("Test paragraph text", result)

    def test_present_paragraph_accepts_preamble_label(self):
        """Test paragraph presentation with a human-facing preambular label."""
        result = self.chair.present_paragraph(
            1,
            "Recalling prior decisions,",
            paragraph_label="Preamble 1",
        )
        self.assertIn("Preamble 1", result)
        self.assertNotIn("Paragraph Preamble 1", result)

    def test_synthesize_round(self):
        """Test round synthesis."""
        proposals = [
            {"agent": "EU", "content": "We propose X"},
            {"agent": "G77", "content": "We propose Y"},
        ]
        result = self.chair.synthesize_round(
            current_text="Current text",
            proposals=proposals,
            round_number=1,
            disputed_points=["Point 1"],
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        self.assertEqual(self.chair.rounds_chaired, 1)

    def test_present_final_text(self):
        """Test final text presentation."""
        result = self.chair.present_final_text("Final text here")
        self.assertIn("Final text here", result)
        self.assertIn("adoption", result.lower())

    def test_revise_for_adoption(self):
        """Test final adoption repair prompt path."""
        result = self.chair.revise_for_adoption(
            candidate_text="1. Acknowledging something and,",
            acceptability_map={
                "likely_accept": ["EU"],
                "likely_object": ["G77_CHINA"],
                "uncertain": [],
                "signals": ["G77_CHINA: likely object - cannot accept paragraph 6"],
            },
            drafting_issues=["Line 1 ends with a dangling conjunction."],
            agent_positions={"G77_CHINA": "We cannot accept this text."},
            scenario_context="Test scenario context",
            preserve_terms=["sustainable development"],
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_declare_outcome_adopted(self):
        """Test outcome declaration when adopted."""
        result = self.chair.declare_outcome(accepted=True, objections=[])
        self.assertIn("ADOPTED", result)

    def test_declare_outcome_not_adopted(self):
        """Test outcome declaration when not adopted."""
        objections = [{"agent": "TEST", "reason": "Cannot accept"}]
        result = self.chair.declare_outcome(accepted=False, objections=objections)
        self.assertIn("NOT", result)
        self.assertIn("TEST", result)


if __name__ == "__main__":
    unittest.main()
