"""Tests for engine components."""

import unittest
from unittest.mock import MagicMock, patch
from src.agents.chair_agent import ChairAgent
from src.engine.phase_manager import PhaseManager, NegotiationPhase
from src.engine.turn_manager import TurnManager
from src.engine.amendment_processor import AmendmentProcessor
from src.engine.negotiation_engine import NegotiationEngine
from src.engine.text_manager import TextParagraph, TextManager
from src.llm.llm_backend import LLMResponse
from src.memory.negotiation_memory import NegotiationMemory


class TestPhaseManager(unittest.TestCase):
    """Test cases for PhaseManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "negotiation": {
                "phases": {
                    "opening_statements": {"enabled": True},
                    "first_reading": {"enabled": True},
                    "informal_consultations": {
                        "enabled": True,
                        "max_rounds": 10,
                        "convergence_threshold": 0.8,
                    },
                    "final_plenary": {"enabled": True},
                },
            },
        }
        self.pm = PhaseManager(self.config)

    def test_initialization(self):
        """Test phase manager initializes correctly."""
        self.assertEqual(
            self.pm.current_phase, NegotiationPhase.INITIALIZATION
        )
        self.assertTrue(len(self.pm.active_phases) > 1)

    def test_phase_advancement(self):
        """Test phase advancement."""
        # Start at initialization
        self.assertEqual(
            self.pm.current_phase, NegotiationPhase.INITIALIZATION
        )

        # Advance to opening statements
        next_phase = self.pm.advance_phase()
        self.assertEqual(next_phase, NegotiationPhase.OPENING_STATEMENTS)
        self.assertEqual(
            self.pm.current_phase, NegotiationPhase.OPENING_STATEMENTS
        )

        # Advance to first reading
        next_phase = self.pm.advance_phase()
        self.assertEqual(next_phase, NegotiationPhase.FIRST_READING)

    def test_coalition_caucus_defaults_disabled_for_backward_compat(self):
        """Coalition caucus should stay off unless explicitly enabled."""
        self.assertNotIn(
            NegotiationPhase.COALITION_CAUCUS,
            self.pm.active_phases,
        )

    def test_coalition_caucus_can_be_enabled_between_opening_and_first_reading(self):
        """When enabled, coalition caucus sits between opening and first reading."""
        config = {
            "negotiation": {
                "phases": {
                    "opening_statements": {"enabled": True},
                    "coalition_caucus": {"enabled": True},
                    "first_reading": {"enabled": True},
                    "informal_consultations": {"enabled": False},
                    "final_plenary": {"enabled": False},
                },
            },
        }
        pm = PhaseManager(config)

        self.assertEqual(
            pm.active_phases[:4],
            [
                NegotiationPhase.INITIALIZATION,
                NegotiationPhase.OPENING_STATEMENTS,
                NegotiationPhase.COALITION_CAUCUS,
                NegotiationPhase.FIRST_READING,
            ],
        )

    def test_should_advance_opening(self):
        """Test advancement from opening statements."""
        self.pm.advance_phase()  # To opening
        self.pm.increment_round()
        self.assertTrue(self.pm.should_advance())

    def test_should_advance_consultations_convergence(self):
        """Test advancement from consultations via convergence."""
        # Advance to consultations
        self.pm.advance_phase()  # opening
        self.pm.advance_phase()  # first reading
        self.pm.advance_phase()  # consultations

        self.pm.increment_round()

        # Below threshold
        self.assertFalse(self.pm.should_advance(convergence_score=0.5))

        # Above threshold
        self.assertTrue(self.pm.should_advance(convergence_score=0.9))

    def test_should_not_advance_consultations_when_blockers_remain(self):
        """High convergence alone should not end consultations if coalitions still clash."""
        self.pm.advance_phase()  # opening
        self.pm.advance_phase()  # first reading
        self.pm.advance_phase()  # consultations

        self.pm.increment_round()

        self.assertFalse(
            self.pm.should_advance(
                convergence_score=0.9,
                blocker_count=4,
                open_paragraphs=2,
            )
        )

    def test_should_advance_consultations_max_rounds(self):
        """Test advancement from consultations via max rounds."""
        self.pm.advance_phase()  # opening
        self.pm.advance_phase()  # first reading
        self.pm.advance_phase()  # consultations

        for _ in range(10):
            self.pm.increment_round()

        self.assertTrue(self.pm.should_advance(convergence_score=0.3))

    def test_is_concluded(self):
        """Test conclusion detection."""
        self.assertFalse(self.pm.is_concluded())

        # Advance through all phases
        while not self.pm.is_concluded():
            result = self.pm.advance_phase()
            if result is None:
                break

        self.assertTrue(self.pm.is_concluded())

    def test_get_status(self):
        """Test status reporting."""
        status = self.pm.get_status()
        self.assertIn("current_phase", status)
        self.assertIn("phase_index", status)
        self.assertIn("total_phases", status)

    def test_disabled_phases(self):
        """Test that disabled phases are skipped."""
        config = {
            "negotiation": {
                "phases": {
                    "opening_statements": {"enabled": True},
                    "first_reading": {"enabled": False},  # Disabled
                    "informal_consultations": {"enabled": True, "max_rounds": 5},
                    "final_plenary": {"enabled": True},
                },
            },
        }
        pm = PhaseManager(config)
        self.assertNotIn(
            NegotiationPhase.FIRST_READING, pm.active_phases
        )


class TestTurnManager(unittest.TestCase):
    """Test cases for TurnManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "turns": {
                "randomize_order": False,
                "allow_right_of_reply": True,
                "max_consecutive_passes": 3,
            },
        }
        self.tm = TurnManager(self.config)
        self.tm.set_agents(["AGENT_A", "AGENT_B", "AGENT_C"])

    def test_initialization(self):
        """Test turn manager initializes correctly."""
        self.assertEqual(len(self.tm.agents), 3)

    def test_speaking_order_opening(self):
        """Test opening statement order (alphabetical)."""
        order = self.tm.get_speaking_order(0, "opening_statements")
        self.assertEqual(order, ["AGENT_A", "AGENT_B", "AGENT_C"])

    def test_speaking_order_first_reading_rotation(self):
        """Test first reading rotation."""
        order_0 = self.tm.get_speaking_order(0, "first_reading")
        order_1 = self.tm.get_speaking_order(1, "first_reading")
        # Order should rotate
        self.assertNotEqual(order_0[0], order_1[0])

    def test_record_speaking(self):
        """Test speaking record."""
        self.tm.record_speaking("AGENT_A", 1, "argue")
        self.assertEqual(len(self.tm.speaking_history), 1)
        self.assertEqual(self.tm.passes["AGENT_A"], 0)

    def test_pass_tracking(self):
        """Test pass counter."""
        self.tm.record_speaking("AGENT_A", 1, "pass")
        self.assertEqual(self.tm.passes["AGENT_A"], 1)
        self.tm.record_speaking("AGENT_A", 2, "pass")
        self.assertEqual(self.tm.passes["AGENT_A"], 2)
        # Reset on non-pass action
        self.tm.record_speaking("AGENT_A", 3, "argue")
        self.assertEqual(self.tm.passes["AGENT_A"], 0)

    def test_all_passed(self):
        """Test detection of all agents passing."""
        for agent in ["AGENT_A", "AGENT_B", "AGENT_C"]:
            for i in range(3):
                self.tm.record_speaking(agent, i, "pass")
        self.assertTrue(self.tm.check_all_passed())

    def test_right_of_reply(self):
        """Test right of reply queue."""
        self.tm.request_right_of_reply("AGENT_C")
        order = self.tm.get_speaking_order(1, "consultation")
        # AGENT_C should be first
        self.assertEqual(order[0], "AGENT_C")
        # Queue should be cleared
        self.assertEqual(len(self.tm.right_of_reply_queue), 0)

    def test_reset_passes(self):
        """Test pass counter reset."""
        self.tm.record_speaking("AGENT_A", 1, "pass")
        self.tm.reset_passes()
        self.assertEqual(self.tm.passes["AGENT_A"], 0)


class TestAmendmentProcessor(unittest.TestCase):
    """Test cases for AmendmentProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = AmendmentProcessor()

    def test_parse_add(self):
        """Test parsing ADD amendments."""
        response = "PROPOSE ADD: new paragraph about adaptation finance"
        amendments = self.processor.parse_response(response)
        self.assertTrue(len(amendments) > 0)
        self.assertEqual(amendments[0].action, "add")
        self.assertIn("adaptation finance", amendments[0].proposed_text)

    def test_parse_delete(self):
        """Test parsing DELETE amendments."""
        response = "PROPOSE DELETE: the reference to voluntary contributions"
        amendments = self.processor.parse_response(response)
        self.assertTrue(len(amendments) > 0)
        self.assertEqual(amendments[0].action, "delete")

    def test_parse_modify(self):
        """Test parsing MODIFY amendments."""
        response = 'PROPOSE MODIFY: "shall consider" → "shall implement"'
        amendments = self.processor.parse_response(response)
        self.assertTrue(len(amendments) > 0)
        self.assertEqual(amendments[0].action, "modify")

    def test_parse_support(self):
        """Test parsing SUPPORT statements."""
        response = "SUPPORT: EU's proposal on transparency"
        amendments = self.processor.parse_response(response)
        self.assertTrue(len(amendments) > 0)
        self.assertEqual(amendments[0].action, "support")

    def test_parse_oppose(self):
        """Test parsing OPPOSE statements."""
        response = "OPPOSE: the deletion of CBDR reference"
        amendments = self.processor.parse_response(response)
        self.assertTrue(len(amendments) > 0)
        self.assertEqual(amendments[0].action, "oppose")

    def test_parse_accept(self):
        """Test parsing ACCEPT."""
        response = "ACCEPT"
        amendments = self.processor.parse_response(response)
        self.assertTrue(len(amendments) > 0)
        self.assertEqual(amendments[0].action, "accept")

    def test_parse_pass(self):
        """Test parsing PASS."""
        response = "PASS"
        amendments = self.processor.parse_response(response)
        self.assertTrue(len(amendments) > 0)
        self.assertEqual(amendments[0].action, "pass")

    def test_parse_multiple_amendments(self):
        """Test parsing multiple amendments."""
        response = """PROPOSE ADD: reference to 1.5 degrees
PROPOSE DELETE: the word "voluntary"
PROPOSE MODIFY: "should" → "shall"
"""
        amendments = self.processor.parse_response(response)
        self.assertEqual(len(amendments), 3)
        actions = [a.action for a in amendments]
        self.assertIn("add", actions)
        self.assertIn("delete", actions)
        self.assertIn("modify", actions)

    def test_parse_natural_language(self):
        """Test parsing natural language proposals."""
        response = (
            "We propose to add a reference to loss and damage in this paragraph."
        )
        amendments = self.processor.parse_response(response)
        self.assertTrue(len(amendments) > 0)

    def test_parse_unstructured_opposition(self):
        """Test parsing unstructured opposition."""
        response = (
            "We cannot accept this paragraph as it undermines CBDR principles."
        )
        amendments = self.processor.parse_response(response)
        self.assertTrue(len(amendments) > 0)
        self.assertEqual(amendments[0].action, "oppose")

    def test_primary_action_prefers_opening_oppose_verdict(self):
        """Primary action should follow the opening verdict in plenary."""
        response = """On behalf of the Group of 77 and China, we OPPOSE the current text.

PROPOSE MODIFY: [paragraph 6] → [revised paragraph 6]
"""
        self.assertEqual(
            self.processor.get_primary_action(response),
            "oppose",
        )

    def test_parse_structured_oppose_without_colon(self):
        """Structured parser should capture bare OPPOSE lines."""
        response = """OPPOSE

PROPOSE MODIFY: [paragraph 6] → [revised paragraph 6]
"""
        amendments = self.processor.parse_response(response)
        self.assertEqual(amendments[0].action, "oppose")

    def test_primary_action_support_then_cannot_accept_is_oppose(self):
        """Late hard objections should override earlier support language."""
        response = """On behalf of the African Group, we support the second option.

However, we cannot accept the text until paragraph 6 is resolved.
"""
        self.assertEqual(
            self.processor.get_primary_action(response),
            "oppose",
        )

    def test_summarize_amendments(self):
        """Test amendment summarization."""
        agent_amendments = {
            "EU": [
                self.processor.parse_response("PROPOSE ADD: new text")[0],
            ],
            "G77": [
                self.processor.parse_response("OPPOSE: EU's proposal")[0],
            ],
        }
        summary = self.processor.summarize_amendments(agent_amendments)
        self.assertIn("EU", summary)
        self.assertIn("G77", summary)


class TestNegotiationEngineHelpers(unittest.TestCase):
    """Test helper methods that do not require a live LLM backend."""

    def setUp(self):
        self.engine = NegotiationEngine.__new__(NegotiationEngine)

    @staticmethod
    def _make_agent_stub(agent_id, coalition_partners=None):
        agent = MagicMock()
        agent.agent_id = agent_id
        agent.display_name = agent_id
        agent.config = {
            "agent_id": agent_id,
            "display_name": agent_id,
            "interaction_style": {
                "coalition_partners": coalition_partners or [],
            },
        }
        agent.memory = NegotiationMemory(agent_id=agent_id, working_memory_size=30)
        agent.rounds_participated = 0
        return agent

    def test_merge_scenario_overrides_applies_phase_settings(self):
        config = {
            "negotiation": {
                "phases": {
                    "informal_consultations": {
                        "enabled": True,
                        "max_rounds": 15,
                        "patience": 3,
                    },
                },
            },
        }
        scenario = {
            "scenario_name": "test",
            "phase_overrides": {
                "informal_consultations": {
                    "max_rounds": 12,
                    "key_dispute_points": ["scope"],
                },
            },
        }

        merged = NegotiationEngine._merge_scenario_overrides(config, scenario)

        informal = merged["negotiation"]["phases"]["informal_consultations"]
        self.assertEqual(informal["max_rounds"], 12)
        self.assertEqual(informal["patience"], 3)
        self.assertEqual(informal["key_dispute_points"], ["scope"])
        self.assertEqual(
            config["negotiation"]["phases"]["informal_consultations"]["max_rounds"],
            15,
        )
        self.assertEqual(merged["scenario"]["scenario_name"], "test")

    def test_run_skips_coalition_caucus_when_disabled_by_default(self):
        config = {
            "negotiation": {
                "phases": {
                    "opening_statements": {"enabled": True},
                    "first_reading": {"enabled": True},
                    "informal_consultations": {"enabled": False},
                    "final_plenary": {"enabled": False},
                },
            },
        }
        engine = NegotiationEngine.__new__(NegotiationEngine)
        engine.config = config
        engine.scenario = {"scenario_name": "test"}
        engine.simulation_name = "test"
        engine.phase_manager = PhaseManager(config)
        engine.text_manager = MagicMock()
        engine.text_manager.get_full_text.return_value = ""
        engine.llm = MagicMock()
        engine.llm.get_stats.return_value = {}
        engine.agents = {}
        engine.results = {"phases": {}, "agent_stats": {}}
        engine.interaction_log = []
        engine.total_rounds = 0
        engine.budgeted_rounds = 0
        engine.max_total_rounds = 30
        engine._run_initialization = MagicMock()
        engine._run_opening_statements = MagicMock()
        engine._run_coalition_caucus = MagicMock()
        engine._run_first_reading = MagicMock()
        engine._run_informal_consultations = MagicMock()
        engine._run_final_plenary = MagicMock()

        engine.run()

        engine._run_coalition_caucus.assert_not_called()
        self.assertNotIn("coalition_caucus", engine.results["phases"])

    def test_get_disputed_points_adds_scenario_issues_only_when_text_disputed(self):
        self.engine.config = {
            "negotiation": {
                "phases": {
                    "informal_consultations": {
                        "key_dispute_points": ["scope_and_definition"],
                    },
                },
            },
        }
        self.engine.text_manager = MagicMock()
        self.engine.text_manager.get_disputed_points_summary.return_value = [
            "Paragraph 1: bracketed text...",
        ]

        disputed = self.engine._get_disputed_points()

        self.assertIn("Paragraph 1: bracketed text...", disputed)
        self.assertIn("Scenario issue: scope and definition", disputed)

        self.engine.text_manager.get_disputed_points_summary.return_value = []
        self.assertEqual(self.engine._get_disputed_points(), [])

    def test_text_manager_preserves_operative_numbers_in_disputed_summary(self):
        manager = TextManager({"text": {}})
        manager.load_draft_text(
            "Recalling prior decisions,\n\n"
            "1. Decides to establish a framework.\n\n"
            "2. Requests a work programme."
        )
        manager.paragraphs[0].status = "disputed"
        manager.paragraphs[1].status = "disputed"

        disputed = manager.get_disputed_points_summary()

        self.assertIn("Preamble 1: Recalling prior decisions", disputed[0])
        self.assertIn("Paragraph 1: Decides to establish a framework", disputed[1])

    def test_should_preserve_brackets_when_clean_text_still_faces_objection(self):
        self.engine.config = {
            "negotiation": {
                "phases": {
                    "informal_consultations": {
                        "min_convergence_for_clean_text": 0.95,
                    },
                },
            },
        }

        keep_brackets = self.engine._should_preserve_brackets(
            current_text="1. Decides [option A] [option B]",
            revised_text="1. Decides compromise text",
            round_proposals=[
                {"agent": "EIG", "content": "We cannot accept text without environmental integrity safeguards."},
                {"agent": "EU", "content": "We support compromise text."},
            ],
            convergence={"convergence_score": 0.96, "blocking_issues": []},
        )

        self.assertTrue(keep_brackets)

    def test_should_allow_clean_text_when_no_blockers_remain(self):
        self.engine.config = {
            "negotiation": {
                "phases": {
                    "informal_consultations": {
                        "min_convergence_for_clean_text": 0.95,
                    },
                },
            },
        }

        keep_brackets = self.engine._should_preserve_brackets(
            current_text="1. Decides [option A] [option B]",
            revised_text="1. Decides compromise text",
            round_proposals=[
                {"agent": "EU", "content": "We support the compromise package."},
                {"agent": "EIG", "content": "We support the compromise package."},
            ],
            convergence={"convergence_score": 0.98, "blocking_issues": []},
        )

        self.assertFalse(keep_brackets)

    def test_extract_revised_text_ignores_chair_metadata(self):
        response = """**PROCEDURAL NOTE:** I will now present a revision.
**REVISED TEXT:**
The Conference of the Parties,
1. Decides clean text;
2. Requests follow-up;

**PROGRESS SUMMARY:**
Some progress.

**CHAIR'S NOTE:**
Procedural guidance.
"""
        revised = self.engine._extract_revised_text(response)

        self.assertIn("The Conference of the Parties", revised)
        self.assertIn("1. Decides clean text", revised)
        self.assertNotIn("PROCEDURAL NOTE", revised)
        self.assertNotIn("PROGRESS SUMMARY", revised)
        self.assertNotIn("CHAIR'S NOTE", revised)
        self.assertNotIn("**", revised)

    def test_extract_revised_text_conservative_fallback(self):
        response = """PROCEDURAL NOTE: short note
The Conference of the Parties,
1. Decides clean text;
Done
"""
        revised = self.engine._extract_revised_text(response)

        self.assertEqual(
            revised,
            "The Conference of the Parties,\n1. Decides clean text;",
        )

    def test_finalize_text_for_plenary_debrackets_and_updates_text_manager(self):
        self.engine.text_manager = MagicMock()
        self.engine.text_manager.get_full_text.return_value = "1. Decides [option A] [option B]"
        self.engine.text_manager.get_adoption_ready_text.return_value = "1. Decides option A"

        self.engine._finalize_text_for_plenary()

        self.engine.text_manager.update_full_text.assert_called_once_with(
            "1. Decides option A",
            source="pre_plenary_cleanup",
        )

    def test_finalize_text_for_plenary_returns_clean_text_after_update(self):
        self.engine.text_manager = MagicMock()
        state = {"text": "1. Decides [option A] [option B]"}
        self.engine.interaction_log = []
        self.engine.agents = {}

        def get_full_text():
            return state["text"]

        def update_full_text(text, source):
            state["text"] = text

        self.engine.text_manager.get_full_text.side_effect = get_full_text
        self.engine.text_manager.get_adoption_ready_text.return_value = "1. Decides option A"
        self.engine.text_manager.update_full_text.side_effect = update_full_text

        final_text = self.engine._finalize_text_for_plenary()

        self.assertEqual(final_text, "1. Decides option A")
        self.assertNotIn("[", final_text)

    def test_detect_final_text_issues_flags_dangling_conjunction_and_fragment(self):
        issues = self.engine._detect_final_text_issues(
            "Acknowledging that non-market approaches contribute and,\n"
            "(d) the mechanisms established under Article 6, paragraphs 2 and 4;"
        )

        self.assertEqual(len(issues), 2)
        self.assertIn("dangling conjunction", issues[0].lower())
        self.assertIn("fragment", issues[1].lower())

    def test_detect_final_text_issues_flags_missing_relation_phrase(self):
        issues = self.engine._detect_final_text_issues(
            "Acknowledging that non-market approaches contribute to the implementation "
            "of the Paris Agreement and other approaches under Article 6,"
        )

        self.assertEqual(len(issues), 1)
        self.assertIn("relation phrase", issues[0].lower())

    def test_build_endgame_acceptability_map_classifies_latest_signals(self):
        result = self.engine._build_endgame_acceptability_map(
            {
                "EU": "We can accept this text.",
                "G77_CHINA": "We cannot accept this text.",
                "AOSIS": "We remain concerned about paragraph 2.",
            }
        )

        self.assertEqual(result["likely_accept"], ["EU"])
        self.assertEqual(result["likely_object"], ["G77_CHINA"])
        self.assertEqual(result["uncertain"], ["AOSIS"])

    def test_build_endgame_acceptability_map_treats_hard_textual_demands_as_objectors(self):
        result = self.engine._build_endgame_acceptability_map(
            {
                "LMDC": 'We strongly maintain our position. The text must read "equal importance".',
                "AFRICAN_GROUP": "We accept this text as a foundation.",
            }
        )

        self.assertEqual(result["likely_object"], ["LMDC"])
        self.assertEqual(result["likely_accept"], ["AFRICAN_GROUP"])
        self.assertEqual(
            result["blocker_tags_by_agent"]["LMDC"],
            ["general_acceptability"],
        )

    def test_build_endgame_acceptability_map_collects_blocker_themes(self):
        result = self.engine._build_endgame_acceptability_map(
            {
                "EU": "We can accept this text.",
                "G77_CHINA": (
                    "We cannot accept this package. Support for developing "
                    "countries and capacity-building must remain explicit."
                ),
                "LMDC": (
                    "We cannot accept a final package without CBDR, equity, "
                    "and differentiated reporting."
                ),
            }
        )

        self.assertIn("support", result["blocker_tags_by_agent"]["G77_CHINA"])
        self.assertIn("principles", result["blocker_tags_by_agent"]["LMDC"])
        self.assertIn("reporting", result["blocker_tags_by_agent"]["LMDC"])
        self.assertIn("G77_CHINA", result["blocker_themes"]["support"])
        self.assertIn("LMDC", result["blocker_themes"]["principles"])
        self.assertIn("LMDC", result["blocker_themes"]["reporting"])

    def test_build_endgame_acceptability_map_uses_active_scenario_blocking_conditions(self):
        self.engine.scenario = {
            "scenario_constraints": {
                "agent_blocking_conditions": {
                    "AFRICAN_GROUP": [
                        "Final text omits adaptation, resilience, development needs, or support for vulnerable developing countries."
                    ]
                }
            }
        }
        result = self.engine._build_endgame_acceptability_map(
            {
                "AFRICAN_GROUP": (
                    "We strongly support establishing the framework, but we stress "
                    "that adaptation, resilience, and support for vulnerable "
                    "developing countries must remain central."
                )
            },
            candidate_text=(
                "1. Decides to establish a framework.\n\n"
                "2. Decides that the framework shall aim to:\n"
                "(a) Promote non-market approaches that support sustainable development;"
            ),
        )

        self.assertEqual(result["likely_object"], [])
        self.assertEqual(result["conditional_accept"], ["AFRICAN_GROUP"])
        self.assertIn("support", result["blocker_tags_by_agent"]["AFRICAN_GROUP"])
        self.assertIn(
            "AFRICAN_GROUP",
            result["paragraph_blockers"]["general"]["conditional_acceptors"],
        )

    def test_build_endgame_acceptability_map_keeps_mixed_scenario_signals_uncertain(self):
        self.engine.scenario = {
            "scenario_constraints": {
                "agent_blocking_conditions": {
                    "AOSIS": [
                        "Final text omits adaptation, resilience, vulnerable countries, or support benefits from the scope of non-market approaches.",
                        "Final text weakens environmental integrity safeguards.",
                    ]
                }
            }
        }
        result = self.engine._build_endgame_acceptability_map(
            {
                "AOSIS": (
                    "We support a framework that protects vulnerable countries. "
                    "Environmental integrity and adaptation support both remain essential."
                )
            },
            candidate_text=(
                "2. Decides that the framework shall aim to:\n"
                "(a) Promote non-market approaches that support sustainable development;\n"
                "(d) Ensure environmental integrity and avoid double counting;"
            ),
        )

        self.assertEqual(result["likely_object"], [])
        self.assertEqual(result["uncertain"], ["AOSIS"])

    def test_build_endgame_acceptability_map_detects_unmet_conditional_block(self):
        candidate_text = (
            "2. Decides that the framework shall:\n"
            "(d) Operate in a manner that is consistent with and complementary "
            "to other approaches under the Paris Agreement;"
        )
        result = self.engine._build_endgame_acceptability_map(
            {
                "EU": (
                    "We focus on paragraph 2(d). We strongly support retaining "
                    "the bracketed language on environmental integrity and "
                    "avoiding double counting. We oppose any deletion of this "
                    "critical language."
                )
            },
            candidate_text=candidate_text,
        )

        self.assertEqual(result["likely_object"], ["EU"])
        self.assertIn("integrity", result["blocker_tags_by_agent"]["EU"])
        self.assertIn("EU", result["paragraph_blockers"]["2(d)"]["objectors"])
        self.assertIn("integrity", result["paragraph_blockers"]["2(d)"]["themes"])

    def test_build_endgame_acceptability_map_detects_positive_hard_condition(self):
        candidate_text = (
            "2. Decides that the framework shall aim to:\n"
            "(c) Promote mitigation ambition in the context of sustainable development;"
        )
        result = self.engine._build_endgame_acceptability_map(
            {
                "AOSIS": (
                    "We welcome progress on paragraph 2(c), but the ultimate "
                    "test is whether vulnerable countries and adaptation support "
                    "remain explicit in the package."
                )
            },
            candidate_text=candidate_text,
        )

        self.assertEqual(result["likely_object"], [])
        self.assertEqual(result["conditional_accept"], ["AOSIS"])
        self.assertIn("support", result["blocker_tags_by_agent"]["AOSIS"])
        self.assertIn(
            "AOSIS",
            result["paragraph_blockers"]["2(c)"]["conditional_acceptors"],
        )

    def test_build_endgame_acceptability_map_tracks_status_subordination_theme(self):
        candidate_text = (
            "3. Decides that the framework shall complement and not duplicate "
            "existing arrangements;"
        )
        result = self.engine._build_endgame_acceptability_map(
            {
                "LMDC": (
                    "We welcome movement on paragraph 3, but it must not "
                    "subordinate the framework to other approaches. The "
                    "framework must retain distinct standing and equal footing."
                )
            },
            candidate_text=candidate_text,
        )

        self.assertEqual(result["likely_object"], [])
        self.assertEqual(result["conditional_accept"], ["LMDC"])
        self.assertIn("status", result["blocker_tags_by_agent"]["LMDC"])
        self.assertIn(
            "LMDC",
            result["paragraph_blockers"]["3"]["conditional_acceptors"],
        )
        self.assertIn("status", result["paragraph_blockers"]["3"]["themes"])

    def test_extract_structured_bridge_fields_parses_late_stage_format(self):
        fields = self.engine._extract_structured_bridge_fields(
            "ADOPTION FLOOR: Paragraph 2(d) must prioritize adaptation support.\n"
            "PREFERRED IMPROVEMENT: Add explicit reference to the most vulnerable.\n"
            "CAN ACCEPT WITHOUT PREFERRED IMPROVEMENT: Yes, if the core support clause remains.\n"
            "BRIDGE TEXT: Keep support in 2(d) and move qualifiers to reporting."
        )

        self.assertTrue(fields["present"])
        self.assertIn("Paragraph 2(d)", fields["adoption_floor"])
        self.assertIn("most vulnerable", fields["preferred_improvement"])
        self.assertTrue(fields["can_accept_without_preferred"])
        self.assertIn("move qualifiers", fields["bridge_text"])

    def test_build_endgame_acceptability_map_uses_structured_floor_and_preference(self):
        candidate_text = (
            "2. Decides that the framework shall aim to:\n"
            "(d) Prioritize support for adaptation and resilience in developing country Parties;"
        )
        result = self.engine._build_endgame_acceptability_map(
            {
                "AOSIS": (
                    "ADOPTION FLOOR: Paragraph 2(d) must prioritize support for adaptation "
                    "and resilience in developing country Parties.\n"
                    "PREFERRED IMPROVEMENT: Add explicit reference to the most vulnerable, "
                    "including small island developing States.\n"
                    "CAN ACCEPT WITHOUT PREFERRED IMPROVEMENT: Yes.\n"
                    "BRIDGE TEXT: Keep the current support clause and place vulnerable-country "
                    "specificity in a linked paragraph."
                )
            },
            candidate_text=candidate_text,
        )

        self.assertEqual(result["likely_object"], [])
        self.assertEqual(result["conditional_accept"], ["AOSIS"])
        self.assertIn(
            "AOSIS",
            result["paragraph_blockers"]["2(d)"]["conditional_acceptors"],
        )

    def test_build_endgame_acceptability_map_treats_structured_indispensable_preference_as_object(self):
        candidate_text = (
            "2. Decides that the framework shall aim to:\n"
            "(d) Prioritize support for adaptation and resilience in developing country Parties;"
        )
        result = self.engine._build_endgame_acceptability_map(
            {
                "LMDC": (
                    "ADOPTION FLOOR: Paragraph 2(d) must prioritize support for adaptation "
                    "and resilience in developing country Parties.\n"
                    "PREFERRED IMPROVEMENT: Add explicit reference to CBDR and developed "
                    "country obligations.\n"
                    "CAN ACCEPT WITHOUT PREFERRED IMPROVEMENT: No.\n"
                    "BRIDGE TEXT: Add the CBDR safeguard in reporting language."
                )
            },
            candidate_text=candidate_text,
        )

        self.assertEqual(result["likely_object"], ["LMDC"])
        self.assertIn("principles", result["blocker_tags_by_agent"]["LMDC"])

    def test_select_targeted_consultation_focus_uses_latest_blocker_paragraph(self):
        latest_map = {
            "likely_accept": ["EU", "EIG", "UMBRELLA"],
            "conditional_accept": [],
            "uncertain": ["AOSIS"],
            "paragraph_blockers": {
                "2(d)": {
                    "objectors": ["G77_CHINA", "LMDC"],
                    "conditional_acceptors": [],
                    "themes": ["status", "support"],
                },
                "5": {
                    "objectors": ["AOSIS"],
                    "conditional_acceptors": [],
                    "themes": ["support"],
                },
            },
            "overloaded_paragraphs": {},
        }
        self.engine.phase_manager = MagicMock()
        self.engine.phase_manager.get_phase_data.return_value = latest_map

        focus = self.engine._select_targeted_consultation_focus(
            round_number=11,
            max_rounds=12,
            rounds_without_progress=1,
        )

        self.assertEqual(focus["paragraph_ref"], "2(d)")
        self.assertEqual(focus["objectors"], ["G77_CHINA", "LMDC"])
        self.assertIn("EU", focus["supporters"])
        self.assertIn("status", focus["themes"])
        self.assertEqual(focus["resolution_mode"], "merge")

    def test_select_targeted_consultation_focus_penalizes_overloaded_paragraphs(self):
        latest_map = {
            "likely_accept": ["EU", "EIG"],
            "conditional_accept": ["AOSIS", "AFRICAN_GROUP", "LDC"],
            "likely_object": ["G77_CHINA"],
            "uncertain": [],
            "paragraph_blockers": {
                "3": {
                    "objectors": ["G77_CHINA"],
                    "conditional_acceptors": ["AOSIS", "AFRICAN_GROUP", "LDC"],
                    "themes": ["support", "integrity", "status", "timing"],
                },
                "2(b)": {
                    "objectors": ["G77_CHINA"],
                    "conditional_acceptors": [],
                    "themes": ["support"],
                },
            },
            "overloaded_paragraphs": {
                "3": {
                    "themes": ["support", "integrity", "status", "timing"],
                    "hard_objectors": ["G77_CHINA"],
                    "conditional_acceptors": ["AOSIS", "AFRICAN_GROUP", "LDC"],
                    "reason": "4 themes across 4 pressure signals",
                }
            },
        }
        self.engine.phase_manager = MagicMock()
        self.engine.phase_manager.get_phase_data.return_value = latest_map

        focus = self.engine._select_targeted_consultation_focus(
            round_number=11,
            max_rounds=12,
            rounds_without_progress=1,
        )

        self.assertEqual(focus["paragraph_ref"], "2(b)")
        self.assertFalse(focus["overloaded"])
        self.assertEqual(focus["resolution_mode"], "merge")

    def test_detect_overloaded_paragraphs_recommends_split_for_mixed_political_and_technical_themes(self):
        overloaded = self.engine._detect_overloaded_paragraphs(
            {
                "2(d)": {
                    "objectors": ["EU", "G77_CHINA"],
                    "conditional_acceptors": ["AOSIS", "LDC"],
                    "themes": ["support", "status", "integrity", "reporting"],
                }
            }
        )

        self.assertEqual(
            overloaded["2(d)"]["recommended_resolution"],
            "split",
        )

    def test_round_budget_counts_only_consultations_and_final_plenary(self):
        self.engine.total_rounds = 0
        self.engine.budgeted_rounds = 0

        self.engine._record_round_progress("opening_statements")
        self.engine._record_round_progress("coalition_caucus")
        self.engine._record_round_progress("first_reading")
        self.engine._record_round_progress("informal_consultations")
        self.engine._record_round_progress("final_plenary")

        self.assertEqual(self.engine.total_rounds, 5)
        self.assertEqual(self.engine.budgeted_rounds, 2)

    def test_first_reading_treats_opposition_without_amendment_as_disputed(self):
        self.assertTrue(
            self.engine._first_reading_has_live_disagreement(
                paragraph_actions=["support", "oppose"],
                paragraph_amendments=[],
            )
        )
        self.assertFalse(
            self.engine._first_reading_has_live_disagreement(
                paragraph_actions=["support", "accept", "pass"],
                paragraph_amendments=[],
            )
        )

    def test_get_disputed_points_falls_back_to_first_reading_map(self):
        self.engine.config = {
            "negotiation": {
                "phases": {
                    "informal_consultations": {
                        "key_dispute_points": ["reporting_requirements"],
                    },
                },
            },
        }
        self.engine.text_manager = MagicMock()
        self.engine.text_manager.get_disputed_points_summary.return_value = []
        self.engine.phase_manager = MagicMock()
        self.engine.phase_manager.get_phase_data.return_value = [
            "Paragraph 6: complement and not duplicate..."
        ]

        disputed = self.engine._get_disputed_points()

        self.assertIn("Paragraph 6: complement and not duplicate...", disputed)
        self.assertIn("Scenario issue: reporting requirements", disputed)

    def test_build_structure_guidance_summarizes_preamble_and_anchors(self):
        text = (
            "The Conference of the Parties serving as the meeting of the Parties to the Paris Agreement,\n\n"
            "Recalling the provisions of Article 6.8 of the Paris Agreement,\n\n"
            "1. Requests the Subsidiary Body for Scientific and Technological Advice to develop a work programme.\n\n"
            "2. Invites Parties to submit views via the submission portal on reporting requirements."
        )

        guidance = self.engine._build_structure_guidance(text)

        self.assertIn("Leading preambular paragraphs", guidance)
        self.assertIn("Current operative paragraph skeleton", guidance)
        self.assertIn("Article 6.8", guidance)
        self.assertIn("submission portal", guidance)

    def test_stabilize_revised_text_structure_restores_leading_preamble(self):
        current_text = (
            "The Conference of the Parties serving as the meeting of the Parties to the Paris Agreement,\n\n"
            "Recalling the provisions of Articles 6.8 and 6.9 of the Paris Agreement,\n\n"
            "1. Decides to establish a framework;"
        )
        revised_text = "1. Decides to establish a framework;"

        stabilized = self.engine._stabilize_revised_text_structure(
            current_text=current_text,
            revised_text=revised_text,
        )

        self.assertTrue(
            stabilized.startswith(
                "The Conference of the Parties serving as the meeting of the Parties to the Paris Agreement,"
            )
        )
        self.assertIn("1. Decides to establish a framework;", stabilized)

    def test_stabilize_revised_text_structure_clean_only_skips_bracketed_preamble(self):
        current_text = (
            "Recalling the relevant provisions,\n\n"
            "[Acknowledging unresolved wording,]\n\n"
            "1. Decides to establish a framework;"
        )
        revised_text = "1. Decides to establish a framework;"

        stabilized = self.engine._stabilize_revised_text_structure(
            current_text=current_text,
            revised_text=revised_text,
            clean_only=True,
        )

        self.assertIn("Recalling the relevant provisions,", stabilized)
        self.assertNotIn("[Acknowledging unresolved wording,]", stabilized)

    def test_finalize_text_for_plenary_runs_repair_when_issues_detected(self):
        self.engine.text_manager = MagicMock()
        state = {"text": "Acknowledging that non-market approaches contribute and,"}
        self.engine.agents = {"EU": MagicMock(), "G77_CHINA": MagicMock()}
        self.engine.interaction_log = [
            {
                "phase": "informal_consultations",
                "agent": "EU",
                "content": "We can accept this package.",
            },
            {
                "phase": "informal_consultations",
                "agent": "G77_CHINA",
                "content": "We cannot accept this package.",
            },
        ]
        self.engine.chair = MagicMock()
        self.engine.results = {}

        def get_full_text():
            return state["text"]

        def update_full_text(text, source):
            state["text"] = text

        self.engine.text_manager.get_full_text.side_effect = get_full_text
        self.engine.text_manager.get_adoption_ready_text.return_value = state["text"]
        self.engine.text_manager.update_full_text.side_effect = update_full_text
        self.engine.chair.revise_for_adoption.return_value = (
            "REVISED TEXT:\nAcknowledging that non-market approaches contribute."
        )

        final_text = self.engine._finalize_text_for_plenary()

        self.engine.chair.revise_for_adoption.assert_called_once()
        self.assertEqual(
            final_text,
            "Acknowledging that non-market approaches contribute.",
        )
        self.assertTrue(self.engine.results["pre_plenary_checks"]["repair_applied"])
        kwargs = self.engine.chair.revise_for_adoption.call_args.kwargs
        self.assertEqual(kwargs["scenario_context"], "")
        self.assertEqual(kwargs["preserve_terms"], [])

    def test_finalize_text_for_plenary_includes_preserve_verbatim_section_in_chair_prompt(self):
        self.engine.config = {
            "negotiation": {
                "text": {
                    "preserve_unchanged_paragraphs": True,
                },
            },
        }
        self.engine.scenario = {}
        self.engine.results = {}
        self.engine.interaction_log = [
            {
                "phase": "informal_consultations",
                "agent": "G77_CHINA",
                "content": "We cannot accept this package.",
            },
        ]
        self.engine.agents = {"G77_CHINA": MagicMock()}
        self.engine.text_manager = TextManager(
            {"text": {"preserve_unchanged_paragraphs": True}}
        )
        self.engine.text_manager.load_draft_text(
            "Recalling prior decisions,\n\n"
            "1. Acknowledging something and,"
        )
        self.engine.text_manager.add_amendment(
            agent_id="G77_CHINA",
            paragraph_id=2,
            amendment_type="modify",
            original_text="Acknowledging something and,",
            proposed_text="Acknowledging something.",
        )
        mock_llm = MagicMock()
        mock_llm.generate.return_value = LLMResponse(
            content=(
                "REVISED TEXT:\n"
                "Recalling prior decisions,\n\n"
                "1. Acknowledging something."
            ),
            model="test-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            finish_reason="stop",
        )
        self.engine.chair = ChairAgent(mock_llm, {"llm": {}})

        self.engine._finalize_text_for_plenary()

        prompt = mock_llm.generate.call_args.kwargs["messages"][1]["content"]
        self.assertIn("PRESERVE VERBATIM", prompt)
        preserve_section = prompt.split("PRESERVE VERBATIM", 1)[1]
        self.assertIn("Recalling prior decisions,", preserve_section)

    def test_finalize_text_for_plenary_omits_preserve_verbatim_section_when_disabled(self):
        self.engine.config = {
            "negotiation": {
                "text": {
                    "preserve_unchanged_paragraphs": False,
                },
            },
        }
        self.engine.scenario = {}
        self.engine.results = {}
        self.engine.interaction_log = [
            {
                "phase": "informal_consultations",
                "agent": "G77_CHINA",
                "content": "We cannot accept this package.",
            },
        ]
        self.engine.agents = {"G77_CHINA": MagicMock()}
        self.engine.text_manager = TextManager(
            {"text": {"preserve_unchanged_paragraphs": False}}
        )
        self.engine.text_manager.load_draft_text(
            "Recalling prior decisions,\n\n"
            "1. Acknowledging something and,"
        )
        self.engine.text_manager.add_amendment(
            agent_id="G77_CHINA",
            paragraph_id=2,
            amendment_type="modify",
            original_text="Acknowledging something and,",
            proposed_text="Acknowledging something.",
        )
        mock_llm = MagicMock()
        mock_llm.generate.return_value = LLMResponse(
            content=(
                "REVISED TEXT:\n"
                "Recalling prior decisions,\n\n"
                "1. Acknowledging something."
            ),
            model="test-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            finish_reason="stop",
        )
        self.engine.chair = ChairAgent(mock_llm, {"llm": {}})

        self.engine._finalize_text_for_plenary()

        prompt = mock_llm.generate.call_args.kwargs["messages"][1]["content"]
        self.assertNotIn("PRESERVE VERBATIM", prompt)

    def test_finalize_text_for_plenary_compares_pre_and_post_repair_acceptability(self):
        self.engine.text_manager = MagicMock()
        state = {"text": "1. Decides to establish a framework."}
        self.engine.agents = {"EU": MagicMock()}
        self.engine.interaction_log = [
            {
                "phase": "informal_consultations",
                "agent": "EU",
                "content": "We can accept this package.",
            }
        ]
        self.engine.chair = MagicMock()
        self.engine.results = {}

        def get_full_text():
            return state["text"]

        def update_full_text(text, source):
            state["text"] = text

        baseline_map = {
            "likely_accept": [],
            "conditional_accept": ["EU"],
            "likely_object": ["G77_CHINA"],
            "uncertain": [],
            "signals": [],
            "blocker_tags_by_agent": {},
            "blocker_themes": {},
            "paragraph_blockers": {},
            "overloaded_paragraphs": {},
        }
        repaired_map = {
            "likely_accept": ["EU"],
            "conditional_accept": [],
            "likely_object": [],
            "uncertain": [],
            "signals": [],
            "blocker_tags_by_agent": {},
            "blocker_themes": {},
            "paragraph_blockers": {},
            "overloaded_paragraphs": {},
        }

        self.engine.text_manager.get_full_text.side_effect = get_full_text
        self.engine.text_manager.get_adoption_ready_text.return_value = state["text"]
        self.engine.text_manager.update_full_text.side_effect = update_full_text
        self.engine.chair.revise_for_adoption.return_value = (
            "REVISED TEXT:\n1. Decides to establish a framework."
        )

        with patch.object(
            self.engine,
            "_build_endgame_acceptability_map",
            side_effect=[baseline_map, repaired_map],
        ), patch.object(
            self.engine,
            "_detect_final_text_issues",
            side_effect=[[], []],
        ):
            self.engine._finalize_text_for_plenary()

        checks = self.engine.results["pre_plenary_checks"]
        self.assertTrue(checks["repair_applied"])
        self.assertEqual(checks["post_repair_scorecard"]["hard_objectors"], 0)
        self.assertEqual(checks["accepted_scorecard"]["hard_objectors"], 0)

    def test_finalize_text_for_plenary_falls_back_to_first_reading_positions(self):
        self.engine.text_manager = MagicMock()
        state = {"text": "1. Decides to establish a framework."}
        self.engine.agents = {"EU": MagicMock(), "G77_CHINA": MagicMock()}
        self.engine.interaction_log = [
            {
                "phase": "first_reading",
                "agent": "EU",
                "content": "We can accept this package.",
            },
            {
                "phase": "first_reading",
                "agent": "G77_CHINA",
                "content": "We cannot accept this package without support for developing countries.",
            },
        ]
        self.engine.chair = MagicMock()
        self.engine.results = {}

        def get_full_text():
            return state["text"]

        def update_full_text(text, source):
            state["text"] = text

        self.engine.text_manager.get_full_text.side_effect = get_full_text
        self.engine.text_manager.get_adoption_ready_text.return_value = state["text"]
        self.engine.text_manager.update_full_text.side_effect = update_full_text
        self.engine.chair.revise_for_adoption.return_value = "REVISED TEXT:\n1. Decides to establish a framework."

        self.engine._finalize_text_for_plenary()

        kwargs = self.engine.chair.revise_for_adoption.call_args.kwargs
        self.assertEqual(kwargs["acceptability_map"]["likely_accept"], ["EU"])
        self.assertEqual(kwargs["acceptability_map"]["likely_object"], ["G77_CHINA"])

    def test_run_final_plenary_sets_consensus_status(self):
        self.engine.config = {}
        self.engine.results = {"phases": {}}
        self.engine.interaction_log = []
        self.engine.total_rounds = 0
        self.engine.budgeted_rounds = 0
        self.engine.scenario_context = "Test scenario context"
        self.engine.chair = MagicMock()
        self.engine.chair.present_final_text.return_value = "Chair presentation"
        self.engine.chair.declare_outcome.return_value = "Outcome declared"
        self.engine.phase_manager = MagicMock()
        self.engine.turn_manager = MagicMock()
        self.engine.turn_manager.get_speaking_order.return_value = ["EU", "LMDC"]
        self.engine.amendment_processor = MagicMock()
        self.engine.amendment_processor.parse_response.side_effect = [[], []]
        self.engine.amendment_processor.get_primary_action.side_effect = [
            "accept",
            "oppose",
        ]
        self.engine.agents = {
            "EU": MagicMock(),
            "LMDC": MagicMock(),
        }
        self.engine.agents["EU"].generate_final_plenary_response.return_value = "ACCEPT"
        self.engine.agents["LMDC"].generate_final_plenary_response.return_value = (
            "OPPOSE: We cannot accept this package."
        )

        with patch.object(
            self.engine,
            "_finalize_text_for_plenary",
            return_value="1. Decides to establish a framework.",
        ):
            self.engine._run_final_plenary()

        self.assertEqual(self.engine.results["consensus_status"], "no_consensus")
        self.assertEqual(
            self.engine.results["phases"]["final_plenary"]["consensus_status"],
            "no_consensus",
        )

    def test_extract_revised_text_strips_markdown_emphasis(self):
        chair_response = (
            "REVISED TEXT:\n"
            "_Recalling_ the provisions of Article 6.8,\n\n"
            "1. *Decides* to establish a framework.\n"
        )

        extracted = self.engine._extract_revised_text(chair_response)

        self.assertEqual(
            extracted,
            "Recalling the provisions of Article 6.8,\n\n1. Decides to establish a framework.",
        )

    def test_run_first_reading_keeps_text_open_when_consultations_enabled(self):
        self.engine.config = {
            "negotiation": {
                "phases": {
                    "first_reading": {"max_amendments_per_agent": 5},
                    "informal_consultations": {"enabled": True},
                }
            }
        }
        self.engine.scenario_context = "Test context"
        self.engine.results = {"phases": {}}
        self.engine.interaction_log = []
        self.engine.total_rounds = 0
        self.engine.budgeted_rounds = 0
        self.engine.text_manager = MagicMock()
        paragraph = TextParagraph(
            paragraph_id=1,
            text="Decides to establish a framework.",
            is_numbered=True,
            original_number="1.",
        )
        self.engine.text_manager.paragraphs = [paragraph]
        self.engine.text_manager.get_full_text.return_value = "1. Decides to establish a framework."
        self.engine.text_manager.get_disputed_points_summary.side_effect = (
            lambda: ["Paragraph 1: Decides to establish a framework...."]
            if paragraph.status == "disputed"
            else []
        )
        self.engine.text_manager.add_amendment.side_effect = lambda **kwargs: paragraph.amendments.append(object())
        self.engine.chair = MagicMock()
        self.engine.turn_manager = MagicMock()
        self.engine.turn_manager.get_speaking_order.return_value = ["EU", "G77_CHINA"]
        self.engine.phase_manager = MagicMock()
        self.engine.agents = {
            "EU": MagicMock(),
            "G77_CHINA": MagicMock(),
        }
        self.engine.agents["EU"].display_name = "EU"
        self.engine.agents["G77_CHINA"].display_name = "G77+China"
        self.engine.agents["EU"].generate_first_reading_response.return_value = "SUPPORT"
        self.engine.agents["G77_CHINA"].generate_first_reading_response.return_value = (
            "PROPOSE MODIFY: Add support for developing countries."
        )
        self.engine.amendment_processor = MagicMock()
        self.engine.amendment_processor.parse_response.side_effect = [
            [MagicMock(action="support", original_text="", proposed_text="")],
            [MagicMock(action="modify", original_text="", proposed_text="support for developing countries")],
        ]
        self.engine.amendment_processor.get_primary_action.side_effect = [
            "support",
            "modify",
        ]

        self.engine._run_first_reading()

        self.engine.chair.synthesize_round.assert_not_called()
        self.engine.text_manager.update_full_text.assert_not_called()
        self.assertEqual(paragraph.status, "disputed")
        self.engine.chair.present_paragraph.assert_called_once_with(
            1,
            "Decides to establish a framework.",
            paragraph_label="1",
        )
        eu_kwargs = self.engine.agents["EU"].generate_first_reading_response.call_args.kwargs
        self.assertEqual(eu_kwargs["paragraph_number"], 1)
        self.assertEqual(eu_kwargs["paragraph_text"], "Decides to establish a framework.")
        self.assertEqual(eu_kwargs["scenario_context"], "Test context")
        self.assertEqual(eu_kwargs["paragraph_label"], "1")

    def test_run_coalition_caucus_adds_alignment_notes_for_transitive_clusters(self):
        self.engine.config = {
            "negotiation": {
                "phases": {
                    "coalition_caucus": {
                        "enabled": True,
                        "use_llm": False,
                    }
                }
            }
        }
        self.engine.results = {"phases": {}}
        self.engine.interaction_log = [
            {
                "phase": "opening_statements",
                "agent": "A",
                "content": "We prioritize adaptation finance and resilience.",
            },
            {
                "phase": "opening_statements",
                "agent": "B",
                "content": "We support resilience and finance for vulnerable countries.",
            },
            {
                "phase": "opening_statements",
                "agent": "C",
                "content": "We stress adaptation finance and capacity-building.",
            },
            {
                "phase": "opening_statements",
                "agent": "D",
                "content": "We focus on transparency arrangements.",
            },
        ]
        self.engine.total_rounds = 0
        self.engine.budgeted_rounds = 0
        self.engine.phase_manager = MagicMock()
        self.engine.agents = {
            "A": self._make_agent_stub("A", ["B"]),
            "B": self._make_agent_stub("B", ["C"]),
            "C": self._make_agent_stub("C", []),
            "D": self._make_agent_stub("D", []),
        }

        self.engine._run_coalition_caucus()

        self.assertEqual(len(self.engine.results["phases"]["coalition_caucus"]), 3)
        self.assertEqual(
            self.engine.agents["A"].memory.working_memory[-1].phase,
            "coalition_caucus",
        )
        self.assertIn(
            "COALITION_ALIGNMENT_NOTE",
            self.engine.agents["A"].memory.working_memory[-1].content,
        )
        self.assertIn(
            "B, C",
            self.engine.agents["A"].memory.working_memory[-1].content,
        )
        self.assertEqual(self.engine.agents["D"].memory.working_memory, [])
        self.engine.phase_manager.increment_round.assert_called_once()

    def test_run_coalition_caucus_handles_singletons_without_crashing(self):
        self.engine.config = {
            "negotiation": {
                "phases": {
                    "coalition_caucus": {
                        "enabled": True,
                        "use_llm": False,
                    }
                }
            }
        }
        self.engine.results = {"phases": {}}
        self.engine.interaction_log = [
            {
                "phase": "opening_statements",
                "agent": "SOLO",
                "content": "We prefer to speak in our own capacity.",
            }
        ]
        self.engine.total_rounds = 0
        self.engine.budgeted_rounds = 0
        self.engine.phase_manager = MagicMock()
        self.engine.agents = {
            "SOLO": self._make_agent_stub("SOLO", []),
        }

        self.engine._run_coalition_caucus()

        self.assertEqual(self.engine.results["phases"]["coalition_caucus"], [])
        self.assertEqual(self.engine.agents["SOLO"].memory.working_memory, [])

    def test_coalition_caucus_does_not_change_first_reading_flow(self):
        self.engine.config = {
            "negotiation": {
                "phases": {
                    "coalition_caucus": {"enabled": True, "use_llm": False},
                    "first_reading": {"max_amendments_per_agent": 5},
                    "informal_consultations": {"enabled": True},
                }
            }
        }
        self.engine.scenario_context = "Test context"
        self.engine.results = {"phases": {}}
        self.engine.interaction_log = [
            {
                "phase": "opening_statements",
                "agent": "EU",
                "content": "We support a resilient framework with adaptation support.",
            },
            {
                "phase": "opening_statements",
                "agent": "G77_CHINA",
                "content": "We need adaptation support for developing countries.",
            },
        ]
        self.engine.total_rounds = 0
        self.engine.budgeted_rounds = 0
        self.engine.text_manager = MagicMock()
        paragraph = TextParagraph(
            paragraph_id=1,
            text="Decides to establish a framework.",
            is_numbered=True,
            original_number="1.",
        )
        self.engine.text_manager.paragraphs = [paragraph]
        self.engine.text_manager.get_full_text.return_value = (
            "1. Decides to establish a framework."
        )
        self.engine.text_manager.get_disputed_points_summary.side_effect = (
            lambda: ["Paragraph 1: Decides to establish a framework...."]
            if paragraph.status == "disputed"
            else []
        )
        self.engine.text_manager.add_amendment.side_effect = (
            lambda **kwargs: paragraph.amendments.append(object())
        )
        self.engine.chair = MagicMock()
        self.engine.turn_manager = MagicMock()
        self.engine.turn_manager.get_speaking_order.return_value = ["EU", "G77_CHINA"]
        self.engine.phase_manager = MagicMock()
        self.engine.agents = {
            "EU": self._make_agent_stub("EU", ["G77_CHINA"]),
            "G77_CHINA": self._make_agent_stub("G77_CHINA", ["EU"]),
        }
        self.engine.agents["EU"].generate_first_reading_response.return_value = "SUPPORT"
        self.engine.agents["G77_CHINA"].generate_first_reading_response.return_value = (
            "PROPOSE MODIFY: Add support for developing countries."
        )
        self.engine.amendment_processor = MagicMock()
        self.engine.amendment_processor.parse_response.side_effect = [
            [MagicMock(action="support", original_text="", proposed_text="")],
            [MagicMock(action="modify", original_text="", proposed_text="support for developing countries")],
        ]
        self.engine.amendment_processor.get_primary_action.side_effect = [
            "support",
            "modify",
        ]

        self.engine._run_coalition_caucus()
        self.engine._run_first_reading()

        self.assertIn("coalition_caucus", self.engine.results["phases"])
        self.engine.chair.synthesize_round.assert_not_called()
        self.engine.text_manager.update_full_text.assert_not_called()
        self.assertEqual(paragraph.status, "disputed")
        self.engine.chair.present_paragraph.assert_called_once_with(
            1,
            "Decides to establish a framework.",
            paragraph_label="1",
        )
        eu_kwargs = (
            self.engine.agents["EU"]
            .generate_first_reading_response.call_args.kwargs
        )
        self.assertEqual(eu_kwargs["paragraph_number"], 1)
        self.assertEqual(eu_kwargs["paragraph_text"], "Decides to establish a framework.")
        self.assertEqual(eu_kwargs["scenario_context"], "Test context")
        self.assertEqual(eu_kwargs["paragraph_label"], "1")
        self.assertIn(
            "COALITION_ALIGNMENT_NOTE",
            self.engine.agents["EU"].memory.working_memory[-1].content,
        )

    def test_build_agent_runtime_briefings_is_runtime_only_and_scenario_scoped(self):
        self.engine.config = {
            "negotiation": {
                "phases": {
                    "informal_consultations": {
                        "key_dispute_points": ["scope", "reporting_requirements"],
                    }
                }
            }
        }
        self.engine.scenario_context = "Scenario context"
        mock_agent = MagicMock()
        mock_agent._build_runtime_briefing.return_value = "Runtime briefing"
        self.engine.agents = {"EU": mock_agent}

        briefings = self.engine._build_agent_runtime_briefings("Draft text")

        self.assertEqual(briefings, {"EU": "Runtime briefing"})
        mock_agent._build_runtime_briefing.assert_called_once_with(
            "Scenario context",
            "Draft text",
            disputed_points=["scope", "reporting_requirements"],
            max_issues=4,
        )


if __name__ == "__main__":
    unittest.main()
