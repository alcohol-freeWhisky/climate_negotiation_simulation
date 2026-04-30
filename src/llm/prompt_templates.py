"""
Prompt Templates for Climate Negotiation Simulation.
Contains all system prompts, phase-specific prompts, and utility prompts.
"""

from typing import Dict, List, Any, Optional


def _format_weight_label(value: float) -> str:
    """Map a scalar weight to a qualitative label."""
    if value < 0.25:
        return "LOW"
    if value < 0.5:
        return "MODERATE"
    if value < 0.75:
        return "HIGH"
    return "VERY HIGH"


def _format_fairness_weights(weights: dict) -> str:
    """Format fairness weights into a prompt section."""
    if not isinstance(weights, dict) or not weights:
        return ""

    fairness_fields = [
        ("historical_responsibility", "Historical responsibility"),
        ("current_capability", "Current capability"),
        ("future_needs", "Future needs / vulnerability"),
    ]

    lines = [
        "FAIRNESS FRAMING (use these weights when arguing whether a proposal is fair):"
    ]
    for key, label in fairness_fields:
        if key in weights:
            value = float(weights[key])
            lines.append(f"- {label}: {_format_weight_label(value)} ({value:.2f})")

    return "\n".join(lines) if len(lines) > 1 else ""


def _format_behavioral_tendencies(params: dict) -> str:
    """Format behavioral tendencies into a prose block."""
    if not isinstance(params, dict) or not params:
        return ""

    tendency_fields = [
        (
            "coalition_tendency",
            "Coalition tendency",
            "You strongly prefer speaking as part of a coalition and frequently reference allied blocs",
            "You tend to speak in your own voice rather than through coalitions",
        ),
        (
            "leadership_tendency",
            "Leadership tendency",
            "You often take initiative to propose bridging text and drive the agenda",
            "You prefer to react rather than lead",
        ),
        (
            "procedural_strictness",
            "Procedural strictness",
            "You insist on formal procedure, rules, and transparency",
            "You are flexible about procedural form when it helps outcomes",
        ),
        (
            "risk_aversion",
            "Risk aversion",
            "You prefer conservative, well-defined commitments over ambitious ones",
            "You are willing to accept ambitious language even with some uncertainty",
        ),
        (
            "time_discount",
            "Time discount",
            "You weight near-term benefits heavily over long-term ones",
            "You take a long-term perspective and are willing to defer benefits",
        ),
    ]

    lines = ["BEHAVIORAL TENDENCIES:"]
    for key, label, high_text, low_text in tendency_fields:
        if key in params:
            value = float(params[key])
            tendency_text = high_text if value >= 0.5 else low_text
            lines.append(
                f"- {label}: {_format_weight_label(value)} ({value:.2f}). {tendency_text}"
            )

    return "\n".join(lines) if len(lines) > 1 else ""


def _format_epistemic_trust(trust: dict) -> str:
    """Format epistemic trust scores into a prompt section."""
    if not isinstance(trust, dict) or not trust:
        return ""

    trust_fields = [
        ("ipcc", "IPCC and scientific assessments"),
        ("industry", "Industry reports"),
        ("civil_society", "Civil society"),
        ("other_parties", "Statements from other parties"),
    ]

    lines = ["EPISTEMIC TRUST (how much you credit different information sources):"]
    for key, label in trust_fields:
        if key in trust:
            value = float(trust[key])
            lines.append(f"- {label}: {_format_weight_label(value)} ({value:.2f})")

    return "\n".join(lines) if len(lines) > 1 else ""


class PromptTemplates:
    """Central repository for all prompt templates used in the simulation."""

    # =========================================================================
    # SYSTEM PROMPTS
    # =========================================================================

    @staticmethod
    def preserve_verbatim_section(
        preserve_verbatim_paragraphs: Optional[List[str]] = None,
    ) -> str:
        """Render preserve-verbatim guidance only when there are paragraphs to keep."""
        paragraph_texts = [
            paragraph.strip()
            for paragraph in (preserve_verbatim_paragraphs or [])
            if paragraph and paragraph.strip()
        ]
        if not paragraph_texts:
            return ""

        return (
            "\n\n"
            "PRESERVE VERBATIM (paragraphs below received no amendments during "
            "negotiation — retain them exactly as-is in the final text):\n"
            + "\n\n".join(paragraph_texts)
        )

    @staticmethod
    def agent_system_prompt(agent_config: Dict[str, Any]) -> str:
        """
        Generate the system prompt for a negotiation agent.
        This is the core identity and behavioral instruction.
        """
        agent_id = agent_config["display_name"]
        group_cat = agent_config.get("group_category", "unknown")
        nf = agent_config.get("normative_frame", {})
        principles = nf.get("primary_principles", [])
        key_phrases = nf.get("key_phrases", [])
        style = agent_config.get("interaction_style", {})
        opening = style.get("typical_opening", f"On behalf of {agent_id}")
        tactics = style.get("negotiation_tactics", [])
        language_patterns = style.get("language_patterns", [])
        behavioral = agent_config.get("behavioral_params", {})
        stubbornness = behavioral.get("stubbornness", 0.5)
        compromise_willingness = behavioral.get("compromise_willingness", 0.5)
        coalition_tendency = behavioral.get("coalition_tendency", 0.5)
        risk_aversion = behavioral.get("risk_aversion", 0.5)
        leadership_tendency = behavioral.get("leadership_tendency", 0.5)
        procedural_strictness = behavioral.get("procedural_strictness", 0.5)

        # Build stance summary
        stance_text = ""
        stances = agent_config.get("stance", {})
        for issue, details in stances.items():
            if isinstance(details, dict):
                pos = details.get("position", "No position specified")
                red_lines = details.get("red_lines", [])
                priority = details.get("priority", "medium")
                stance_text += f"\n  - {issue.upper()} (priority: {priority}):\n"
                stance_text += f"    Position: {pos}\n"
                if red_lines:
                    stance_text += f"    RED LINES (never concede): {'; '.join(red_lines)}\n"

        # Build hard facts summary
        facts_text = ""
        hard_facts = agent_config.get("hard_facts", {})
        for fact_name, fact_data in hard_facts.items():
            if isinstance(fact_data, dict):
                val = fact_data.get("value", fact_data.get("description", ""))
                unit = fact_data.get("unit", "")
                facts_text += f"  - {fact_name}: {val} {unit}\n"
            else:
                facts_text += f"  - {fact_name}: {fact_data}\n"

        # Stubbornness instructions
        if stubbornness > 0.7:
            flex_instruction = (
                "You are VERY FIRM in your positions. You rarely make concessions "
                "and only do so on minor issues when you receive significant "
                "concessions in return. You NEVER compromise on your red lines."
            )
        elif stubbornness > 0.4:
            flex_instruction = (
                "You are moderately firm. You can make tactical concessions on "
                "lower-priority issues to gain advantage on high-priority ones, "
                "but you NEVER compromise on your red lines."
            )
        else:
            flex_instruction = (
                "You are relatively flexible and willing to seek compromise. "
                "You actively look for middle ground and are willing to adjust "
                "positions on medium-priority issues. However, you still defend "
                "your core red lines."
            )

        behavioral_instructions = []
        if coalition_tendency >= 0.7:
            behavioral_instructions.append(
                "You usually coordinate closely with coalition partners and will often associate with aligned interventions instead of restating them."
            )
        elif coalition_tendency <= 0.3:
            behavioral_instructions.append(
                "You are comfortable taking distinct positions and do not automatically align with coalition partners."
            )

        if compromise_willingness >= 0.7:
            behavioral_instructions.append(
                "You readily table narrow bridge language on lower-priority issues when it protects your core priorities."
            )
        elif compromise_willingness <= 0.3:
            behavioral_instructions.append(
                "You make concessions sparingly and only when the text clearly protects your core interests."
            )

        if risk_aversion >= 0.7:
            behavioral_instructions.append(
                "You are cautious about ambiguity and avoid accepting text whose downstream implications are unclear."
            )

        if leadership_tendency >= 0.7:
            behavioral_instructions.append(
                "You often take initiative by proposing compromise wording or framing package deals."
            )

        if procedural_strictness >= 0.7:
            behavioral_instructions.append(
                "You are attentive to legal drafting, institutional roles, timelines, and procedural precision."
            )
        elif procedural_strictness <= 0.3:
            behavioral_instructions.append(
                "You focus more on political direction than on highly technical drafting detail."
            )

        behavior_detail = (
            chr(10).join(f"- {instruction}" for instruction in behavioral_instructions)
            if behavioral_instructions
            else "- Balance principle, coalition logic, and drafting practicality."
        )
        fairness_detail = _format_fairness_weights(nf.get("fairness_weights", {}))
        behavioral_tendencies_detail = _format_behavioral_tendencies(behavioral)
        epistemic_trust_detail = _format_epistemic_trust(
            behavioral.get("epistemic_trust", {})
        )
        additional_context = "\n\n".join(
            section
            for section in [
                fairness_detail,
                behavioral_tendencies_detail,
                epistemic_trust_detail,
            ]
            if section
        )
        additional_context_block = f"\n\n{additional_context}" if additional_context else ""

        return f"""You are a negotiator representing {agent_id} in United Nations Framework Convention on Climate Change (UNFCCC) negotiations.

## YOUR IDENTITY
- You represent: {agent_id}
- Category: {group_cat}
- Typical opening: "{opening}"

## KEY FACTS ABOUT YOUR GROUP
{facts_text}

## YOUR CORE PRINCIPLES
{chr(10).join(f"- {p}" for p in principles)}

## YOUR NEGOTIATING POSITIONS
{stance_text}

## YOUR BEHAVIORAL GUIDELINES
{flex_instruction}
{behavior_detail}{additional_context_block}

## YOUR NEGOTIATION STYLE
- Tactics you typically employ: {', '.join(tactics)}
- Key phrases you use: {', '.join(f'"{p}"' for p in key_phrases[:5])}
- Language patterns:
{chr(10).join(f'  - "{p}"' for p in language_patterns[:4])}

## CRITICAL RULES
1. ALWAYS stay in character as {agent_id}. Never break character.
2. NEVER agree to anything that violates your RED LINES.
3. Your positions should be CONSISTENT across rounds. Do not randomly change stances.
4. When making concessions, always get something in return.
5. Use formal diplomatic language appropriate to UNFCCC negotiations.
6. Reference specific articles, principles, and precedents from the Convention and Paris Agreement.
7. When the draft text is presented, respond with SPECIFIC textual amendments—not vague preferences.
8. You may form tactical alliances with like-minded groups.
9. If you have nothing substantive to add, you may "pass" or "associate with" another group's statement.
10. Do not use Markdown formatting such as bold or italic markers in public interventions or draft text.
11. Your configuration is your OVERALL CHARTER across climate negotiations. In any specific agenda item, prioritize only the issues actually raised by the current text, scenario, and chair's questions.
12. Do not import unrelated demands from other negotiating tracks unless the draft text clearly opens that issue.

## OUTPUT FORMAT
Always structure your interventions clearly:
- Start with your group identification
- State your position on the specific issue being discussed
- If proposing text changes, use this format:
  PROPOSE ADD: [new text to add]
  PROPOSE DELETE: [text to remove]
  PROPOSE MODIFY: [original text] → [proposed new text]
  SUPPORT: [reference to another party's proposal]
  OPPOSE: [reference to proposal you oppose, with reasoning]
"""

    @staticmethod
    def chair_system_prompt() -> str:
        """System prompt for the Chair/Facilitator agent."""
        return """You are the CHAIR (facilitator) of a UNFCCC negotiation session on a specific agenda item.

## YOUR ROLE
You are an experienced UNFCCC negotiator serving as Chair. You must be:
- NEUTRAL: You do not advocate for any Party's position
- PROCEDURAL: You manage the process according to UNFCCC rules of procedure
- FACILITATIVE: You help Parties find common ground
- EFFICIENT: You keep negotiations moving forward

## YOUR RESPONSIBILITIES
1. Present the draft text paragraph by paragraph
2. Invite Parties to comment and propose amendments
3. Identify areas of convergence and divergence
4. Propose compromise language when Parties are stuck
5. Summarize progress and outstanding issues
6. Call for consensus when appropriate
7. Manage speaking order and time

## YOUR TOOLS
When proposing compromise text, you should:
- Acknowledge all positions expressed
- Find creative bridging language
- Prefer neutral, procedural formulations over ideological or accusatory wording
- Narrow scope when needed: delete contested detail before adding new detail
- Avoid expansive drafting that introduces new obligations, beneficiaries, institutions, or categories unless Parties have clearly converged on them
- In endgame drafting, convert options into one clean formulation with no square brackets
- Use standard UNFCCC compromise techniques:
  * "Chapeau" language that frames the operative text
  * Preambular references that acknowledge principles without creating obligations
  * "As appropriate" or "as applicable" for soft differentiation
  * "Shall" vs "should" vs "may" for different obligation levels
  * Footnotes for definitional disagreements

## OUTPUT FORMAT
Structure your interventions as:
- PROCEDURAL NOTE: [process guidance]
- TEXT PRESENTED: [the paragraph/section being discussed]
- SUMMARY: [summary of positions expressed]
- CHAIR'S PROPOSAL: [compromise text if appropriate]
- NEXT STEPS: [what happens next]

## CRITICAL RULES
1. Never take sides
2. Always acknowledge all positions before proposing compromise
3. Use standard UN procedural language
4. Avoid sharp, absolutist, or politically loaded wording when a softer procedural formulation can do the job
5. If consensus is not possible, streamline the text by deleting the most contested detail rather than expanding the package
6. Do not use Markdown bold or italic formatting in proposed draft text
7. When the negotiating text contains enumerated sub-items under an operative paragraph (e.g., (a), (b), (c), (d)), you must preserve the sub-item structure in your revised text. You may reword individual sub-items but you must not merge or collapse them into a single sentence or paragraph. If parties have agreed to delete a sub-item, you may remove it, but do not consolidate remaining items.
8. In UNFCCC decisions, it is standard practice to include provisions on: (i) a work programme or mandate, (ii) a reporting or progress-review clause, and (iii) an invitation for submissions by Parties. Ensure the final text retains all three elements unless the negotiating parties explicitly agree to remove one.
"""

    # =========================================================================
    # PHASE-SPECIFIC PROMPTS
    # =========================================================================

    @staticmethod
    def opening_statement_prompt(
        agent_name: str,
        scenario_context: str,
        draft_text: str,
        agenda_focus: Optional[str] = None,
        scenario_briefing: Optional[str] = None,
    ) -> str:
        """Prompt for the opening statements phase."""
        focus_section = ""
        if agenda_focus:
            focus_section = f"""
## AGENDA-SPECIFIC FOCUS FROM YOUR GENERAL CHARTER
Use these issues as your main lens for this agenda item:
{agenda_focus}
"""
        briefing_section = ""
        if scenario_briefing:
            briefing_section = f"""
## SCENARIO-SPECIFIC ISSUE BRIEFING
Use this runtime briefing to interpret your general charter for the current text:
{scenario_briefing}
"""
        return f"""The Chair has opened the negotiation session on the following agenda item.

## CONTEXT
{scenario_context}

## DRAFT TEXT UNDER CONSIDERATION
{draft_text}
{focus_section}
{briefing_section}

## YOUR TASK
Deliver your opening statement. You should:
1. State your group's general position on this agenda item
2. Identify your key priorities (what you want to see in the final text)
3. Signal your main concerns with the current draft
4. Indicate areas where you see potential for agreement

Keep your statement focused and under 300 words. This is a formal plenary statement.
"""

    @staticmethod
    def coalition_caucus_alignment_prompt(
        agent_config: Dict[str, Any],
        allies_statements: List[Dict[str, str]],
    ) -> str:
        """Prompt for a short internal coalition-alignment reflection."""
        agent_name = agent_config.get(
            "display_name",
            agent_config.get("agent_id", "your delegation"),
        )
        allies_text = "No allied statements were provided."
        if allies_statements:
            lines = []
            for ally in allies_statements:
                ally_name = ally.get("agent", "ALLY")
                content = ally.get("content", "").strip()
                lines.append(f"- {ally_name}: {content}")
            allies_text = "\n".join(lines)

        return f"""You are preparing a short internal coalition-caucus reflection for {agent_name}.

## ALLIES' OPENING STATEMENTS
{allies_text}

## YOUR TASK
Write a short internal note that:
1. Identifies where your coalition broadly aligns with your own position
2. Flags any notable emphasis differences you should track in first reading
3. Suggests how to keep your coalition messaging coordinated

Keep it concise, practical, and under 120 words.
This is private internal reflection, not a public intervention.
Do not use Markdown bold or italic formatting.
"""

    @staticmethod
    def first_reading_prompt(
        agent_name: str,
        paragraph_number: int,
        paragraph_text: str,
        other_proposals: List[str],
        scenario_context: str,
        paragraph_label: Optional[str] = None,
        agenda_focus: Optional[str] = None,
        scenario_briefing: Optional[str] = None,
    ) -> str:
        """Prompt for the first reading phase (paragraph-by-paragraph)."""
        display_label = str(paragraph_label or paragraph_number)
        proposals_text = ""
        if other_proposals:
            proposals_text = "\n## PROPOSALS ALREADY MADE BY OTHER PARTIES\n"
            for p in other_proposals:
                proposals_text += f"- {p}\n"

        focus_section = ""
        if agenda_focus:
            focus_section = f"""
## AGENDA-SPECIFIC FOCUS FROM YOUR GENERAL CHARTER
For this paragraph, prioritize:
{agenda_focus}
"""
        briefing_section = ""
        if scenario_briefing:
            briefing_section = f"""
## SCENARIO-SPECIFIC ISSUE BRIEFING
Use this runtime briefing to sharpen your response to this paragraph:
{scenario_briefing}
"""

        return f"""We are in the FIRST READING phase, going through the text paragraph by paragraph.

## CURRENT TEXT ITEM ({display_label})
{paragraph_text}

## CONTEXT
{scenario_context}
{proposals_text}
{focus_section}
{briefing_section}

## YOUR TASK
Review this paragraph and respond with ONE of the following:
1. PROPOSE AMENDMENTS: Specific textual changes using the format:
   - PROPOSE ADD: [text]
   - PROPOSE DELETE: [text]
   - PROPOSE MODIFY: [original] → [new text]
2. SUPPORT: If you support another party's proposal, state which one
3. NO TEXTUAL CHANGE THIS ROUND: If you do not need an amendment now, say so clearly without treating this as final agreement
4. PASS: If you need more time to consider

Be SPECIFIC about text changes. Do not make vague statements.
This is a tabling-and-reaction stage, not a final consensus check.
Do not present your response as final adoption of the paragraph.
Limit to maximum 3 amendments per paragraph.
Do not use Markdown bold or italic formatting.
"""

    @staticmethod
    def informal_consultation_prompt(
        agent_name: str,
        current_text: str,
        disputed_points: List[str],
        round_number: int,
        max_rounds: int,
        recent_history: str,
        scenario_context: str,
        agenda_focus: Optional[str] = None,
        scenario_briefing: Optional[str] = None,
        stance_reminder: Optional[str] = None,
        targeted_focus: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Prompt for informal consultations (the core negotiation phase)."""
        stance_section = ""
        if stance_reminder:
            stance_section = f"""
## STANCE REMINDER
Remember your core positions and red lines:
{stance_reminder}
"""

        focus_section = ""
        if agenda_focus:
            focus_section = f"""
## AGENDA-SPECIFIC FOCUS FROM YOUR GENERAL CHARTER
For this round, your most relevant issues are:
{agenda_focus}
"""
        briefing_section = ""
        if scenario_briefing:
            briefing_section = f"""
## SCENARIO-SPECIFIC ISSUE BRIEFING
Use this runtime briefing to focus on the contested parts of this package:
{scenario_briefing}
"""
        targeted_section = ""
        targeted_guidance = ""
        if targeted_focus:
            paragraph_ref = targeted_focus.get("paragraph_ref", "general")
            objectors = ", ".join(targeted_focus.get("objectors", [])) or "Not identified"
            conditional_acceptors = (
                ", ".join(targeted_focus.get("conditional_acceptors", []))
                or "Not identified"
            )
            supporters = ", ".join(targeted_focus.get("supporters", [])) or "Not identified"
            themes = ", ".join(targeted_focus.get("themes", [])) or "general acceptability"
            resolution_mode = targeted_focus.get("resolution_mode", "merge")
            overloaded_note = ""
            if targeted_focus.get("overloaded"):
                overload_reason = (
                    targeted_focus.get("overload_details", {}).get("reason")
                    or "too many distinct demands are being packed into one paragraph"
                )
                recommended_resolution = (
                    targeted_focus.get("overload_details", {}).get(
                        "recommended_resolution",
                        resolution_mode,
                    )
                )
                overloaded_note = f"""
This paragraph appears overloaded: {overload_reason}
Recommended resolution mode: {recommended_resolution}
Treat this as a structural closing problem. Separate your minimum adoption floor from any preferred additions, and identify if overflow language belongs in a linked procedural paragraph instead.
"""
            targeted_section = f"""
## LATE-STAGE TARGETED BRIDGE
This round is narrowed to a single remaining blocker paragraph.
Target paragraph: {paragraph_ref}
Likely objectors on this paragraph: {objectors}
Reservation-based parties on this paragraph: {conditional_acceptors}
Parties currently closer to the package: {supporters}
Main blocker themes: {themes}
Suggested resolution mode: {resolution_mode}
{overloaded_note}
"""
            targeted_guidance = """
- This is a late-stage bridge round: focus first on the target paragraph
- Do not reopen unrelated paragraphs unless a very small linked procedural fix is indispensable
- Offer the smallest workable bridge that protects your red lines
- If your concern is about status, hierarchy, or subordination, state the exact relational safeguard you need
- Distinguish clearly between your ADOPTION FLOOR and your PREFERRED IMPROVEMENT
- Say whether you can accept the package without your preferred improvement
- If the target paragraph is overloaded, identify which overflow issue could move to a procedural or linked paragraph instead of expanding the target sentence
- If the suggested resolution mode is split, say whether the issue should become a separate subparagraph or a separate operative paragraph
- If the suggested resolution mode is relocate, say which clause belongs in work programme, reporting, governance, or another linked procedural paragraph
"""

        return f"""We are in INFORMAL CONSULTATIONS (Round {round_number}/{max_rounds}).

## CURRENT NEGOTIATING TEXT EXCERPT
{current_text}

## KEY DISPUTED POINTS
{chr(10).join(f"- {dp}" for dp in disputed_points)}

## RECENT DISCUSSION SNAPSHOT
{recent_history}

## CONTEXT
{scenario_context}
{focus_section}
{briefing_section}
{stance_section}
{targeted_section}

## YOUR TASK
This is an informal setting. You may:
1. ARGUE for your position on disputed points (with reasoning)
2. PROPOSE specific compromise language
3. ACCEPT another party's proposal (with or without modifications)
4. OPPOSE a proposal (must give reasons)
5. LINK issues (offer concession on one point for gain on another)
6. PASS if you have nothing to add this round

Guidelines:
- Focus on the most important disputed points for your group
- Address at most TWO disputed points in this round
- Be strategic: consider trade-offs between issues
- Use PASS or SUPPORT when another group has already stated your position
- The text excerpt prioritizes unresolved paragraphs; do not assume omitted sections are reopened
- If you sense convergence, help build consensus
- If round {round_number} is getting close to {max_rounds}, consider whether compromise is better than no agreement
- Remember: in UNFCCC, no agreement means the issue carries over to the next session
- Do not accept a compromise that violates your red lines or scenario-specific blocking conditions
- Keep agenda discipline: protect your relevant interests without reopening unrelated climate tracks
{targeted_guidance}

If this is a late-stage targeted bridge round, prefer this compact structure:
ADOPTION FLOOR: [the minimum text or safeguard you need]
PREFERRED IMPROVEMENT: [the stronger text you still want]
CAN ACCEPT WITHOUT PREFERRED IMPROVEMENT: [yes/no]
BRIDGE TEXT: [the smallest workable wording or linked procedural fix]
RESOLUTION MODE VIEW: [merge/split/relocate + one short reason]

Respond in 160 words or less. Be concrete and specific.
Do not use Markdown bold or italic formatting.
"""

    @staticmethod
    def chair_synthesis_prompt(
        current_text: str,
        all_proposals: List[Dict[str, str]],
        round_number: int,
        disputed_points: List[str],
        structure_guidance: str = "",
        preserve_verbatim_paragraphs: Optional[List[str]] = None,
    ) -> str:
        """Prompt for chair to synthesize proposals into revised text."""
        proposals_text = ""
        for p in all_proposals:
            proposals_text += f"\n{p['agent']}: {p['content']}\n"
        preserve_section = PromptTemplates.preserve_verbatim_section(
            preserve_verbatim_paragraphs
        )
        structure_section = ""
        if structure_guidance:
            structure_section = f"""
## STRUCTURAL AND LEGAL-DRAFTING ANCHORS
Use these anchors to preserve the document's architecture while revising disputed text:
{structure_guidance}
"""

        return f"""As Chair, you need to synthesize the proposals from Round {round_number}.

## CURRENT TEXT
{current_text}
{preserve_section}

## PROPOSALS FROM PARTIES
{proposals_text}

## OUTSTANDING DISPUTED POINTS
{chr(10).join(f"- {dp}" for dp in disputed_points)}
{structure_section}

## YOUR TASK
1. Identify where convergence exists and clean up agreed text (remove brackets)
2. Where divergence remains, either:
   a. Propose one compromise formulation that narrows scope and softens the wording
   b. Use brackets only as a temporary mid-negotiation fallback
3. Prefer deletion, simplification, or procedural language over expansive drafting.
4. Avoid sharp terms, hierarchy claims, or new obligations unless they are clearly supported.
5. Before writing the final package, consider which Parties are likely to accept or object and avoid premature convergence if a hardliner red line remains.
6. Summarize progress and remaining issues
7. Preserve already-agreed preambular paragraphs, paragraph skeleton, and explicit procedural anchors unless Parties are actively contesting them.
8. Do not replace a specific existing legal or procedural anchor with a vaguer paraphrase unless the specific anchor itself is the source of disagreement.

Critical formatting rules:
- The REVISED TEXT section must contain ONLY the draft decision text.
- Do not put PROCEDURAL NOTE, PROGRESS SUMMARY, REMAINING ISSUES, CHAIR'S NOTE, markdown bold markers, or commentary inside REVISED TEXT.
- Preserve numbered paragraph boundaries. Put a blank line between numbered paragraphs.
- If you include unresolved options, keep them inside the relevant numbered paragraph and keep them minimal.
- Do not use Markdown bold or italic formatting anywhere in the response.

Format your output as:
REVISED TEXT:
[The full revised text. Prefer one clean compromise formulation; use square brackets only if strictly necessary mid-negotiation.]

PROGRESS SUMMARY:
[What was resolved this round]

REMAINING ISSUES:
[What still needs resolution]

CHAIR'S NOTE:
[Any procedural guidance or suggestions for next round]
"""

    @staticmethod
    def chair_finalization_prompt(
        candidate_text: str,
        acceptability_map: Dict[str, Any],
        drafting_issues: List[str],
        agent_positions: Dict[str, str],
        scenario_context: str = "",
        preserve_terms: Optional[List[str]] = None,
        structure_guidance: str = "",
        preserve_verbatim_paragraphs: Optional[List[str]] = None,
    ) -> str:
        """Prompt for the final pre-plenary clean-up by the Chair."""
        likely_accept = ", ".join(acceptability_map.get("likely_accept", [])) or "None identified"
        conditional_accept = (
            ", ".join(acceptability_map.get("conditional_accept", []))
            or "None identified"
        )
        likely_object = ", ".join(acceptability_map.get("likely_object", [])) or "None identified"
        uncertain = ", ".join(acceptability_map.get("uncertain", [])) or "None identified"

        signal_lines = []
        for signal in acceptability_map.get("signals", [])[:8]:
            signal_lines.append(f"- {signal}")
        signals_text = "\n".join(signal_lines) if signal_lines else "- No strong signals extracted"
        blocker_theme_lines = []
        for theme, agents in acceptability_map.get("blocker_themes", {}).items():
            pretty_theme = theme.replace("_", " ")
            blocker_theme_lines.append(f"- {pretty_theme}: {', '.join(agents)}")
        blocker_themes_text = (
            "\n".join(blocker_theme_lines)
            if blocker_theme_lines
            else "- No recurring blocker themes identified"
        )
        blocker_agent_lines = []
        for agent_id, tags in acceptability_map.get("blocker_tags_by_agent", {}).items():
            pretty_tags = ", ".join(tag.replace("_", " ") for tag in tags)
            blocker_agent_lines.append(f"- {agent_id}: {pretty_tags}")
        blocker_agents_text = (
            "\n".join(blocker_agent_lines)
            if blocker_agent_lines
            else "- No theme-tagged objectors identified"
        )
        paragraph_lines = []
        for paragraph_ref, details in acceptability_map.get("paragraph_blockers", {}).items():
            objectors = ", ".join(details.get("objectors", [])) or "None identified"
            themes = ", ".join(details.get("themes", [])) or "no dominant theme"
            paragraph_lines.append(
                f"- {paragraph_ref}: objectors={objectors}; themes={themes}"
            )
        paragraph_text = (
            "\n".join(paragraph_lines)
            if paragraph_lines
            else "- No paragraph-specific blocker clusters identified"
        )
        overloaded_lines = []
        for paragraph_ref, details in acceptability_map.get(
            "overloaded_paragraphs", {}
        ).items():
            themes = ", ".join(details.get("themes", [])) or "general pressure"
            overloaded_lines.append(
                f"- {paragraph_ref}: {details.get('reason', 'overloaded paragraph')}; themes={themes}; recommended_resolution={details.get('recommended_resolution', 'merge')}"
            )
        overloaded_text = (
            "\n".join(overloaded_lines)
            if overloaded_lines
            else "- No overloaded paragraphs identified"
        )

        drafting_text = "\n".join(f"- {issue}" for issue in drafting_issues) if drafting_issues else "- No explicit drafting issues detected"
        preserve_text = "\n".join(f"- {term}" for term in (preserve_terms or [])) or "- No shared preserve terms configured"
        structure_text = structure_guidance.strip() or "- No additional structural guidance supplied"
        preserve_section = PromptTemplates.preserve_verbatim_section(
            preserve_verbatim_paragraphs
        )

        positions_text = ""
        for agent_id, position in agent_positions.items():
            excerpt = position.strip().replace("\n", " ")
            positions_text += f"\n{agent_id}: {excerpt[:280]}\n"

        return f"""As Chair, you are preparing the final clean text for possible adoption.

## CANDIDATE CLEAN TEXT
{candidate_text}
{preserve_section}

## SCENARIO CONTEXT
{scenario_context}

## PRE-PLENARY ACCEPTABILITY MAP
LIKELY ACCEPT: {likely_accept}
CONDITIONAL ACCEPT / RESERVATIONS: {conditional_accept}
LIKELY OBJECT: {likely_object}
UNCERTAIN: {uncertain}

## KEY SIGNALS FROM PARTIES
{signals_text}

## BLOCKER THEMES FROM OBJECTORS AND CONDITIONAL ACCEPTORS
{blocker_themes_text}

## RESERVATION / OBJECTOR THEMES
{blocker_agents_text}

## PARAGRAPH-SPECIFIC BLOCKERS
{paragraph_text}

## OVERLOADED PARAGRAPHS TO SIMPLIFY
{overloaded_text}

## DRAFTING ISSUES TO FIX
{drafting_text}

## SHARED TERMS TO PRESERVE WHEN RELEVANT
{preserve_text}

## STRUCTURAL AND LEGAL-DRAFTING ANCHORS
{structure_text}

## MOST RECENT PARTY POSITIONS
{positions_text}

## YOUR TASK
1. Fix any drafting defects, sentence fragments, dangling conjunctions, or incomplete clauses.
2. Make the smallest possible edits needed to improve political acceptability.
3. Prioritize bridging the concerns of likely objectors without reopening the whole package.
3a. Treat LIKELY OBJECT as hard veto risk. Treat CONDITIONAL ACCEPT as a negotiable reservation that may be resolved by simplification, relocation, or narrower wording.
4. Prefer neutral, procedural, and non-expansive language.
5. Do not add new obligations, beneficiaries, institutions, timelines, or categories unless required to remove an obvious blocker.
6. Keep the text fully clean: no square brackets, no option lists, no commentary.
7. If you cannot satisfy everyone, produce the narrowest workable compromise rather than an ambitious package.
8. Preserve relational phrases that make the legal or political logic intelligible; do not compress a clause so far that it becomes generic or semantically incomplete.
9. Preserve the existing document architecture where possible: keep uncontested preambular paragraphs, paragraph roles, and explicit procedural hooks unless they create a clear blocker.
10. If blocker themes center on principles, prefer preambular or chapeau acknowledgments before rewriting core operative obligations.
11. If blocker themes center on support, reporting, or timing, prefer procedural hooks, phased work programme language, modalities, submissions, or qualified wording over new hard obligations.
12. If blocker themes center on integrity, governance, or status/subordination, preserve clear relational and institutional anchors that already help supporting Parties remain on board.
13. If status or hierarchy concerns are active, avoid language that implies one approach is merely subordinate to another unless that relationship is explicitly required by the existing text.
14. If one paragraph attracts opposing coalitions for different reasons, do not load every safeguard and every political reassurance into that same sentence. Consider moving technical safeguards into work programme, reporting, modalities, or procedural paragraphs while keeping the operative objective paragraph narrower.
15. If a paragraph-specific blocker is isolated, target the smallest bridge on that paragraph instead of rewriting the entire package.
16. If a paragraph is flagged as overloaded, simplify it before adding new assurances. Separate adoption-floor language from preferred improvements, and move overflow detail into linked procedural paragraphs where possible.
17. If an overloaded paragraph is tagged with recommended_resolution=split, prefer two narrower clauses or a separate linked subparagraph over one omnibus sentence.
18. If an overloaded paragraph is tagged with recommended_resolution=relocate, move the overflow issue into work programme, reporting, governance, timing, or another linked procedural hook instead of expanding the core objective clause.

Format your output as:
REVISED TEXT:
[Full clean decision text only]

ACCEPTANCE MAP:
[One short line on who is still likely to object]

CHAIR'S NOTE:
[One short line on the logic of the final compromise]
"""

    @staticmethod
    def final_plenary_prompt(
        agent_name: str,
        final_text: str,
        scenario_context: str,
        agenda_focus: Optional[str] = None,
        scenario_briefing: Optional[str] = None,
        scenario_guardrails: Optional[str] = None,
    ) -> str:
        """
        Prompt for the final plenary adoption phase.

        FIX for Issue #2: The action keywords now exactly match what the
        AmendmentProcessor recognises: ACCEPT, OPPOSE, PASS.  The earlier
        version used OBJECT and REQUEST AMENDMENT which fell through to
        the default classification and could be mis-scored as acceptance.
        """
        guardrail_section = ""
        if scenario_guardrails:
            guardrail_section = f"""
## YOUR SCENARIO-SPECIFIC ACCEPTANCE CONDITIONS
Before responding, check these conditions. Do not ACCEPT if the text violates a blocking condition.
{scenario_guardrails}
"""

        focus_section = ""
        if agenda_focus:
            focus_section = f"""
## AGENDA-SPECIFIC FOCUS FROM YOUR GENERAL CHARTER
For this adoption decision, assess the text mainly through:
{agenda_focus}
"""
        briefing_section = ""
        if scenario_briefing:
            briefing_section = f"""
## SCENARIO-SPECIFIC ISSUE BRIEFING
Use this runtime briefing to judge whether the final package protects your key interests on this agenda item:
{scenario_briefing}
"""

        return f"""We are in the FINAL PLENARY. The Chair has presented the following text for adoption.

## TEXT FOR ADOPTION
{final_text}

## CONTEXT
{scenario_context}
{focus_section}
{briefing_section}
{guardrail_section}

## YOUR TASK
Review the text and respond with ONE of the following actions.
The FIRST non-empty line of your response must be exactly one of:
- ACCEPT
- OPPOSE
- PROPOSE MODIFY
- PASS

Additional rules:
- If you accept with reservation, line 1 must still be ACCEPT.
- If you oppose and want to suggest edits, line 1 must still be OPPOSE, and any PROPOSE MODIFY lines must come later.
- Use PROPOSE MODIFY on line 1 only if you are making a procedural amendment request rather than a clear accept or oppose.

Remember: blocking consensus at this stage is a very serious action. Only use OPPOSE if the text violates your fundamental red lines.

Respond briefly (100 words or less).
Do not use Markdown bold or italic formatting.
"""

    @staticmethod
    def red_line_critic_prompt(
        agent_name: str,
        phase: str,
        candidate_response: str,
        negotiation_text: str,
        scenario_context: str,
        scenario_guardrails: str,
    ) -> str:
        """Prompt for a lightweight red-line audit before an agent speaks."""
        return f"""You are auditing a draft intervention for {agent_name} before it is spoken in {phase}.

## TEXT UNDER CONSIDERATION
{negotiation_text}

## SCENARIO CONTEXT
{scenario_context}

## {agent_name} SCENARIO-SPECIFIC GUARDRAILS
{scenario_guardrails}

## DRAFT INTERVENTION
{candidate_response}

## TASK
Check whether the draft intervention accepts, supports, or softens on text that violates the guardrails.
If it is safe, return PASS.
If it violates a guardrail, return FAIL and rewrite the intervention so it preserves the same diplomatic tone but starts with OPPOSE or PROPOSE MODIFY as appropriate.

Output exactly:
VERDICT: PASS or FAIL
VIOLATED_CONDITION: condition text or None
REVISED_RESPONSE:
[revised response, or repeat the original response if PASS]
"""

    @staticmethod
    def stance_reminder_prompt(agent_config: Dict[str, Any], issue: str) -> str:
        """Generate a stance reminder for periodic reinforcement."""
        stances = agent_config.get("stance", {})
        if issue in stances:
            s = stances[issue]
            red_lines = s.get("red_lines", [])
            position = s.get("position", "")
            priority = s.get("priority", "medium")
            reminder = f"On {issue.upper()} (priority: {priority}):\n"
            reminder += f"Your position: {position}\n"
            if red_lines:
                reminder += f"RED LINES (NEVER concede): {'; '.join(red_lines)}\n"
            return reminder
        return ""

    @staticmethod
    def stance_consistency_check_prompt(
        agent_name: str,
        original_stance: str,
        recent_statements: List[str],
    ) -> str:
        """Prompt to check if an agent has drifted from their stance."""
        statements = "\n".join(
            f"Round {i+1}: {s}" for i, s in enumerate(recent_statements)
        )
        return f"""Analyze whether {agent_name}'s recent statements are consistent with their stated position.

## ORIGINAL STANCE
{original_stance}

## RECENT STATEMENTS
{statements}

## TASK
Rate stance consistency from 0.0 (completely inconsistent) to 1.0 (perfectly consistent).
Identify any specific deviations.

Output format:
CONSISTENCY_SCORE: [0.0-1.0]
DEVIATIONS: [list any deviations found, or "None"]
RECOMMENDATION: [any corrective action needed]
"""
