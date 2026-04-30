"""
Evaluator - Multi-dimensional evaluation of simulation results
against real negotiation outcomes.
"""

import re
import logging
from typing import Dict, Any, List, Optional

from src.engine.amendment_processor import AmendmentProcessor
from src.evaluation.llm_judge import LLMStanceJudge
from src.llm.llm_backend import LLMBackend

logger = logging.getLogger(__name__)


class NegotiationEvaluator:
    """
    Evaluates simulation results against reference texts and
    predefined criteria.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        llm_backend: Optional[LLMBackend] = None,
    ):
        self.config = config
        self.llm_backend = llm_backend
        self.metrics = config.get("evaluation", {}).get(
            "metrics",
            ["rouge_l", "key_clause_match", "bracket_resolution_rate"],
        )

    def evaluate(
        self,
        simulated_text: str,
        reference_text: str,
        interaction_log: List[Dict[str, Any]],
        agent_configs: Dict[str, Any],
        key_clauses: Optional[List[Dict[str, Any]]] = None,
        scenario: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run full evaluation suite."""
        results = {}
        scenario_config = scenario or self.config.get("scenario", {})

        if "rouge_l" in self.metrics:
            results["rouge_l"] = self._compute_rouge(simulated_text, reference_text)

        if "bertscore" in self.metrics:
            results["bertscore"] = self._compute_bertscore(
                simulated_text, reference_text
            )

        if "key_clause_match" in self.metrics and key_clauses:
            results["key_clause_match"] = self._check_key_clauses(
                simulated_text, key_clauses
            )

        if "structural_similarity" in self.metrics:
            results["structural_similarity"] = self._structural_similarity(
                simulated_text, reference_text
            )

        if "bracket_resolution_rate" in self.metrics:
            results["bracket_resolution_rate"] = self._bracket_resolution(
                simulated_text
            )

        results["stance_consistency"] = self._evaluate_stance_consistency(
            interaction_log, agent_configs, scenario_config
        )

        results["process_realism"] = self._evaluate_process_realism(
            interaction_log
        )

        results["acceptability"] = self._evaluate_acceptability(
            interaction_log
        )
        results["political_outcome"] = self._extract_political_outcome(
            results["acceptability"]
        )

        results["summary_scores"] = self._compute_summary_scores(results)
        results["legacy_overall_score"] = self._compute_legacy_overall_score(results)
        results["overall_score"] = self._compute_overall_score(results)

        return results

    def _compute_rouge(
        self, simulated: str, reference: str
    ) -> Dict[str, float]:
        """Compute ROUGE scores."""
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            scores = scorer.score(reference, simulated)

            return {
                "rouge1_f": scores["rouge1"].fmeasure,
                "rouge2_f": scores["rouge2"].fmeasure,
                "rougeL_f": scores["rougeL"].fmeasure,
                "rouge1_precision": scores["rouge1"].precision,
                "rouge1_recall": scores["rouge1"].recall,
            }
        except ImportError:
            logger.warning("rouge-score not installed. Skipping ROUGE.")
            return {"error": "rouge-score not installed"}

    def _compute_bertscore(
        self, simulated: str, reference: str
    ) -> Dict[str, float]:
        """Compute BERTScore."""
        try:
            from bert_score import score as bert_score

            P, R, F1 = bert_score(
                [simulated], [reference], lang="en", verbose=False
            )
            return {
                "precision": P.item(),
                "recall": R.item(),
                "f1": F1.item(),
            }
        except ImportError:
            logger.warning("bert-score not installed. Skipping BERTScore.")
            return {"error": "bert-score not installed"}
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            return {"error": str(e)}

    def _check_key_clauses(
        self,
        text: str,
        key_clauses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check whether key clauses/concepts appear in the simulated text.

        Matching is bracket-aware:
        - `found` means present in agreed, non-bracketed text
        - `found_anywhere` also counts unresolved bracketed options
        """
        agreed_text = self._strip_bracketed_content(text)
        results = {
            "total_clauses": len(key_clauses),
            "matched": 0,
            "matched_anywhere": 0,
            "bracketed_only_count": 0,
            "details": [],
        }

        for clause_def in key_clauses:
            clause = clause_def.get("clause", "")
            expected = clause_def.get("expected", True)
            agreed_match = self._match_clause_definition(agreed_text, clause_def)
            any_match = self._match_clause_definition(text, clause_def)
            found = agreed_match["found"]
            found_anywhere = any_match["found"]
            found_only_in_brackets = found_anywhere and not found

            correct = found == expected
            if correct:
                results["matched"] += 1
            if found_anywhere == expected:
                results["matched_anywhere"] += 1
            if found_only_in_brackets:
                results["bracketed_only_count"] += 1

            results["details"].append(
                {
                    "clause": clause,
                    "expected": expected,
                    "found": found,
                    "found_anywhere": found_anywhere,
                    "found_only_in_brackets": found_only_in_brackets,
                    "bracket_status": (
                        "agreed"
                        if found
                        else "bracketed_only"
                        if found_only_in_brackets
                        else "absent"
                    ),
                    "correct": correct,
                    "matched_by": agreed_match.get("matched_by"),
                    "evidence": agreed_match.get("evidence"),
                    "matched_anywhere_by": any_match.get("matched_by"),
                    "evidence_anywhere": any_match.get("evidence"),
                }
            )

        results["accuracy"] = (
            results["matched"] / results["total_clauses"]
            if results["total_clauses"] > 0
            else 0.0
        )
        results["mention_accuracy"] = (
            results["matched_anywhere"] / results["total_clauses"]
            if results["total_clauses"] > 0
            else 0.0
        )

        return results

    @staticmethod
    def _strip_bracketed_content(text: str) -> str:
        """Remove unresolved bracketed segments while preserving agreed text."""
        parts: List[str] = []
        depth = 0
        for char in text:
            if char == "[":
                depth += 1
                continue
            if char == "]":
                depth = max(0, depth - 1)
                continue
            if depth == 0:
                parts.append(char)

        stripped = "".join(parts)
        stripped = re.sub(r"[ \t]+", " ", stripped)
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        return stripped.strip()

    @staticmethod
    def _extract_numeric_references(text: str) -> List[str]:
        """
        Extract normalized article/paragraph references from text.

        UNFCCC decisions often express a decimal-style reference like `6.2`
        as `Article 6, paragraph 2` or `Article 6, paragraphs 2 and 4`.
        This normalizes those forms into `6.2` / `6.4` while also preserving
        already-decimal references found in the text.
        """
        refs: List[str] = []

        for match in re.finditer(
            (
                r"\barticle\s+(\d+)\s*,?\s+paragraphs?\s+"
                r"((?:\d+\s*(?:,\s*and\s*|,\s*|and\s+)?)+)"
            ),
            text,
            re.IGNORECASE,
        ):
            article_num = match.group(1)
            paragraph_block = match.group(2)
            for paragraph_num in re.findall(r"\d+", paragraph_block):
                refs.append(f"{article_num}.{paragraph_num}")

        refs.extend(re.findall(r"\b\d+(?:\.\d+)+\b", text))
        return list(dict.fromkeys(refs))

    def _match_clause_definition(
        self,
        text: str,
        clause_def: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Match a clause definition against text using configurable signals.

        Supported optional fields on each clause definition:
        - aliases: alternate phrases that can satisfy the clause
        - patterns: regex patterns searched case-insensitively
        - required_terms: specific substrings that must appear
        - min_required_terms: threshold for required_terms (default: len(required_terms))
        - keyword_threshold: fallback keyword ratio threshold (default: 0.6)
        """
        text_lower = text.lower()
        clause = clause_def.get("clause", "")
        aliases = clause_def.get("aliases", []) or []
        patterns = clause_def.get("patterns", []) or []
        required_terms = clause_def.get("required_terms", []) or []
        keyword_threshold = clause_def.get("keyword_threshold", 0.6)

        candidates = [clause] + [alias for alias in aliases if alias]
        numeric_refs = self._extract_numeric_references(" ".join(candidates))
        text_numeric_refs = set(self._extract_numeric_references(text))

        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return {
                    "found": True,
                    "matched_by": "pattern",
                    "evidence": pattern,
                }

        if required_terms:
            term_hits = [
                term for term in required_terms if term.lower() in text_lower
            ]
            min_required = clause_def.get(
                "min_required_terms",
                len(required_terms),
            )
            if len(term_hits) >= min_required:
                return {
                    "found": True,
                    "matched_by": "required_terms",
                    "evidence": term_hits,
                }
            return {
                "found": False,
                "matched_by": "required_terms",
                "evidence": term_hits,
            }

        for candidate in candidates:
            candidate_lower = candidate.lower()
            if candidate_lower and candidate_lower in text_lower:
                return {
                    "found": True,
                    "matched_by": "exact_phrase",
                    "evidence": candidate,
                }

        if numeric_refs:
            numeric_hits = [ref for ref in numeric_refs if ref in text_numeric_refs]
            min_numeric_refs = clause_def.get(
                "min_numeric_refs",
                len(numeric_refs),
            )
            if len(numeric_hits) < min_numeric_refs:
                return {
                    "found": False,
                    "matched_by": "numeric_refs",
                    "evidence": numeric_hits,
                }

        best_candidate = ""
        best_hits: List[str] = []
        best_ratio = 0.0

        for candidate in candidates:
            keywords = self._meaningful_keywords(candidate.lower())
            if not keywords:
                continue

            hits = [kw for kw in keywords if kw in text_lower]
            ratio = len(hits) / len(keywords)
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate = candidate
                best_hits = hits

        if best_candidate and best_ratio >= keyword_threshold:
            return {
                "found": True,
                "matched_by": "keyword_overlap",
                "evidence": {
                    "candidate": best_candidate,
                    "hits": best_hits,
                    "ratio": round(best_ratio, 3),
                },
            }

        return {
            "found": False,
            "matched_by": "keyword_overlap",
            "evidence": {
                "candidate": best_candidate or clause,
                "hits": best_hits,
                "ratio": round(best_ratio, 3),
            },
        }

    def _structural_similarity(
        self, simulated: str, reference: str
    ) -> Dict[str, Any]:
        """Compare the structural similarity of two texts."""

        def extract_structure(text):
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            numbered = re.findall(r"^\d+\.", text, re.MULTILINE)
            lettered = re.findall(r"^\([a-z]\)", text, re.MULTILINE)
            return {
                "num_paragraphs": len(paragraphs),
                "num_numbered_items": len(numbered),
                "num_lettered_items": len(lettered),
                "total_words": len(text.split()),
            }

        sim_struct = extract_structure(simulated)
        ref_struct = extract_structure(reference)

        similarities = {}
        for key in sim_struct:
            sim_val = sim_struct[key]
            ref_val = ref_struct[key]
            if ref_val == 0 and sim_val == 0:
                similarities[key] = 1.0
            elif ref_val == 0 or sim_val == 0:
                similarities[key] = 0.0
            else:
                ratio = min(sim_val, ref_val) / max(sim_val, ref_val)
                similarities[key] = ratio

        avg_similarity = (
            sum(similarities.values()) / len(similarities)
            if similarities
            else 0.0
        )

        return {
            "simulated_structure": sim_struct,
            "reference_structure": ref_struct,
            "component_similarities": similarities,
            "average_similarity": avg_similarity,
        }

    def _bracket_resolution(self, text: str) -> Dict[str, Any]:
        """
        Measure how many brackets remain in the final text.

        FIX for previous review: returns a proportional resolution rate
        based on paragraph counts, not a binary 0/1.
        """
        open_brackets = text.count("[")
        close_brackets = text.count("]")
        unmatched_brackets = open_brackets != close_brackets
        bracket_pairs = min(open_brackets, close_brackets)

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        total_paragraphs = max(len(paragraphs), 1)

        bracketed_paragraphs = sum(1 for p in paragraphs if "[" in p or "]" in p)

        clean_paragraphs = total_paragraphs - bracketed_paragraphs
        resolution_rate = clean_paragraphs / total_paragraphs

        if open_brackets == 0 and close_brackets == 0:
            note = "Fully resolved – no brackets remain"
        elif unmatched_brackets:
            note = (
                f"Unmatched brackets remain ({open_brackets} '[' vs "
                f"{close_brackets} ']') in "
                f"{bracketed_paragraphs}/{total_paragraphs} paragraphs"
            )
        else:
            note = (
                f"{bracket_pairs} bracket pair(s) remaining in "
                f"{bracketed_paragraphs}/{total_paragraphs} paragraphs"
            )

        return {
            "remaining_bracket_pairs": bracket_pairs,
            "remaining_open_brackets": open_brackets,
            "total_paragraphs": total_paragraphs,
            "bracketed_paragraphs": bracketed_paragraphs,
            "resolution_rate": round(resolution_rate, 4),
            "note": note,
        }

    def _evaluate_stance_consistency(
        self,
        interaction_log: List[Dict[str, Any]],
        agent_configs: Dict[str, Any],
        scenario: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate whether each agent maintained consistent positions.

        The evaluator first looks for scenario-specific blocking conditions.
        If none are provided, it falls back to global agent red lines, filtered
        by scenario salient issues when available. This avoids penalizing an
        Article 6.8 negotiation for unrelated red lines such as mitigation
        peaking language.
        """
        scenario = scenario or {}
        llm_judge_enabled = self.config.get("evaluation", {}).get(
            "llm_judge_enabled",
            False,
        )
        llm_judge = None
        if llm_judge_enabled:
            if self.llm_backend is None:
                logger.warning(
                    "evaluation.llm_judge_enabled is true but no llm_backend "
                    "was provided. Skipping LLM stance-judge pass."
                )
            else:
                llm_judge = LLMStanceJudge(
                    self.llm_backend,
                    self.config.get("evaluation", {}),
                )

        agent_statements: Dict[str, List[Dict[str, Any]]] = {}
        for entry in interaction_log:
            agent = entry.get("agent", "")
            content = entry.get("content", "")
            if agent and agent not in ("CHAIR", "CHAIR_ESCALATION", ""):
                if agent not in agent_statements:
                    agent_statements[agent] = []
                agent_statements[agent].append(
                    {
                        "content": content,
                        "phase": entry.get("phase", ""),
                        "round": entry.get("round"),
                    }
                )

        results = {}
        for agent_id, statements in agent_statements.items():
            if agent_id not in agent_configs:
                continue

            config = agent_configs[agent_id]
            red_line_details = self._get_relevant_stance_conditions(
                agent_id=agent_id,
                agent_config=config,
                scenario=scenario,
            )
            red_lines = [r["red_line"] for r in red_line_details]

            violations = []

            # Concession/acceptance language
            concession_phrases = [
                "we accept", "we agree to", "we concede", "we can accept",
                "we are willing to accept", "we will accept",
                "we withdraw our objection", "we no longer insist",
            ]

            # Softening/flexibility language
            softening_phrases = [
                "we are flexible on", "we could consider",
                "we are open to", "we can be flexible",
                "we are prepared to compromise on",
                "we will not block",
            ]

            for stmt_idx, statement_info in enumerate(statements):
                statement = statement_info.get("content", "")
                statement_lower = statement.lower()
                phase_name = statement_info.get("phase", "")
                skip_acceptance_checks = self._is_tentative_acceptance(
                    statement_lower,
                    phase_name,
                )

                for rl_info in red_line_details:
                    rl = rl_info["red_line"]
                    source = rl_info.get("source", "agent_red_line")
                    rl_lower = rl.lower()

                    rl_keywords = self._meaningful_keywords(rl_lower)
                    if not rl_keywords:
                        continue

                    # Check if enough red-line keywords appear in the statement
                    keyword_hits = sum(
                        1 for kw in rl_keywords if kw in statement_lower
                    )
                    keyword_ratio = keyword_hits / len(rl_keywords)

                    threshold = 0.55 if source == "scenario_blocking_condition" else 0.4
                    if keyword_ratio < threshold:
                        # Red-line topic not being discussed
                        continue

                    if source == "scenario_blocking_condition" and self._is_reassuring_statement(
                        statement_lower, rl_lower
                    ):
                        continue

                    # Check 1: Direct concession near red-line topic
                    if not skip_acceptance_checks:
                        for phrase in concession_phrases:
                            if phrase in statement_lower:
                                violations.append({
                                    "type": "direct_concession",
                                    "red_line": rl,
                                    "issue": rl_info["issue"],
                                    "source": source,
                                    "trigger_phrase": phrase,
                                    "statement_index": stmt_idx,
                                    "excerpt": statement[:200],
                                    "severity": "high",
                                })
                                break

                    # Check 2: Softening language near red-line topic
                    if not skip_acceptance_checks:
                        for phrase in softening_phrases:
                            if phrase in statement_lower:
                                violations.append({
                                    "type": "softening",
                                    "red_line": rl,
                                    "issue": rl_info["issue"],
                                    "source": source,
                                    "trigger_phrase": phrase,
                                    "statement_index": stmt_idx,
                                    "excerpt": statement[:200],
                                    "severity": "medium",
                                })
                                break

                    # Check 3: Contradiction detection
                    # If red line says "No X" or "Must not X", check if
                    # agent says "support X" or "we propose X"
                    negation_match = re.search(
                        r'(?:no|not|never|must not|shall not)\s+(.+)',
                        rl_lower,
                    )
                    if negation_match:
                        forbidden_concept = negation_match.group(1).strip()
                        forbidden_keywords = self._meaningful_keywords(
                            forbidden_concept
                        )
                        if forbidden_keywords:
                            support_phrases = [
                                "we support", "we propose", "we welcome",
                                "we endorse", "we call for",
                            ]
                            for sp in support_phrases:
                                if sp in statement_lower:
                                    fk_hits = sum(
                                        1 for fk in forbidden_keywords
                                        if fk in statement_lower
                                    )
                                    if fk_hits >= len(forbidden_keywords) * 0.5:
                                        violations.append({
                                            "type": "contradiction",
                                            "red_line": rl,
                                            "issue": rl_info["issue"],
                                            "source": source,
                                            "trigger_phrase": sp,
                                            "statement_index": stmt_idx,
                                            "excerpt": statement[:200],
                                            "severity": "high",
                                        })
                                        break

            # Deduplicate violations by (statement_index, red_line)
            seen = set()
            unique_violations = []
            for v in violations:
                key = (v["statement_index"], v["red_line"])
                if key not in seen:
                    seen.add(key)
                    unique_violations.append(v)

            total_statements = max(len(statements), 1)
            heuristic_score = self._score_stance_violations(
                unique_violations,
                total_statements,
            )

            results[agent_id] = {
                "total_statements": len(statements),
                "red_lines_count": len(red_lines),
                "condition_source": (
                    red_line_details[0].get("source", "none")
                    if red_line_details else "none"
                ),
                "total_violations": len(unique_violations),
                "high_severity": heuristic_score["high_count"],
                "medium_severity": heuristic_score["medium_count"],
                "violations": unique_violations[:10],
                "consistency_score": heuristic_score["consistency_score"],
                "note": (
                    "Scenario-scoped heuristic evaluation. Treat as a flag "
                    "for review, not a definitive legal assessment."
                ),
            }
            if llm_judge is not None:
                llm_confirmed_violations = self._verify_stance_violations_with_llm(
                    llm_judge=llm_judge,
                    agent_id=agent_id,
                    agent_display_name=config.get("display_name", agent_id),
                    heuristic_violations=unique_violations,
                )
                llm_verified_score = self._score_stance_violations(
                    llm_confirmed_violations,
                    total_statements,
                )
                results[agent_id]["heuristic_violations"] = unique_violations
                results[agent_id]["llm_confirmed_violations"] = llm_confirmed_violations
                results[agent_id]["consistency_score_llm_verified"] = (
                    llm_verified_score["consistency_score"]
                )

        return results

    @staticmethod
    def _score_stance_violations(
        violations: List[Dict[str, Any]],
        total_statements: int,
    ) -> Dict[str, float]:
        """Score stance consistency using the historical heuristic formula."""
        high_count = sum(1 for v in violations if v["severity"] == "high")
        medium_count = sum(1 for v in violations if v["severity"] == "medium")
        weighted_violations = high_count * 1.0 + medium_count * 0.5
        consistency = max(
            0.0,
            1.0 - (weighted_violations / max(total_statements, 1)) * 3,
        )
        return {
            "high_count": high_count,
            "medium_count": medium_count,
            "consistency_score": round(consistency, 3),
        }

    @staticmethod
    def _verify_stance_violations_with_llm(
        llm_judge: LLMStanceJudge,
        agent_id: str,
        agent_display_name: str,
        heuristic_violations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter heuristic violations through the optional LLM judge."""
        llm_confirmed_violations = []
        for violation in heuristic_violations:
            try:
                judge_result = llm_judge.verify_violation(
                    agent_id=agent_id,
                    agent_display_name=agent_display_name,
                    red_line=violation["red_line"],
                    statement_excerpt=violation["excerpt"],
                    violation_type=violation["type"],
                )
            except Exception as exc:
                logger.warning(
                    "LLM stance judge failed for %s on red line %r: %s",
                    agent_id,
                    violation.get("red_line", ""),
                    exc,
                )
                continue

            if judge_result.get("confirmed"):
                llm_confirmed_violations.append(violation)

        return llm_confirmed_violations

    def _get_relevant_stance_conditions(
        self,
        agent_id: str,
        agent_config: Dict[str, Any],
        scenario: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Return scenario-scoped conditions for an agent."""
        constraints = scenario.get("scenario_constraints", {})
        agent_conditions = (
            constraints.get("agent_blocking_conditions", {}).get(agent_id, [])
        )
        if agent_conditions:
            return [
                {
                    "issue": "scenario_blocking_condition",
                    "red_line": condition,
                    "source": "scenario_blocking_condition",
                }
                for condition in agent_conditions
            ]

        salient_issues = set(constraints.get("salient_issues", []))
        stances = agent_config.get("stance", {})
        details: List[Dict[str, str]] = []

        for issue, stance_details in stances.items():
            if salient_issues and issue not in salient_issues:
                continue
            if not isinstance(stance_details, dict):
                continue
            for red_line in stance_details.get("red_lines", []):
                details.append(
                    {
                        "issue": issue,
                        "red_line": red_line,
                        "source": "agent_red_line",
                    }
                )

        return details

    @staticmethod
    def _is_tentative_acceptance(statement: str, phase: str) -> bool:
        """
        Ignore procedural or conditional acceptance language in non-final phases.

        Negotiators often "accept" a bracketed paragraph for drafting purposes
        while still preserving a preference or condition. That is not equivalent
        to abandoning a substantive red line.
        """
        if phase == "final_plenary":
            return False

        bracket_markers = [
            "bracketed option",
            "bracketed options",
            "including the bracketed options",
            "including both options",
            "[option",
            "option 1",
            "option 2",
            "option 3",
            "current text of paragraph",
            "we note our preference",
            "we note our strong preference",
        ]
        conditional_markers = [
            "provided that",
            "provided the text",
            "if the text includes",
            "if the text retains",
            "subject to",
            "on the understanding that",
            "so long as",
            "with the inclusion of",
            "we can accept adding",
            "we can accept if",
            "we can accept provided",
        ]

        has_acceptance = "accept" in statement or "open to" in statement
        if not has_acceptance:
            return False

        return any(marker in statement for marker in bracket_markers + conditional_markers)

    @staticmethod
    def _meaningful_keywords(text: str) -> List[str]:
        """Extract non-trivial keywords for heuristic stance checks."""
        stop_words = {
            "must", "will", "shall", "should", "the", "and", "for", "not",
            "with", "that", "this", "from", "have", "been", "are", "was",
            "were", "any", "new", "all", "our", "their", "its", "also",
            "final", "text", "omits", "omitted", "allows", "allow",
            "where", "issue", "issues", "party", "parties", "approach",
            "approaches", "condition", "conditions", "discussed",
        }
        return [
            word for word in re.findall(r"\w+", text.lower())
            if len(word) > 3 and word not in stop_words
        ]

    @staticmethod
    def _is_reassuring_statement(statement: str, condition: str) -> bool:
        """
        Avoid false positives when a statement affirms a safeguard that the
        blocking condition says must not be lost.
        """
        positive_markers = [
            "does not",
            "not",
            "preserve",
            "preserves",
            "preserving",
            "protect",
            "protects",
            "retain",
            "retains",
            "maintain",
            "maintains",
            "include",
            "includes",
            "including",
            "ensure",
            "ensures",
            "uphold",
            "upholds",
            "recognize",
            "recognizes",
            "reflect",
            "reflects",
            "address",
            "addresses",
            "complement",
            "complements",
            "equal status",
        ]
        negative_markers = [
            "without",
            "no",
            "omit",
            "omits",
            "omitted",
            "lack",
            "lacks",
            "lacking",
            "exclude",
            "excludes",
            "excluded",
            "remove",
            "removes",
            "removed",
        ]

        condition_keywords = NegotiationEvaluator._meaningful_keywords(condition)
        if not condition_keywords:
            return False

        salient_hits = [
            keyword for keyword in condition_keywords
            if keyword in statement
        ]
        if not salient_hits:
            return False

        if not any(marker in statement for marker in positive_markers):
            return False

        for keyword in salient_hits:
            negative_patterns = [
                f"{marker} {keyword}" for marker in negative_markers
            ]
            if any(pattern in statement for pattern in negative_patterns):
                return False

        return True

    def _evaluate_process_realism(
        self, interaction_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate the realism of the negotiation process."""
        total_entries = len(interaction_log)
        agent_entries = [
            e
            for e in interaction_log
            if e.get("agent", "") not in ("CHAIR", "CHAIR_ESCALATION", "")
        ]

        diplomatic_markers = [
            "on behalf of",
            "we propose",
            "we support",
            "we cannot accept",
            "we call for",
            "in accordance with",
            "we associate",
            "we welcome",
            "we urge",
            "respectfully",
            "mr. chair",
            "madam chair",
            "distinguished",
        ]

        diplomatic_count = 0
        for entry in agent_entries:
            content_lower = entry.get("content", "").lower()
            if any(marker in content_lower for marker in diplomatic_markers):
                diplomatic_count += 1

        diplomatic_rate = (
            diplomatic_count / len(agent_entries) if agent_entries else 0.0
        )

        support_count = 0
        for entry in agent_entries:
            content_lower = entry.get("content", "").lower()
            if any(
                phrase in content_lower
                for phrase in [
                    "we support",
                    "we associate",
                    "we align",
                    "we echo",
                ]
            ):
                support_count += 1

        coalition_rate = (
            support_count / len(agent_entries) if agent_entries else 0.0
        )

        phases_present = set(e.get("phase", "") for e in interaction_log)

        return {
            "total_interactions": total_entries,
            "agent_interactions": len(agent_entries),
            "diplomatic_language_rate": round(diplomatic_rate, 3),
            "coalition_behavior_rate": round(coalition_rate, 3),
            "phases_covered": list(phases_present),
            "realism_score": round(
                diplomatic_rate * 0.5 + coalition_rate * 0.3 + 0.2, 3
            ),
        }

    def _evaluate_acceptability(
        self,
        interaction_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Estimate whether the final package was politically passable."""
        plenary_entries = [
            entry for entry in interaction_log
            if entry.get("phase") == "final_plenary"
            and entry.get("agent", "") not in ("CHAIR", "CHAIR_ESCALATION", "")
        ]

        counts = {
            "accept": 0,
            "oppose": 0,
            "modify": 0,
            "pass": 0,
            "ambiguous": 0,
        }
        accepting_agents: List[str] = []
        blocking_agents: List[str] = []

        for entry in plenary_entries:
            action = self._infer_plenary_action(entry)
            if action in counts:
                counts[action] += 1
            else:
                counts["ambiguous"] += 1

            if action == "accept":
                accepting_agents.append(entry.get("agent", ""))
            elif action in ("oppose", "modify"):
                blocking_agents.append(entry.get("agent", ""))

        total = max(len(plenary_entries), 1)
        support_units = counts["accept"]
        blocking_units = counts["oppose"] + counts["modify"]
        acceptability_score = max(
            0.0,
            min(1.0, (support_units / total) - (blocking_units / total)),
        )

        return {
            "total_parties": len(plenary_entries),
            "accept_count": counts["accept"],
            "oppose_count": counts["oppose"],
            "modify_count": counts["modify"],
            "pass_count": counts["pass"],
            "ambiguous_count": counts["ambiguous"],
            "accepting_agents": accepting_agents,
            "blocking_agents": blocking_agents,
            "consensus_possible": blocking_units == 0 and counts["accept"] > 0,
            "acceptability_score": round(acceptability_score, 3),
        }

    @staticmethod
    def _extract_political_outcome(
        acceptability: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Surface the adoption outcome separately from score-oriented fields."""
        return {
            "adopted": bool(acceptability.get("consensus_possible", False)),
            "accept_count": int(acceptability.get("accept_count", 0)),
            "block_count": int(acceptability.get("oppose_count", 0))
            + int(acceptability.get("modify_count", 0)),
            "accepting_agents": list(acceptability.get("accepting_agents", [])),
            "blocking_agents": list(acceptability.get("blocking_agents", [])),
        }

    @staticmethod
    def _infer_plenary_action(entry: Dict[str, Any]) -> str:
        """Infer a plenary action from a logged entry."""
        action = entry.get("action")
        if action in ("accept", "oppose", "modify", "pass"):
            content = entry.get("content", "")
            content_lower = content.lower()
            if action != "oppose" and any(
                marker in content_lower
                for marker in (
                    "cannot accept",
                    "we oppose",
                    "we object",
                    "cannot support",
                    "cannot join consensus",
                    "cannot agree",
                    "will not accept",
                )
            ):
                return "oppose"
            return str(action).lower()

        content = entry.get("content", "")
        inferred = AmendmentProcessor().get_primary_action(content)
        return inferred if inferred in ("accept", "oppose", "modify", "pass") else "ambiguous"

    def _compute_legacy_overall_score(self, results: Dict[str, Any]) -> float:
        """Compute the historical reference-heavy overall score."""
        scores = []
        weights = []

        if "rouge_l" in results and "rougeL_f" in results["rouge_l"]:
            scores.append(results["rouge_l"]["rougeL_f"])
            weights.append(0.25)

        if "key_clause_match" in results:
            scores.append(results["key_clause_match"].get("accuracy", 0.0))
            weights.append(0.25)

        if "structural_similarity" in results:
            scores.append(
                results["structural_similarity"].get("average_similarity", 0.0)
            )
            weights.append(0.15)

        if "bracket_resolution_rate" in results:
            rate = results["bracket_resolution_rate"].get("resolution_rate", 0.0)
            scores.append(rate)
            weights.append(0.10)

        if "stance_consistency" in results:
            consistency_scores = [
                v.get("consistency_score", 0.0)
                for v in results["stance_consistency"].values()
                if isinstance(v, dict) and "consistency_score" in v
            ]
            if consistency_scores:
                scores.append(sum(consistency_scores) / len(consistency_scores))
                weights.append(0.15)

        if "process_realism" in results:
            scores.append(results["process_realism"].get("realism_score", 0.0))
            weights.append(0.10)

        if not scores:
            return 0.0

        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

    def _compute_overall_score(self, results: Dict[str, Any]) -> float:
        """
        Compute the primary overall score.

        This now blends textual alignment with negotiation quality so that a
        text that looks realistic but cannot pass politically is not overrated.
        """
        summary = results.get("summary_scores", {})
        if summary:
            reference_alignment = summary.get("reference_alignment_score", 0.0)
            negotiation_quality = summary.get("negotiation_quality_score", 0.0)
            return round(reference_alignment * 0.45 + negotiation_quality * 0.55, 4)
        return self._compute_legacy_overall_score(results)

    def _compute_summary_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Provide split summary scores for text alignment and negotiation quality."""
        reference_scores = []
        reference_weights = []
        process_scores = []
        process_weights = []

        if "rouge_l" in results and "rougeL_f" in results["rouge_l"]:
            reference_scores.append(results["rouge_l"]["rougeL_f"])
            reference_weights.append(0.35)
        if "bertscore" in results and "f1" in results["bertscore"]:
            reference_scores.append(results["bertscore"]["f1"])
            reference_weights.append(0.15)
        if "key_clause_match" in results:
            reference_scores.append(results["key_clause_match"].get("accuracy", 0.0))
            reference_weights.append(0.25)
        if "structural_similarity" in results:
            reference_scores.append(
                results["structural_similarity"].get("average_similarity", 0.0)
            )
            reference_weights.append(0.25)

        if "bracket_resolution_rate" in results:
            process_scores.append(
                results["bracket_resolution_rate"].get("resolution_rate", 0.0)
            )
            process_weights.append(0.25)
        if "stance_consistency" in results:
            stance_values = [
                value.get("consistency_score", 0.0)
                for value in results["stance_consistency"].values()
                if isinstance(value, dict)
            ]
            if stance_values:
                process_scores.append(sum(stance_values) / len(stance_values))
                process_weights.append(0.25)
        if "process_realism" in results:
            process_scores.append(results["process_realism"].get("realism_score", 0.0))
            process_weights.append(0.20)
        if "acceptability" in results:
            process_scores.append(results["acceptability"].get("acceptability_score", 0.0))
            process_weights.append(0.30)

        reference_alignment = (
            sum(score * weight for score, weight in zip(reference_scores, reference_weights))
            / sum(reference_weights)
            if reference_weights else 0.0
        )
        negotiation_quality = (
            sum(score * weight for score, weight in zip(process_scores, process_weights))
            / sum(process_weights)
            if process_weights else 0.0
        )

        return {
            "reference_alignment_score": round(reference_alignment, 4),
            "text_quality_score": round(reference_alignment, 4),
            "negotiation_quality_score": round(negotiation_quality, 4),
            "overall_score": round(reference_alignment * 0.45 + negotiation_quality * 0.55, 4),
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("NEGOTIATION SIMULATION EVALUATION REPORT")
        report.append("=" * 60)

        overall = results.get("overall_score", 0.0)
        report.append(f"\n📊 OVERALL SCORE: {overall:.3f} / 1.000")
        text_quality = results.get("summary_scores", {}).get("text_quality_score", 0.0)
        report.append(f"Text quality score: {text_quality:.3f}")
        report.append("-" * 40)

        if "summary_scores" in results:
            summary = results["summary_scores"]
            report.append(
                f"   Reference alignment: {summary.get('reference_alignment_score', 0):.3f}"
            )
            report.append(
                f"   Negotiation quality: {summary.get('negotiation_quality_score', 0):.3f}"
            )

        if "rouge_l" in results and isinstance(results["rouge_l"], dict):
            rouge = results["rouge_l"]
            report.append("\n📝 TEXT SIMILARITY (ROUGE)")
            if "error" not in rouge:
                report.append(f"   ROUGE-1 F1: {rouge.get('rouge1_f', 0):.3f}")
                report.append(f"   ROUGE-2 F1: {rouge.get('rouge2_f', 0):.3f}")
                report.append(f"   ROUGE-L F1: {rouge.get('rougeL_f', 0):.3f}")
            else:
                report.append(f"   Error: {rouge['error']}")

        if "bertscore" in results and isinstance(results["bertscore"], dict):
            bert = results["bertscore"]
            report.append("\n📝 TEXT SIMILARITY (BERTScore)")
            if "error" not in bert:
                report.append(f"   Precision: {bert.get('precision', 0):.3f}")
                report.append(f"   Recall:    {bert.get('recall', 0):.3f}")
                report.append(f"   F1:        {bert.get('f1', 0):.3f}")
            else:
                report.append(f"   Error: {bert['error']}")

        if "key_clause_match" in results:
            kc = results["key_clause_match"]
            report.append("\n🔍 KEY CLAUSE MATCHING")
            report.append(
                f"   Agreed-text accuracy: {kc.get('accuracy', 0):.1%} "
                f"({kc.get('matched', 0)}/{kc.get('total_clauses', 0)})"
            )
            report.append(
                f"   Mentioned anywhere: {kc.get('mention_accuracy', 0):.1%} "
                f"({kc.get('matched_anywhere', 0)}/{kc.get('total_clauses', 0)})"
            )
            if kc.get("bracketed_only_count", 0):
                report.append(
                    f"   Present only in brackets: {kc.get('bracketed_only_count', 0)}"
                )
            for detail in kc.get("details", []):
                if detail.get("found_only_in_brackets"):
                    status = "⚠️"
                    found_str = "present only in bracketed text"
                else:
                    status = "✅" if detail["correct"] else "❌"
                    found_str = "found" if detail["found"] else "not found"
                report.append(
                    f"   {status} \"{detail['clause']}\" - {found_str} "
                    f"(expected: {'present' if detail['expected'] else 'absent'})"
                )

        if "structural_similarity" in results:
            ss = results["structural_similarity"]
            report.append("\n🏗️ STRUCTURAL SIMILARITY")
            report.append(
                f"   Average: {ss.get('average_similarity', 0):.3f}"
            )

        if "bracket_resolution_rate" in results:
            br = results["bracket_resolution_rate"]
            report.append("\n🔓 BRACKET RESOLUTION")
            report.append(f"   Rate: {br.get('resolution_rate', 0):.1%}")
            report.append(f"   {br.get('note', 'N/A')}")

        if "stance_consistency" in results:
            report.append("\n🎯 STANCE CONSISTENCY (per agent)")
            for agent_id, sc in results["stance_consistency"].items():
                if isinstance(sc, dict) and "consistency_score" in sc:
                    score = sc["consistency_score"]
                    high = sc.get("high_severity", 0)
                    medium = sc.get("medium_severity", 0)
                    report.append(
                        f"   {agent_id}: {score:.2f} "
                        f"(violations: {high} high, {medium} medium)"
                    )

        if "process_realism" in results:
            pr = results["process_realism"]
            report.append("\n🌐 PROCESS REALISM")
            report.append(
                f"   Diplomatic language rate: {pr.get('diplomatic_language_rate', 0):.1%}"
            )
            report.append(
                f"   Coalition behavior rate: {pr.get('coalition_behavior_rate', 0):.1%}"
            )
            report.append(
                f"   Realism score: {pr.get('realism_score', 0):.3f}"
            )

        if "acceptability" in results:
            acceptability = results["acceptability"]
            report.append("\n🤝 POLITICAL ACCEPTABILITY")
            report.append(
                f"   Accept count: {acceptability.get('accept_count', 0)}"
            )
            report.append(
                f"   Blocking count: "
                f"{acceptability.get('oppose_count', 0) + acceptability.get('modify_count', 0)}"
            )
            report.append(
                f"   Acceptability score: {acceptability.get('acceptability_score', 0):.3f}"
            )
            consensus_text = "yes" if acceptability.get("consensus_possible") else "no"
            report.append(f"   Consensus possible: {consensus_text}")

        report.append("\n" + "=" * 60)
        return "\n".join(report)
