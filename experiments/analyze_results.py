#!/usr/bin/env python3
"""
Analyze and visualize simulation results.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(path: str) -> Dict[str, Any]:
    """Load simulation results from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def find_result_files(path: str) -> List[str]:
    """Find result JSON files in a directory tree."""
    result_files: List[str] = []
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith("_results.json"):
                result_files.append(os.path.join(root, filename))
    return sorted(result_files, key=os.path.getmtime)


def analyze_single(results: Dict[str, Any]) -> str:
    """Analyze a single simulation run."""
    report = []
    report.append("=" * 60)
    report.append("SIMULATION ANALYSIS REPORT")
    report.append("=" * 60)

    report.append(f"\nSimulation: {results.get('simulation_name', 'unknown')}")
    report.append(f"Scenario: {results.get('scenario', 'unknown')}")
    report.append(f"Outcome: {results.get('outcome', 'unknown')}")
    report.append(f"Total Rounds: {results.get('total_rounds', 0)}")

    start = results.get("start_time", "")
    end = results.get("end_time", "")
    report.append(f"Start: {start}")
    report.append(f"End: {end}")

    # Agent statistics
    report.append("\n--- AGENT STATISTICS ---")
    agent_stats = results.get("agent_stats", {})
    for agent_id, stats in sorted(agent_stats.items()):
        report.append(
            f"  {agent_id}: "
            f"rounds={stats.get('rounds_participated', 0)}, "
            f"concessions_made={stats.get('concessions_made', 0)}, "
            f"concessions_received={stats.get('concessions_received', 0)}"
        )

    # Interaction analysis
    interaction_log = results.get("interaction_log", [])
    if interaction_log:
        report.append("\n--- INTERACTION ANALYSIS ---")

        phase_counts = Counter(
            e.get("phase", "unknown") for e in interaction_log
        )
        report.append("  Interactions per phase:")
        for phase, count in phase_counts.most_common():
            report.append(f"    {phase}: {count}")

        agent_counts = Counter(
            e.get("agent", "unknown")
            for e in interaction_log
            if e.get("agent") not in ("CHAIR", "CHAIR_ESCALATION", "")
        )
        report.append("  Interactions per agent:")
        for agent, count in agent_counts.most_common():
            report.append(f"    {agent}: {count}")

        amendment_counts = Counter()
        for entry in interaction_log:
            for amd in entry.get("amendments", []):
                amendment_counts[amd.get("action", "unknown")] += 1

        if amendment_counts:
            report.append("  Amendment types:")
            for atype, count in amendment_counts.most_common():
                report.append(f"    {atype}: {count}")

        chair_entries = [
            e
            for e in interaction_log
            if e.get("agent") in ("CHAIR", "CHAIR_ESCALATION")
            and e.get("phase") == "informal_consultations"
        ]
        if chair_entries:
            report.append(f"  Chair interventions: {len(chair_entries)}")

    # Final text analysis
    final_text = results.get("final_text", "")
    if final_text:
        report.append("\n--- FINAL TEXT ANALYSIS ---")
        report.append(f"  Total length: {len(final_text)} characters")
        report.append(f"  Total words: {len(final_text.split())}")
        bracket_count = final_text.count("[")
        report.append(f"  Remaining brackets: {bracket_count}")

        paragraphs = [p for p in final_text.split("\n\n") if p.strip()]
        report.append(f"  Paragraphs: {len(paragraphs)}")

    # LLM stats
    llm_stats = results.get("llm_stats", {})
    if llm_stats:
        report.append("\n--- LLM USAGE ---")
        report.append(f"  Provider: {llm_stats.get('provider', 'unknown')}")
        report.append(f"  Model: {llm_stats.get('model', 'unknown')}")
        report.append(f"  Total requests: {llm_stats.get('total_requests', 0)}")

        # FIX Issue #8: Use separate prompt/completion counts for cost
        prompt_tokens = llm_stats.get("total_prompt_tokens", 0)
        completion_tokens = llm_stats.get("total_completion_tokens", 0)
        total_tokens = llm_stats.get(
            "total_tokens_used", prompt_tokens + completion_tokens
        )

        report.append(f"  Prompt tokens: {prompt_tokens}")
        report.append(f"  Completion tokens: {completion_tokens}")
        report.append(f"  Total tokens: {total_tokens}")

        # Estimate cost using separate input/output rates
        model = llm_stats.get("model", "")
        estimated_cost = _estimate_cost(model, prompt_tokens, completion_tokens)
        if estimated_cost > 0:
            report.append(f"  Estimated cost: ${estimated_cost:.4f}")

    # Objections
    objections = results.get("objections", [])
    if objections:
        report.append("\n--- OBJECTIONS ---")
        for obj in objections:
            report.append(
                f"  {obj.get('agent', 'unknown')}: {obj.get('reason', '')[:200]}"
            )

    report.append("\n" + "=" * 60)
    return "\n".join(report)


def _estimate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """
    Estimate cost using separate input/output token rates.

    FIX for Issue #8: Previously the total token count was multiplied
    by both input AND output rates, resulting in ~2x overestimate.
    Now we use the correct per-token rates for each direction.
    """
    # Pricing per 1K tokens (as of 2024-2025, approximate)
    pricing = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-5-haiku": {"input": 0.00025, "output": 0.00125},
        "deepseek-chat": {"input": 0.00014, "output": 0.00028},
    }

    # Find matching pricing
    model_lower = model.lower()
    rates = None
    for key in sorted(pricing, key=len, reverse=True):
        if key in model_lower:
            rates = pricing[key]
            break

    if rates is None:
        return 0.0

    input_cost = (prompt_tokens / 1000) * rates["input"]
    output_cost = (completion_tokens / 1000) * rates["output"]
    return input_cost + output_cost


def compare_runs(results_list: List[Dict[str, Any]]) -> str:
    """Compare multiple simulation runs."""
    report = []
    report.append("=" * 60)
    report.append("COMPARATIVE ANALYSIS")
    report.append("=" * 60)
    report.append(f"\nComparing {len(results_list)} simulation runs\n")

    report.append("--- OUTCOMES ---")
    for i, results in enumerate(results_list):
        name = results.get("simulation_name", f"Run {i+1}")
        outcome = results.get("outcome", "unknown")
        rounds = results.get("total_rounds", 0)
        report.append(f"  {name}: {outcome} (rounds: {rounds})")

    report.append("\n--- FINAL TEXT COMPARISON ---")
    for i, results in enumerate(results_list):
        name = results.get("simulation_name", f"Run {i+1}")
        text = results.get("final_text", "")
        brackets = text.count("[")
        report.append(
            f"  {name}: {len(text.split())} words, {brackets} brackets remaining"
        )

    report.append("\n--- LLM USAGE COMPARISON ---")
    total_cost_all = 0.0
    for i, results in enumerate(results_list):
        name = results.get("simulation_name", f"Run {i+1}")
        stats = results.get("llm_stats", {})
        prompt_tokens = stats.get("total_prompt_tokens", 0)
        completion_tokens = stats.get("total_completion_tokens", 0)
        total_tokens = stats.get(
            "total_tokens_used", prompt_tokens + completion_tokens
        )
        requests = stats.get("total_requests", 0)
        model = stats.get("model", "")
        cost = _estimate_cost(model, prompt_tokens, completion_tokens)
        total_cost_all += cost
        report.append(
            f"  {name}: {total_tokens} tokens, {requests} requests, ~${cost:.4f}"
        )
    report.append(f"  TOTAL ESTIMATED COST: ${total_cost_all:.4f}")

    report.append("\n--- AGENT BEHAVIOR CONSISTENCY ---")
    agent_outcomes = defaultdict(list)
    for results in results_list:
        for agent_id, stats in results.get("agent_stats", {}).items():
            agent_outcomes[agent_id].append(stats)

    for agent_id, stats_list in sorted(agent_outcomes.items()):
        rounds = [s.get("rounds_participated", 0) for s in stats_list]
        concessions = [s.get("concessions_made", 0) for s in stats_list]
        report.append(
            f"  {agent_id}: avg_rounds={sum(rounds)/len(rounds):.1f}, "
            f"avg_concessions={sum(concessions)/len(concessions):.1f}"
        )

    report.append("\n" + "=" * 60)
    return "\n".join(report)


def extract_negotiation_narrative(results: Dict[str, Any]) -> str:
    """Extract a human-readable narrative of the negotiation from the log."""
    narrative = []
    narrative.append("=" * 60)
    narrative.append("NEGOTIATION NARRATIVE")
    narrative.append("=" * 60)

    interaction_log = results.get("interaction_log", [])
    current_phase = ""

    for entry in interaction_log:
        phase = entry.get("phase", "")
        agent = entry.get("agent", "")
        content = entry.get("content", "")

        if phase != current_phase:
            current_phase = phase
            narrative.append(f"\n{'─' * 40}")
            narrative.append(f"PHASE: {phase.upper().replace('_', ' ')}")
            narrative.append(f"{'─' * 40}")

        round_num = entry.get("round", "")
        if agent in ("CHAIR", "CHAIR_ESCALATION"):
            narrative.append(f"\n📋 [CHAIR] (Round {round_num}):")
        else:
            narrative.append(f"\n🗣️ [{agent}] (Round {round_num}):")

        if len(content) > 500:
            narrative.append(f"  {content[:500]}...")
        else:
            narrative.append(f"  {content}")

    narrative.append(f"\n{'═' * 40}")
    outcome = results.get("outcome", "unknown")
    if outcome == "ADOPTED":
        narrative.append("🎉 OUTCOME: TEXT ADOPTED BY CONSENSUS")
    else:
        narrative.append(f"⚠️ OUTCOME: {outcome}")

    narrative.append("=" * 60)
    return "\n".join(narrative)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze Climate Negotiation Simulation Results"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to results JSON file or directory of results",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all results in a directory",
    )
    parser.add_argument(
        "--narrative",
        action="store_true",
        help="Extract negotiation narrative",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save analysis to file",
    )

    args = parser.parse_args()

    if os.path.isfile(args.path):
        results = load_results(args.path)

        if args.narrative:
            report = extract_negotiation_narrative(results)
        else:
            report = analyze_single(results)

    elif os.path.isdir(args.path):
        result_files = find_result_files(args.path)

        if not result_files:
            print(f"No result files found in {args.path}")
            sys.exit(1)

        results_list = [load_results(f) for f in result_files]

        if args.compare_all:
            report = compare_runs(results_list)
        else:
            report = analyze_single(results_list[-1])
    else:
        print(f"Path not found: {args.path}")
        sys.exit(1)

    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nAnalysis saved to {args.output}")


if __name__ == "__main__":
    main()
