#!/usr/bin/env python3
"""
Main entry point for running climate negotiation simulations.

Usage:
    python experiments/run_simulation.py
    python experiments/run_simulation.py --config config/default_config.yaml --scenario config/scenarios/paris_article6_8.yaml
    python experiments/run_simulation.py --dry-run
"""

import os
import sys
import argparse
import logging
import random
import re
from datetime import datetime

import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.negotiation_engine import NegotiationEngine
from src.evaluation.evaluator import NegotiationEvaluator
from src.utils.logging_utils import setup_logging


def load_yaml(path: str) -> dict:
    """Load a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_agent_configs(scenario: dict) -> dict:
    """Load all agent configurations referenced in the scenario."""
    agent_configs = {}
    for agent_id in scenario.get("active_agents", []):
        config_path = f"config/agents/{agent_id.lower()}.yaml"
        if os.path.exists(config_path):
            agent_configs[agent_id] = load_yaml(config_path)
    return agent_configs


def slugify_path_component(value: str, fallback: str = "unnamed_simulation") -> str:
    """Convert a simulation name into a safe directory name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    slug = slug.strip("._-")
    return slug or fallback


def build_run_output_dir(
    base_output_dir: str,
    simulation_name: str,
    timestamp: str,
) -> str:
    """Build the per-run output directory."""
    return os.path.join(
        base_output_dir,
        slugify_path_component(simulation_name),
        timestamp,
    )


def apply_runtime_scenario_overrides(
    scenario: dict,
    draft_text_path: str = None,
    reference_text_path: str = None,
) -> dict:
    """Apply CLI-provided text-path overrides without mutating the source."""
    scenario = dict(scenario)
    if draft_text_path:
        scenario["draft_text_path"] = draft_text_path
    if reference_text_path:
        scenario["reference_text_path"] = reference_text_path
    return scenario


def apply_runtime_seed(config: dict, seed: int = None) -> dict:
    """Apply an optional CLI seed without mutating the source config."""
    config = dict(config)
    config["llm"] = dict(config.get("llm", {}))

    if "simulation" in config:
        config["simulation"] = dict(config.get("simulation", {}))

    if seed is not None:
        config["llm"]["seed"] = seed
        config.setdefault("simulation", {})["random_seed"] = seed

    return config


def initialize_runtime_seed(seed: int = None) -> None:
    """Seed Python and NumPy RNGs when requested."""
    if seed is None:
        return

    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        return

    np.random.seed(seed)


def run_simulation(
    config_path: str = "config/default_config.yaml",
    scenario_path: str = "config/scenarios/paris_article6_8.yaml",
    dry_run: bool = False,
    output_dir: str = None,
    draft_text_path: str = None,
    reference_text_path: str = None,
    seed: int = None,
):
    """
    Run a complete negotiation simulation.
    
    Args:
        config_path: Path to the main configuration file.
        scenario_path: Path to the scenario configuration file.
        dry_run: If True, only validate configuration without running.
        output_dir: Override output directory.
        draft_text_path: Optional override for the scenario draft text file.
        reference_text_path: Optional override for the scenario reference text file.
    """
    # Load configurations
    print(f"Loading configuration from {config_path}...")
    config = apply_runtime_seed(load_yaml(config_path), seed=seed)
    initialize_runtime_seed(seed)

    print(f"Loading scenario from {scenario_path}...")
    scenario = apply_runtime_scenario_overrides(
        load_yaml(scenario_path),
        draft_text_path=draft_text_path,
        reference_text_path=reference_text_path,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simulation_name = config.get("simulation", {}).get("name", "unnamed")
    base_output_dir = output_dir or config.get("simulation", {}).get(
        "output_dir", "outputs/"
    )
    run_output_dir = build_run_output_dir(
        base_output_dir=base_output_dir,
        simulation_name=simulation_name,
        timestamp=timestamp,
    )
    config.setdefault("simulation", {})["output_dir"] = run_output_dir

    # Set up logging
    log_level = config.get("simulation", {}).get("log_level", "INFO")
    os.makedirs(run_output_dir, exist_ok=True)
    log_file = os.path.join(run_output_dir, f"simulation_{timestamp}.log")

    setup_logging(level=log_level, log_file=log_file)
    logger = logging.getLogger(__name__)

    # Print configuration summary
    logger.info("=" * 60)
    logger.info("CLIMATE NEGOTIATION SIMULATION")
    logger.info("=" * 60)
    logger.info(f"Simulation: {config.get('simulation', {}).get('name', 'unnamed')}")
    logger.info(f"Scenario: {scenario.get('scenario_name', 'unnamed')}")
    logger.info(f"LLM Provider: {config.get('llm', {}).get('provider', 'unknown')}")
    logger.info(f"LLM Model: {config.get('llm', {}).get('model', 'unknown')}")
    if seed is not None:
        logger.info(f"Seed: {seed}")
    logger.info(f"Active agents: {scenario.get('active_agents', [])}")
    logger.info(f"Max rounds: {config.get('simulation', {}).get('max_total_rounds', 30)}")
    logger.info(f"Output: {run_output_dir}")

    if dry_run:
        logger.info("\n--- DRY RUN MODE ---")
        logger.info("Configuration validated successfully.")
        logger.info(f"Would run with {len(scenario.get('active_agents', []))} agents.")

        # Validate agent configs exist
        for agent_id in scenario.get("active_agents", []):
            config_path_agent = f"config/agents/{agent_id.lower()}.yaml"
            exists = os.path.exists(config_path_agent)
            status = "✅" if exists else "❌"
            logger.info(f"  {status} {agent_id}: {config_path_agent}")

        # Validate data files
        draft_path = scenario.get("draft_text_path", "")
        ref_path = scenario.get("reference_text_path", "")
        logger.info(f"  Draft text: {'✅' if os.path.exists(draft_path) else '❌'} {draft_path}")
        logger.info(f"  Reference text: {'✅' if os.path.exists(ref_path) else '❌'} {ref_path}")

        return None

    # Check API key
    api_key_env = config.get("llm", {}).get("api_key_env", "DEEPSEEK_API_KEY")
    if not os.environ.get(api_key_env):
        logger.error(
            f"API key not found. Set the {api_key_env} environment variable."
        )
        logger.error(f"  export {api_key_env}='your-api-key-here'")
        sys.exit(1)

    # Initialize and run
    logger.info("\nInitializing NegotiationEngine...")
    engine = NegotiationEngine(config=config, scenario=scenario)

    logger.info("Starting simulation...\n")
    start_time = datetime.now()
    results = engine.run()
    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info(f"\nSimulation completed in {elapsed:.1f} seconds.")
    logger.info(f"Outcome: {results.get('outcome', 'Unknown')}")
    logger.info(f"Total rounds: {results.get('total_rounds', 0)}")

    # Save results
    results_path = engine.save_results(timestamp=timestamp)
    logger.info(f"Results saved to: {results_path}")

    # Run evaluation if reference text exists
    ref_path = scenario.get("reference_text_path", "")
    if os.path.exists(ref_path):
        logger.info("\nRunning evaluation against reference text...")
        with open(ref_path, "r") as f:
            reference_text = f.read()

        agent_configs = load_agent_configs(scenario)
        key_clauses = scenario.get("evaluation", {}).get("key_clauses_to_check", [])

        eval_config = dict(config)
        eval_config["scenario"] = scenario
        evaluator = NegotiationEvaluator(
            eval_config,
            llm_backend=engine.llm,
        )
        eval_results = evaluator.evaluate(
            simulated_text=results.get("final_text", ""),
            reference_text=reference_text,
            interaction_log=results.get("interaction_log", []),
            agent_configs=agent_configs,
            key_clauses=key_clauses,
            scenario=scenario,
        )

        # Generate and print report
        report = evaluator.generate_report(eval_results)
        logger.info("\n" + report)

        # Save evaluation results
        eval_path = os.path.join(run_output_dir, f"evaluation_{timestamp}.json")
        import json
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2, default=str)
        logger.info(f"Evaluation saved to: {eval_path}")
    else:
        logger.info(
            "\nNo reference text found. Skipping evaluation."
            f"\n  Expected: {ref_path}"
        )

    # Print LLM usage stats
    llm_stats = results.get("llm_stats", {})
    logger.info(f"\nLLM Usage Statistics:")
    logger.info(f"  Total requests: {llm_stats.get('total_requests', 0)}")
    logger.info(f"  Total tokens: {llm_stats.get('total_tokens_used', 0)}")
    logger.info(f"  Provider: {llm_stats.get('provider', 'unknown')}")
    logger.info(f"  Model: {llm_stats.get('model', 'unknown')}")

    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Climate Negotiation Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python experiments/run_simulation.py
  
  # Dry run to validate configuration
  python experiments/run_simulation.py --dry-run
  
  # Custom config and scenario
  python experiments/run_simulation.py \\
    --config config/default_config.yaml \\
    --scenario config/scenarios/paris_article6_8.yaml

  # Reuse the default scenario but override the draft/reference text files
  python experiments/run_simulation.py \\
    --draft-text data/draft_texts/other_draft.txt \\
    --reference-text data/final_texts/other_reference.txt
  
  # Override output root directory
  python experiments/run_simulation.py --output outputs/experiment_1/
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to main configuration file",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="config/scenarios/paris_article6_8.yaml",
        help="Path to scenario configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running simulation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--draft-text",
        type=str,
        default=None,
        help=(
            "Override scenario draft_text_path. If provided, takes precedence "
            "over the scenario file."
        ),
    )
    parser.add_argument(
        "--reference-text",
        type=str,
        default=None,
        help=(
            "Override scenario reference_text_path. If provided, takes "
            "precedence over the scenario file."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible runs",
    )

    args = parser.parse_args()

    results = run_simulation(
        config_path=args.config,
        scenario_path=args.scenario,
        dry_run=args.dry_run,
        output_dir=args.output,
        draft_text_path=args.draft_text,
        reference_text_path=args.reference_text,
        seed=args.seed,
    )

    if results:
        # Exit with appropriate code
        if results.get("outcome") == "ADOPTED":
            sys.exit(0)
        elif "FAILED" in results.get("outcome", ""):
            sys.exit(2)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
