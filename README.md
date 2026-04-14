# Climate Negotiation Simulation

A multi-agent LLM-based simulation of UNFCCC climate negotiations.

## Overview

This system uses large language models such as DeepSeek, GPT, and Claude to
simulate climate negotiation processes between multiple Parties and
negotiating groups in the UNFCCC context. Each agent represents a negotiating
bloc with specific positions, red lines, and behavioral parameters.

## Architecture

```text
Simulation Controller
├── Negotiation Engine (orchestrator)
│   ├── Phase Manager (tracks negotiation phases)
│   ├── Text Manager (manages draft text, brackets, amendments)
│   ├── Turn Manager (speaking order, right of reply)
│   └── Amendment Processor (parses agent proposals)
├── Agent Layer
│   ├── Negotiation Agents (EU, G77, AOSIS, etc.)
│   └── Chair Agent (neutral facilitator)
├── LLM Backend (DeepSeek / OpenAI / Anthropic)
├── Memory System (tiered: core, working, summary)
└── Evaluation Engine (ROUGE, BERTScore, key clause matching)
```

## Quick Start

### Prerequisites

- Python 3.9+
- API key for at least one LLM provider:
  - DeepSeek
  - OpenAI
  - Anthropic

### Installation

You can use either `venv` or `conda`. This workspace is now configured to use
the dedicated conda environment `llm_climate_negotiation` in VS Code.

#### Option A: Conda

```bash
conda activate llm_climate_negotiation
python -V
```

If you have not created it yet, the equivalent setup is:

```bash
conda create -n llm_climate_negotiation python=3.11 pip -y
conda activate llm_climate_negotiation
pip install -r requirements.txt
```

#### Option B: venv

```bash
git clone https://github.com/YOUR_USERNAME/climate-negotiation-sim.git
cd climate-negotiation-sim

python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Replace `YOUR_USERNAME` with your actual GitHub handle before publishing.

If `python -m pytest` reports `No module named pytest`, make sure you activated
the same virtual environment where you ran `pip install -r requirements.txt`.

For VS Code, the workspace interpreter is configured at
`.vscode/settings.json` to use:

`/Users/ziqiwei/.julia/conda/3/aarch64/envs/llm_climate_negotiation/bin/python`

### Setting Up Your API Key

DeepSeek is now the default provider for this repo.

```bash
# Default: DeepSeek
export DEEPSEEK_API_KEY="sk-..."

# Optional alternatives:
# export OPENAI_API_KEY="sk-..."
# export ANTHROPIC_API_KEY="sk-ant-..."
```

`config/default_config.yaml` is now DeepSeek-first by default. The other
providers are preserved there as commented reference lines.

If you want to switch providers later, edit `config/default_config.yaml` or use
a provider-specific config file such as `config/deepseek_config.yaml`:

```yaml
llm:
  provider: "deepseek"
  # provider: "openai"
  # provider: "anthropic"
  model: "deepseek-chat"
  # model: "gpt-4"
  # model: "claude-3-5-sonnet"
  api_key_env: "DEEPSEEK_API_KEY"
  # api_key_env: "OPENAI_API_KEY"
  # api_key_env: "ANTHROPIC_API_KEY"
```

### DeepSeek Quick Start

This repo already includes a ready-to-use config:

`config/deepseek_config.yaml`

What you need to provide:

- A valid `DEEPSEEK_API_KEY`
- Optionally, a different DeepSeek model name if you do not want
  `deepseek-chat`
- Optionally, a different scenario file if you want to test another case
- Optionally, custom draft/reference text paths if you want to reuse the
  default 6.8 setup with different text inputs

What you do not need to provide:

- A custom base URL
- An organization ID
- Extra SDK configuration

The backend already points DeepSeek requests at `https://api.deepseek.com`.

Both `config/default_config.yaml` and `config/deepseek_config.yaml` are now
ready for DeepSeek. The dedicated config mainly keeps DeepSeek experiments
separate under `outputs/deepseek/`.

Run a dry run first:

```bash
export DEEPSEEK_API_KEY="sk-..."
python experiments/run_simulation.py --config config/deepseek_config.yaml --dry-run
```

Then run a real experiment:

```bash
export DEEPSEEK_API_KEY="sk-..."
python experiments/run_simulation.py --config config/deepseek_config.yaml
```

### Verifying The Setup

```bash
python experiments/run_simulation.py --dry-run
```

By default, the entry point uses the Article 6.8 scenario. If you pass
`--scenario`, `--draft-text`, or `--reference-text`, those explicit values
take precedence over the defaults.

### Running A Simulation

```bash
# Full run with the default DeepSeek + Article 6.8 setup
python experiments/run_simulation.py

# Dry run with the default DeepSeek + Article 6.8 setup
python experiments/run_simulation.py --dry-run

# Use a different scenario file
python experiments/run_simulation.py \
  --scenario config/scenarios/your_scenario.yaml

# Keep the default scenario settings but override the text files
python experiments/run_simulation.py \
  --draft-text data/draft_texts/your_draft.txt \
  --reference-text data/final_texts/your_reference.txt

# Use a different config and scenario
python experiments/run_simulation.py \
  --config config/deepseek_config.yaml \
  --scenario config/scenarios/your_scenario.yaml \
  --output outputs/deepseek_run/
```

Priority order for runtime inputs:

1. Explicit CLI overrides such as `--draft-text` and `--reference-text`
2. The scenario file passed with `--scenario`
3. The built-in default scenario: `config/scenarios/paris_article6_8.yaml`

If a new scenario needs sharper, text-specific agent guidance, add it in the
scenario file rather than editing the base bloc YAMLs. The runtime can read an
optional `runtime_briefing` block such as:

```yaml
runtime_briefing:
  shared_guidance:
    - "Stay within the current agenda item."
  per_agent_guidance:
    EU:
      - "Prioritize coherence and legal clarity in this scenario."
    G77_CHINA:
      - "Check whether support for developing countries remains explicit."
```

This keeps `config/agents/*.yaml` as reusable general charters while allowing
scenario-specific refinement at runtime.

Each real run is saved in its own folder:

```text
outputs/<simulation_name>/<timestamp>/
```

For example, the default run writes to a folder like
`outputs/COP_Negotiation_Simulation/20260410_145500/`.
The `--output` flag now sets the output root, and the script still creates
the per-simulation and per-run subfolders automatically.

### Analyzing Results

```bash
python experiments/analyze_results.py outputs/
python experiments/analyze_results.py outputs/COP_Negotiation_Simulation/20260410_145500/ --narrative
python experiments/analyze_results.py outputs/ --compare-all
```

### Running Tests

```bash
python -m pytest tests/ -v
python -m unittest discover -s tests -v
python -m compileall src experiments tests
python experiments/run_simulation.py --dry-run
```

## Project Structure

```text
climate-negotiation-sim/
├── config/
│   ├── default_config.yaml
│   ├── agents/                 # EU, G77, AOSIS, Umbrella, LDC, etc.
│   └── scenarios/              # Negotiation scenarios
├── data/
│   ├── draft_texts/            # Initial negotiating text with brackets
│   └── final_texts/            # Real-world final text for evaluation
├── experiments/
│   ├── run_simulation.py       # Main entry point
│   └── analyze_results.py      # Result analysis
├── outputs/                    # Simulation results, grouped by simulation/run
├── src/
│   ├── agents/                 # BaseAgent, NegotiationAgent, ChairAgent
│   ├── engine/                 # NegotiationEngine, PhaseManager, TextManager
│   ├── evaluation/             # NegotiationEvaluator
│   ├── llm/                    # LLMBackend, PromptTemplates
│   ├── memory/                 # NegotiationMemory
│   └── utils/                  # Logging utilities
├── tests/                      # Unit tests
├── requirements.txt
└── README.md
```

## Key Design Decisions

### Agent Architecture

- Each agent has a system prompt encoding identity, positions, and red lines.
- A tiered memory system prevents context overflow.
- Stance reinforcement periodically reminds agents of their positions.
- Behavioral parameters shape stubbornness, coalition tendency, and related
  negotiation behavior.

### Negotiation Process

The simulation follows a UNFCCC-style process:

1. Opening Statements: each group states its position.
2. First Reading: paragraph-by-paragraph review with amendments.
3. Informal Consultations: multi-round, chair-facilitated negotiation.
4. Final Plenary: consensus check for adoption.

### Evaluation

The system supports multi-dimensional evaluation against reference outcomes:

- Text similarity with ROUGE and BERTScore
- Key clause matching
- Structural similarity
- Bracket resolution rate
- Stance consistency
- Process realism

## Extending The System

### Adding A New Agent

1. Create `config/agents/new_agent.yaml` following existing templates.
2. Add the new agent ID to the scenario's `active_agents` list.

### Adding A New Scenario

1. Create draft text in `data/draft_texts/`.
2. Create reference text in `data/final_texts/`.
3. Create `config/scenarios/new_scenario.yaml`.

### Changing LLM Provider

Update `config/default_config.yaml`:

```yaml
llm:
  provider: "deepseek"
  # provider: "openai"
  # provider: "anthropic"
  model: "deepseek-chat"
  # model: "gpt-4"
  # model: "claude-3-5-sonnet"
  api_key_env: "DEEPSEEK_API_KEY"
  # api_key_env: "OPENAI_API_KEY"
  # api_key_env: "ANTHROPIC_API_KEY"
```

## Limitations

- LLMs may exhibit cooperation bias and reach agreement too easily.
- Context window limits constrain negotiation complexity.
- Agents do not learn across separate simulation runs.
- Real negotiations include corridor discussions and informal dynamics that
  are not represented here.
- Cultural, institutional, and personal factors are difficult to encode.
- Some config options are documented but not yet active:
  - `merge_strategy`
  - `verbosity`
  - `max_concessions_per_round`

## Citation

If you use this system in academic work, you can adapt the following:

```bibtex
@software{climate_negotiation_sim,
  title  = {Climate Negotiation Simulation: A Multi-Agent LLM Framework},
  author = {Ziqi Wei},
  year   = {2026},
  note   = {GitHub repository},
  url    = {https://github.com/YOUR_USERNAME/climate-negotiation-sim}
}
```

Replace `YOUR_USERNAME` before publishing if you put this on GitHub.

## License

MIT License
