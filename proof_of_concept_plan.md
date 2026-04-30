# Proof of Concept Plan: Multi-Agent LLM Simulation of UNFCCC Negotiations

## What I Built

I built a working multi-agent system in which 8 LLM agents represent major UNFCCC negotiating blocs: EU, G77 and China, AOSIS, Umbrella Group, LDCs, African Group, LMDC, and EIG. The agents negotiate over real UNFCCC draft text using a realistic COP-style process: opening statements, first reading, informal consultations, and final plenary. A Chair agent synthesizes proposals, tracks disagreement, and moves the negotiation toward convergence.

The system is already operational and validated at the software level. The codebase contains more than 30 files, all 69 unit tests pass, and dry-run validation succeeds. Output quality can be evaluated against real UNFCCC final text using ROUGE, BERTScore, key clause matching, and stance consistency metrics.

## What This PoC Needs to Demonstrate

- Agents maintain distinct and realistic negotiating positions across multiple rounds.
- Simulated final text shows measurable similarity to real COP outcomes, especially the Article 6.8 decision text.
- The system captures expected political dynamics. For example, removing LMDC should make agreement easier, while removing AOSIS should weaken ambition-related language.

## Experiment Plan

| Phase | Goal | Runs | Model | Estimated Cost |
|---|---|---:|---|---:|
| 1 | Smoke test with small configurations to confirm stable execution and coherent dialogue | 6 | GPT-5.4-mini | $5 |
| 2 | Full validation with 8 agents and multi-round negotiation | 12 | GPT-5.4-mini | $20 |
| 3 | Reproducibility testing under repeated identical settings | 10 | GPT-5.4-mini | $10 |
| 4 | Sensitivity analysis by varying agent stubbornness and related parameters | 16 | GPT-5.4-mini | $18 |
| 5 | Counterfactual runs removing key blocs such as LMDC, AOSIS, or EU | 10 | GPT-5.4-mini | $12 |
| 6 | Comparative validation on selected cases | 8 | Claude 3.5 Sonnet | $25 |
| 7 | Buffer for reruns, debugging, and API cost variation | - | GPT-5.4-mini / Claude 3.5 Sonnet | $10 |

**Total budget requested: $100**

## Expected Deliverables

- Quantitative evidence that agents maintain bloc-consistent positions across negotiation rounds.
- ROUGE-L and BERTScore comparisons between simulated outcomes and the real Article 6.8 final text.
- Sensitivity results showing that the system responds meaningfully to political and behavioral changes.
- Comparative results using GPT-5.4-mini and Claude 3.5 Sonnet.
- Reproducible logs, outputs, and analysis scripts suitable for a methods and preliminary results section.

## Why This Is Worth Funding

This workflow tests a novel application of multi-agent LLM systems to multilateral climate treaty negotiation under realistic procedural rules and with verifiable real-world benchmarks. A $100 budget is to support a focused experimental campaign that demonstrates feasibility, generates measurable results, and provides a strong foundation for a larger paper on computational simulation of international climate governance.
