# Negotiation Process Generalization Notes

## Purpose

This note records process-level improvements that are intended to generalize
across scenarios, agenda items, and draft texts rather than optimize for a
single Article 6.8 run.

## Design Principles

1. Keep scenario-specific substance in configuration, not in hard-coded logic.
2. Improve evaluator and process behavior using reusable rules that work for
   any negotiation text with brackets, red lines, and final plenary decisions.
3. Distinguish between:
   - textual similarity to a reference outcome
   - political acceptability within the simulated negotiation
4. Treat non-final-phase acceptance of bracketed or conditional text as a
   procedural drafting move unless there is strong evidence of substantive
   red-line abandonment.

## Implemented General Improvements

### 1. Configurable Clause Matching

The evaluator now supports richer clause definitions with optional fields:

- `aliases`
- `patterns`
- `required_terms`
- `min_required_terms`
- `min_numeric_refs`
- `keyword_threshold`

This allows each scenario to specify how a concept may appear in text without
changing evaluator code.

### 2. Phase-Aware Stance Evaluation

Stance consistency now distinguishes between:

- final acceptance or opposition of a package
- procedural acceptance of bracketed or option-bearing draft language

This reduces false positives in first reading and consultation rounds for any
scenario that uses bracketed drafting.

### 3. Separate Political Acceptability Signal

The evaluator now computes a distinct `acceptability` section based on final
plenary positions. This complements text-quality metrics and helps distinguish:

- "the text looks realistic"
- from "the package could actually pass"

### 4. Split Summary Scores

The evaluator now reports:

- `reference_alignment_score`
- `negotiation_quality_score`

The historical `overall_score` is retained for backward compatibility.

## Scenario Authoring Guidance

For new scenarios, prefer defining evaluation clauses as concepts rather than
single literal phrases. For example, use aliases or regex patterns when a
concept may be expressed in multiple valid procedural forms.

## Non-Goals

These changes do not try to hard-code Article 6.8 content, any single bloc's
preferred wording, or one specific final text. The goal is to make the
negotiation framework more robust across future agenda items.

## Validation

Validated with:

- `conda run -n llm_climate_negotiation python -m pytest tests/ -q`
- `conda run -n llm_climate_negotiation python -m unittest discover -s tests -v`
- `conda run -n llm_climate_negotiation python -m compileall src experiments tests`
- `conda run -n llm_climate_negotiation python experiments/run_simulation.py --dry-run`

Full reference-based evaluation should be run from the
`llm_climate_negotiation` environment because `rouge-score` and `bert-score`
are required for the complete metric suite.

## Example Impact on an Existing Run

The changes were sanity-checked by re-evaluating an existing run from
`outputs/deepseek/COP_Negotiation_Simulation_DeepSeek/20260410_160834/`.
This comparison is included only as a validation example. It is not used as a
hard-coded target inside the evaluator.

Observed improvements:

- Final plenary positions can now be summarized as political acceptability:
  7 accepting parties and 1 blocking party.
- Clause matching now recognizes valid aliases such as
  `reporting and transparency`, reducing literal-phrase false negatives.
- Clause matching now requires explicit numeric references for clauses like
  `Article 6.2 and 6.4`, reducing false positives from generic `Article 6`
  wording.
- Stance consistency is less likely to penalize agents for accepting
  bracketed draft text during first reading or consultations.

Illustrative before/after changes from that run:

- `reporting requirements`: `false` -> `true`
- `relationship to Article 6.2 and 6.4`: `true` -> `false`
- `AOSIS` stance consistency: `0.429` -> `1.000`
- `UMBRELLA` stance consistency: `0.571` -> `1.000`

These are examples of cleaner measurement, not evidence that the framework is
optimized for one outcome. The implementation remains scenario-configurable and
should transfer to other draft texts that use different clause names,
institutional arrangements, or red-line structures.
