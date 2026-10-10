# Eval Dataset Generation Pipeline

Generates a JSONL dataset of ASP constraint instances with natural language descriptions.
Each record pairs an instantiated ASP rule with short/medium/long NL text and clingo solver output.
This is the evaluation dataset — see [../finetune_dataset_gen/README.md](../finetune_dataset_gen/README.md)
for the sibling pipeline that captions existing images instead.

## Pipeline steps

1. **Load** — read an ASP template (`.txt`) from `constraint_templates/`
2. **Instantiate** — randomly assign primed placeholders (`P1'`, `V1'`, `D1'`, `N'`, …) respecting `!=` constraints in the rule
3. **Verbalize** — produce short, medium, and long NL descriptions for the constraint, plus a `subqa`
   dict (question → expected answer) used by `ccig-evaluation`'s `soft-tifa` method
4. **Solve** — run clingo on the full ASP program (background + instantiated rule), record SAT/UNSAT and grounded models
5. **Write** — append a JSON record to the output `.jsonl` file

## Files

| File | Role |
|------|------|
| `domain.py` | Property/value/direction domains (derives from `../common/domain_clevr.py`, excluding `material`); background ASP program generator |
| `instantiate.py` | Template loading, placeholder extraction, valid assignment sampling |
| `../common/verbalize.py` | NL verbalization (short/medium/long + subqa) for all 9 constraint classes (C1–C9) — shared with `finetune_dataset_gen`, which grounds it against real scenes instead of instantiating it randomly |
| `solve.py` | Clingo wrapper; scene formatter for grounded answer sets |
| `run.py` | CLI entry point orchestrating the full pipeline; `--domain` (default `clevr`) tags each output record with which domain's vocabulary was used |

## Run

```bash
cd ccig-dataset-gen

python -m src.eval_dataset_gen.run --classes C1 C2 C3 C4 C5 C6 C7 C8 C9 --samples 1  --output /users/sbsh670/data/ccig_evalData --combo 1 --domain coco
 
# Default: 10 random samples per template, 4-object scenes
python -m src.eval_dataset_gen.run

# Custom number of samples and output path
python -m src.eval_dataset_gen.run --samples 20 --output ../my_dataset.jsonl

# Restrict to specific constraint classes
python -m src.eval_dataset_gen.run --classes C1 C3 C8

# Skip clingo solving (NL verbalization only, no SAT/UNSAT)
python -m src.eval_dataset_gen.run --no_solve

# Tag output records with a domain (default: clevr;)
python -m src.eval_dataset_gen.run --domain clevr

```

Outputs two files derived from `--output` (default: `../data/ccig_eval_dataset.jsonl`, i.e. the
repo-root `data/` folder):
- `ccig_eval_dataset_SAT.jsonl`
- `ccig_eval_dataset_UNSAT.jsonl`

## Output format

Each line in both files is one JSON record:

```json
{
  "id": "C1-1prop-a3f9b2-000",
  "domain": "clevr",
  "complexity_class": "C1",
  "constraint_family": "1prop",
  "prompts": {
    "short": "A red object is in the scene.",
    "medium": "The scene contains at least one red object.",
    "long": "..."
  },
  "subqa": {
    "Does the image contain at least one red object?": "yes",
    "How many objects are red in the image?": "> 0",
    "State True or False: No object is red?": "False"
  },
  "instantiated_rule": ":- #count { X : object(X), hasProperty(X, color, red)} = 0.",
  "asp_template_file": "C1_1prop.txt",
  "status": "SAT",
  "number_of_objects": 4
}
```
