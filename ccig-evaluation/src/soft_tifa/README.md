# soft-tifa

Soft-TIFA scoring, following [facebookresearch/GenEval2](https://github.com/facebookresearch/GenEval2):
for each generated image, ask a VQA model a battery of yes/no, true/false, counting,
and property-value questions about it, and score each one by *the probability the
model assigns to the correct answer* — not whether its top guess happens to be right.
That's the "soft" part: a model that's 90% confident and correct scores 0.9, not 1.0;
a model that's 55/45 torn between right and wrong scores 0.55, not a hard win.

The questions come from `ccig-dataset-gen`'s `verbalize.py`, which already produces a
`subqa` dict (question → expected answer) for every constraint it verbalizes — that's
the design decision that makes this module possible, not something invented here.

## Is your dataset ready?

`subqa` is a newer field. Check one record:

```bash
python -c "import json; print('subqa' in json.loads(open('../data/ccig_eval_dataset/clevr_1_scenes_SAT.jsonl').readline()))"
```

If that prints `False`, regenerate the dataset first — from `ccig-dataset-gen/`:

```bash
python -m src.eval_dataset_gen.run --output ../data/ccig_eval_dataset
```

Records without `subqa` are skipped (not fatal) by `run_soft_tifa`, so partially-stale
datasets still run — you just get fewer scored images than you started with, and a
`[warn]` line telling you how many.

## How a question gets scored

`answer_spec.py` first classifies each `subqa` answer string into one of four shapes
(see its module docstring for the full grammar) — every shape `verbalize.py`'s nine
constraint classes actually produce, checked against the live verbalizer, not guessed:

| Answer shape | Example | Forced-choice candidates | Score |
|---|---|---|---|
| yes/no | `"yes"` | `["Yes", "No"]` | P(the correct one) |
| True/False | `"False"` | `["True", "False"]` | P(the correct one) |
| count comparison | `"> 0"`, `"c > 0"`, `" = c"` | `["0", "1", ..., max_count]` | sum of P(k) over every k that satisfies the comparison |
| open value | `"blue"` | every value in that property's domain vocabulary (e.g. all 8 CLEVR colors) | P(the correct one) |

The VQA model is always prompted with the question plus its forced candidate list,
generated with `max_new_tokens=1` (or the API equivalent), and its softmax
probability over the *entire* vocabulary is read off for those candidate tokens —
not renormalized to sum to 1 over just the candidates. If the model spreads mass onto
something outside the candidate set entirely, that shows up as low scores across the
board rather than being hidden by renormalization. See `base.py`'s docstring.

### Symbolic counts (the one real wrinkle -- and not part of GenEval2)

Some `subqa` dicts are stateful: one question defines a count (`"How many objects
are red?": "c > 0"`), and a later question refers back to it (`"How many are both red
and cube?": " = c"`) rather than repeating a fixed number. **GenEval2's own benchmark
has no equivalent of this** -- every question there has one fixed, literal expected
answer, with no question referring to another's answer. This cross-question linking
exists only because `verbalize.py`'s `subqa` dicts need it; nothing about how it's
resolved comes from the paper.

A VQA model can't literally see "c", so `scoring.py` resolves it the direct way:
whichever question defines a symbol, its top answer (the model's single highest-
probability candidate -- what it would have said if just asked to answer with one
number) is kept as a plain integer. Later questions that reference that symbol are
checked against that one stored number, the same way you'd read the model's answer
off the screen and reuse it by hand -- no joint distribution, no marginalizing across
both questions' uncertainty. The defining question's *own* comparison (`c > 0` in the
example above) is still scored the normal soft-TIFA way (candidate probabilities),
and its full candidate distribution is kept in `SubQAScore.candidates` for inspection
even though only the top answer is used to resolve the symbol downstream.

Questions with **no comparator at all** (`"How many objects are in the image?": "n"`)
only exist to define a symbol for later questions — they have no expected answer of
their own to check, so they're excluded from the score (`excluded_from_score: true`
in the output) but still recorded so you can see what the model answered.

## Backends

Two, registered in `registry.py` exactly like `judge/registry.py`:

- **`gpt-4o`** (`closed/gpt4o.py`, default) — OpenAI API, needs `OPENAI_API_KEY`
  (same variable `vlm-judge` uses). Reads `logprobs`/`top_logprobs=20` on the one
  forced output token; a candidate not among those top 20 alternatives scores 0.0.
  For the small candidate sets used here (2-3 for yes/no or True/False, up to ~10
  digits, a handful of property values) a plausible candidate essentially always
  makes the top 20, so this is a fair stand-in for exact softmax mass, not a
  fundamentally weaker method.
- **`qwen2-vl`** (`open/qwen2_vl.py`) — local, open-weight
  (`Qwen/Qwen2-VL-7B-Instruct`, same checkpoint `vlm-judge`'s `qwen2-vl` backend
  uses), reads the model's exact full-vocabulary softmax directly. No API cost, GPU
  strongly preferred for latency.

(GenEval2 itself uses `Qwen3-VL-8B-Instruct`; the *method* — forced single-token
softmax over candidate answers — is what defines soft-TIFA, not that specific
checkpoint, so this reuses the checkpoint this repo's `vlm-judge` already depends on
rather than adding a new model to install and validate.)

## Files

| File | Role |
|---|---|
| `answer_spec.py` | Parses a `subqa` answer string into a typed spec (`YesNoSpec`, `TrueFalseSpec`, `CountSpec`, `OpenValueSpec`). Pure Python, no model dependency — see its module docstring for the full answer grammar. |
| `base.py` | `VQABackend` interface: `answer_distribution(image, question, candidates) -> {candidate: probability}`. |
| `scoring.py` | `score_subqa()` — walks one image's `subqa` dict in order, resolving symbol references as it goes, producing one `SubQAScore` per question. |
| `open/qwen2_vl.py`, `closed/gpt4o.py` | The two backends. |
| `registry.py` | `VQA_REGISTRY` + `build_vqa_backend(name, device=None)`, selected by `--vqa-backend`. |
| `run.py` | `run_soft_tifa(items, domain_module, backend, out_path)` — per-image scoring, AM/GM aggregation, writes `results.json`. |

## Run

```bash
cd ccig-evaluation

# gpt-4o backend (default) -- needs OPENAI_API_KEY
python -m src.run \
  --images-dir ../data/generated_images/gpt-image-1 \
  --prompts-file ../data/ccig_eval_dataset/clevr_1_scenes_SAT.jsonl \
  --domain clevr \
  --method soft-tifa

# local qwen2-vl backend
python -m src.run \
  --images-dir ../data/generated_images/gpt-image-1 \
  --prompts-file ../data/ccig_eval_dataset/clevr_1_scenes_SAT.jsonl \
  --domain clevr \
  --method soft-tifa --vqa-backend qwen2-vl --device cuda

# combine with the other methods in one run
python -m src.run --images-dir ../data/generated_images/gpt-image-1 \
  --prompts-file ../data/ccig_eval_dataset/clevr_1_scenes_SAT.jsonl --domain clevr \
  --method clipscore vlm-judge soft-tifa \
  --clip-checkpoint <path-or-hf-repo> --judge-backend gpt-4o --vqa-backend gpt-4o
```

No new dependencies: `gpt-4o` only needs `requirements.txt` (already has `openai`),
`qwen2-vl` needs `requirements-open.txt` (already has `torch`/`transformers`) — same
split as `vlm-judge`.

## Output

`outputs/<images-dir-name>/soft_tifa/results.json`:

```json
{
  "method": "soft-tifa",
  "backend": "gpt-4o",
  "dataset_score_am": 0.81,
  "dataset_score_gm": 0.74,
  "results": [
    {
      "id": "23",
      "prompt_field": "short",
      "image_path": "...",
      "score_am": 0.83,
      "score_gm": 0.79,
      "subquestions": [
        {
          "question": "How many objects are cyan?",
          "expected_answer": "c > 0",
          "answer_type": "count_comparison",
          "candidates": {"0": 0.02, "1": 0.61, "2": 0.30, "...": "..."},
          "score": 0.98,
          "excluded_from_score": false
        }
      ],
      "success": true,
      "error": null
    }
  ]
}
```

- `score_am` / `score_gm` on each result: that **image's** arithmetic/geometric mean
  across its own sub-questions (soft-TIFA's "atom-level" score).
- `dataset_score_am` / `dataset_score_gm` at the top: the mean of those per-image
  scores across the whole run (soft-TIFA's "prompt-level" benchmark score) — GM
  punishes a single badly-wrong sub-question much harder than AM does, since one
  score near 0 drags the whole product down; use whichever matches how strict you
  want "getting one thing wrong" to be.
- A record with no `subqa` (stale dataset) or an image that failed to load gets
  `success: false` and `score_am`/`score_gm: null`, same as the other methods'
  failure convention.

## Adding a new VQA backend

Subclass `VQABackend` (`base.py`), implement `answer_distribution()`, register it in
`VQA_REGISTRY` (`registry.py`) — nothing else needs to change, same pattern as
`judge/`'s `VLMJudge` backends.
