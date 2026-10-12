# LogicVal: A Benchmark for Logical Constraint Satisfaction in Text-to-Image Generation

> **Anonymous submission.** This repository accompanies a paper under double-blind review.

LogicVal treats text-to-image (T2I) evaluation as a **constraint-satisfaction problem**.
The benchmark domain (objects, attributes, spatial relations) and the constraints a generated
image must satisfy are written in Answer Set Programming (ASP). Specifications are labelled
**SAT** or **UNSAT** by [clingo](https://potassco.org/clingo/) and verbalised into prompts.
Generated images are then checked by the **LogicVal verifier**, which extracts a scene graph
with neural detectors and tests whether it logically entails the specification. No reference
image is needed.

```
ASP specification C ──clingo──► SAT/UNSAT + prompt ──T2I──► image I ──detect+ground──► scene graph Ŝ(I) ──clingo──► Ŝ(I) ⊨ C ?
```

## Highlights

- **9 constraint classes and 71 ASP template families**, from single-object attribute binding
  (∃, ∀, →) through relational subgraphs with quantifier alternation (∃∀, ∀∃) to counting and
  count comparison.
- **509 solver-labelled specifications** on CLEVR, combining 1, 3 or 6 constraints, both
  satisfiable and contradictory.
- **3,887 human-annotated scene graphs** for images from GPT-Image-2, Gemini-3-Pro-Image and
  LoRA-adapted FLUX.1-dev and Qwen-Image.
- **Findings:** the best model satisfies 85% of single constraints but only 20% of
  six-constraint specifications, and recognises at most 25% of contradictory prompts.
  CLIPScore is at chance level. The LogicVal verifier agrees with human judgments with a
  ROC-AUC of 0.88–0.98.

## Repository structure

The five pipelines run independently and exchange files through a repo-root `data/` folder.
Each one has its own README with setup instructions and the full set of options.

| Pipeline | Role |
|----------|------|
| [`ccig-dataset-gen`](ccig-dataset-gen/README.md) | ASP constraint templates, clingo labelling and verbalisation into the LogicVal benchmark. Also builds the captions used for LoRA training. |
| [`ccig-image-generation`](ccig-image-generation/README.md) | Generates images from benchmark prompts with proprietary (API) and open-weight (diffusers) T2I models. |
| [`ccig-finetuning`](ccig-finetuning/README.md) | Optional LoRA domain adaptation of open models on CLEVR. |
| [`ccig-evaluation`](ccig-evaluation/README.md) | The LogicVal verifier (`perception`) and the baseline metrics CLIPScore, VLM-Judge and Soft-TIFA, plus the meta-evaluation against human labels. |
| [`ccig-human-evaluation`](ccig-human-evaluation/README.md) | Web app for annotating scene graphs by hand. |

Typical flow: `dataset-gen → image-generation (→ finetuning) → evaluation / human-evaluation`.

`ccig-evaluation` imports code from `ccig-dataset-gen` at runtime, so keep both directories side
by side. All pipelines need Python ≥ 3.10 and `clingo` on your `PATH`.

## Data

The benchmark files, generated images, human annotations and checkpoints are too large for this
repository and will be released with the camera-ready version. CLEVR is available from its
[original authors](https://cs.stanford.edu/people/jcjohns/clevr/).

## Citation

```bibtex
@inproceedings{anonymous2026logicval,
  title  = {LogicVal: A Benchmark for Logical Constraint Satisfaction in Text-to-Image Generation},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## License
The code is released under the MIT License (see `LICENSE`). The benchmark data, human annotations
and generated images will be released under CC BY 4.0 with the camera-ready version.
CLEVR is © its original authors and licensed under CC BY 4.0.

