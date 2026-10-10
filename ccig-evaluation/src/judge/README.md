# judge

VLM-as-judge: ask a vision-language model to rate prompt-image alignment on a 1-5
scale (normalized to 0-1 in `JudgeResult.score`), pluggable across backends.

- **`base.py`** — the `VLMJudge` interface (`judge(image, prompt) -> JudgeVerdict`),
  the shared rubric prompt, and the `SCORE:`/`RATIONALE:` response parser every
  backend reuses.
- **`closed/gpt4o.py`** — `GPT4oJudge`, an OpenAI API backend (needs `OPENAI_API_KEY`).
  Mirrors `ccig-image-generation/src/closed/gpt_image.py`'s API-client pattern.
- **`open/qwen2_vl.py`** — `Qwen2VLJudge`, a local Hugging Face model
  (`Qwen/Qwen2-VL-7B-Instruct`). No API cost, runs on CPU or GPU (GPU strongly
  preferred for latency) via the same `device` knob used throughout this pipeline.
- **`registry.py`** — `JUDGE_REGISTRY` + `build_judge(name, device=None)`, selected
  by `--judge-backend` on the CLI.

Pick `gpt-4o` for the strongest general-purpose judgment with no local compute;
pick `qwen2-vl` for free, repeatable, GPU-local scoring at some accuracy cost.

Add a backend by subclassing `VLMJudge`, implementing `judge()`, and registering it
in `JUDGE_REGISTRY` — nothing else needs to change.
