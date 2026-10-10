# CCIG Finetuning

LoRA finetunes open-source text-to-image models on `data/clevr-dataset/` (CLEVR images
paired with constraint descriptions). Produces checkpoints consumed by
`ccig-image-generation`'s `--checkpoint` flag -- finetuning and generation are separate,
decoupled modules connected only by that checkpoint-directory convention.

## Structure

```
src/
├── common/          # shared types + dataset loading, model-agnostic
├── models/          # one LoRA trainer per model (mirrors ccig-image-generation/src/open/)
├── registry.py       # model name -> trainer class
└── run.py            # CLI entry point
configs/               # one YAML per model: dataset paths, LoRA/training hyperparameters
```

Checkpoints land in the repo-level `outputs/` folder (shared across pipelines), under
`outputs/checkpoints-finetuning/<model>/<run_name>-step<NNNNNN>/`.

## Models

| model            | status      | gated? | notes                                     |
|-------------------|-------------|--------|----------------------------------------------|
| `sd3.5-large`     | signature-checked | yes | `StableDiffusion3Pipeline`, flow-matching training step |
| `flux.1-dev`      | signature-checked | yes | `FluxPipeline`, packed-latent flow-matching step, guidance-distilled |
| `flux.1-schnell`  | signature-checked | yes | `FluxPipeline`, same training step as `flux.1-dev`, not guidance-distilled |
| `qwen-image`      | signature-checked | no | `QwenImagePipeline`, packed-latent flow-matching step, guidance-distilled |

"gated" is per the Hugging Face Hub API (checked directly, not assumed) -- gated repos need you
to accept the model's license on its Hub page while logged in, then `huggingface-cli login`
locally before `from_pretrained` will succeed.

"Signature-checked" means each model's `_training_step` was written and cross-checked against
the *actual* transformer `forward()` signature, scheduler class, and pipeline `encode_prompt`
return values by introspecting the installed `diffusers` package (`python -c "import
inspect, diffusers; ..."`) -- not assumed by analogy between models. That catches real
mismatches: e.g. FLUX.1-dev's guidance-distilled transformer requires a `guidance` tensor at
every forward call or it raises inside `time_text_embed`. None of these have been run
end-to-end on real GPU hardware yet, though -- that's the remaining gap between "should be
correct" and "verified by a completed training run." If a run surfaces a mismatch, fix it in
that model's file and update this table.

All models share the model-agnostic parts in `src/models/_diffusers_common.py` (pipeline
loading, LoRA injection via `peft`, dataset/dataloader, optimizer loop, checkpoint saving).
There's deliberately no generic training step: diffusers architectures differ in scheduler
type (flow-matching vs. DDPM) and transformer call signature (packed vs. unpacked latents,
extra positional ids, pooled projections, ...), so each model implements its own
`_training_step`.

## Setup

```bash
cd ccig-finetuning
pip install -r requirements.txt
huggingface-cli login   # required for gated repos: sd3.5-large, flux.1-dev, flux.1-schnell
```

## Run

```bash
python -m src.run --model sd3.5-large --config configs/sd3.5-large.yaml
python -m src.run --model flux.1-schnell --config configs/flux.1-schnell.yaml --run-name run2 --max-steps 500
```

`max_steps` defaults to **one full epoch** over whatever `dataset_path` points at (resolved
automatically from the dataset's actual size once loaded -- see `TrainConfig.max_steps` and
`DiffusersLoraTrainer.train()`) unless set explicitly in the config or via `--max-steps`.

To train for more than one epoch, use `--epochs N` instead of computing a step count by hand
(`max_steps = N * steps_per_epoch`, resolved the same way once the dataset is loaded) -- more
portable than a raw step count, since each batch file has a different size:

```bash
python -m src.run --model flux.1-dev --config configs/flux.1-dev.yaml \
  --dataset ../data/clevr-dataset/finetuning-data/batches/batch_001.json --epochs 2
```

Set at most one of `max_steps`/`epochs` (in the config or via CLI) -- setting both raises an
error rather than silently picking one.

Writes a checkpoint every `checkpoint_every` steps (default 500, see `TrainConfig` in
`src/common/types.py`) and always at `max_steps`, to
`../outputs/checkpoints-finetuning/<model>/<run_name>-step<NNNNNN>/` (step number zero-padded
to 6 digits) -- each new save deletes the previous one, so only the latest checkpoint exists on
disk at a time rather than accumulating one per save. Feed that path straight into image
generation:

```bash
cd ../ccig-image-generation
python -m src.run --model sd3.5-large --checkpoint ../outputs/checkpoints-finetuning/sd3.5-large/run1-step001000
```

## Eval loss (are you actually learning, or just memorizing the batch?)

All four configs set `eval_dataset_path` to `data/clevr-dataset/finetuning-data/eval_holdout.json` -- a fixed,
class-balanced set of 198 images never used in any training batch (see
`scripts/build_eval_holdout.py`). Every `eval_every` steps (default 50; also always at the final
step), training pauses briefly and computes the average loss over the *entire* eval set, using
the exact same loss formula as training (see `DiffusersLoraTrainer._training_step` in each
model's file) but under `torch.no_grad()` with no optimizer step -- it doesn't affect training,
it's a read-only measurement.

```
[flux.1-dev] step 50/1921 eval_loss=0.8123 (n=198)
```

Training loss tells you how well the model fits the batch it's currently seeing; eval loss tells
you whether that's translating into learning the general mapping from constraint text to CLEVR
scenes, versus just memorizing this batch's specific images. If eval loss trends down alongside
training loss, that's real learning; if training loss keeps falling while eval loss plateaus or
rises, that's overfitting.

Set `eval_dataset_path: null` in a config (or omit `--eval-dataset` and don't set it) to disable
eval entirely. `--eval-dataset`/`--eval-every` on the CLI override the config's values.

**Important for the incremental-batches workflow below**: `eval_holdout.json` was built by
excluding every image already used across `batch_001`-`batch_010`, so it stays valid across the
*entire* remaining sequence (batch2, batch3, ...) as one fixed, comparable yardstick -- do not
build additional batches from data that overlaps with it, or the comparison stops being fair.

## Incremental training on small, (class, variant)-balanced batches

Instead of one long run over the whole dataset, you can train in small increments -- fine-tune
on ~2000 instances, evaluate, and only pull in another batch if you need to, continuing from
the previous batch's checkpoint each time.

**1. Cut the dataset into balanced batches** (one-time; skip if `data/clevr-dataset/finetuning-data/batches/`
already exists):

```bash
cd ccig-finetuning
python scripts/build_sequential_batches.py --batch-size 2000
```

Reads `finetune_prompts_clevr_train_filtered.json` and writes `batch_001.json`, `batch_002.json`,
... to `data/clevr-dataset/finetuning-data/batches/`. Each batch targets the *same* number of instances from
every **(class, variant) pair** (61 pairs total -- e.g. `C1/1prop`, `C8/2prop_exact_neg` -- not
just the same count per class), so every variant is guaranteed representation, not just every
class. A pair that doesn't have enough images left simply contributes whatever it has (never
more than the target); once exhausted it contributes nothing to later batches. No image is ever
reused for the same (class, variant) pair across batches.

The direct trade-off: since classes have very different numbers of variants (2 for C4/C6, up to
18 for C8) and per-variant availability ranges from 1 image to 21,083, per-class totals and
overall batch sizes are *not* equal across classes or batches -- e.g. batch 1 has 352 C1
instances vs. 64 C4 instances (both classes fully represented across all their variants), and
batch sizes range ~1,888-1,921 out of a requested 2000, since two `C2` variants (`3prop`,
`4prop`) only had 32 and 1 image available respectively and ran out immediately. The script
prints a full report of which pairs fell short in each batch, and what's left unused afterward.

**2. Train on batch 1** (fresh LoRA weights, `--dataset`/`--run-name` override the config's
defaults so you don't need a separate YAML per batch; `--max-steps` doesn't need to be given at
all -- it defaults to one epoch over whichever batch file you pass, regardless of that batch's
actual size):

```bash
python -m src.run --model flux.1-dev --config configs/flux.1-dev.yaml \
  --dataset ../data/clevr-dataset/finetuning-data/batches/batch_001.json \
  --run-name batch1
```

**3. Evaluate the result** (checkpoint path printed at the end of the run, e.g.
`../outputs/checkpoints-finetuning/flux.1-dev/batch1-step001921`, via
`ccig-image-generation --checkpoint <that path>`). If it's good enough, stop. If you need more
data, continue training on batch 2 *from that checkpoint* with `--init-ckpt`, rather than
starting over:

```bash
python -m src.run --model flux.1-dev --config configs/flux.1-dev.yaml \
  --dataset ../data/clevr-dataset/finetuning-data/batches/batch_002.json \
  --run-name batch2 \
  --init-ckpt ../outputs/checkpoints-finetuning/flux.1-dev/batch1-step001921
```

`--init-ckpt` loads the previous run's saved LoRA adapter (weights *and* its rank/alpha/target
modules) onto the base model instead of initializing new LoRA weights -- `lora_rank`/`lora_alpha`
in the config are ignored when it's set. Repeat with `batch_003.json`, `--init-ckpt` pointing at
whatever checkpoint path `batch2`'s run printed, and so on, for as many batches as you need.

## Adding a model

- **Diffusers-based**: add a class in `src/models/<model>.py` subclassing
  `DiffusersLoraTrainer` (set `pipeline_cls`, `hf_repo`), register it in `src/registry.py`,
  add a `configs/<model>.yaml`.
- **Non-diffusers**: subclass `LoraTrainer` directly and implement `train()` with the model's
  own training code.
