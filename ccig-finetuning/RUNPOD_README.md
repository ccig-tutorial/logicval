# Running ccig-finetuning on RunPod

RunPod-specific commands for this module. The general-purpose sync/tmux tooling lives in the
standalone [`runpod/`](../runpod/README.md) package at the repo root -- this file just shows the
exact sequence for `ccig-finetuning`. See `README.md` in this folder for what the training code
itself does.

All commands below are run from the repo root, using the project name `finetuning` (see
`runpod/projects/finetuning.env`).

## 0) One-time: connection config

```bash
cp runpod/runpod.env.example runpod/runpod.env   # once
# edit runpod/runpod.env with your pod's host/port/key
```

## 1) Sync code + data to the pod

The training data (`data/clevr-dataset/...`) is a separate project mapping from the code, so
sync both the first time:

```bash
./runpod/scripts/sync_to_runpod.sh --project data          # dataset + images (large, first time only)
./runpod/scripts/sync_to_runpod.sh --project finetuning    # this module's code
```

After that first full copy, push just your code edits:

```bash
./runpod/scripts/sync_changed_to_runpod.sh --project finetuning
```

## 2) Bootstrap the remote Python environment

```bash
./runpod/scripts/bootstrap_remote.sh --project finetuning
```

Creates `.venv` under `/workspace/CCIG_Eval/ccig-finetuning` and installs `requirements.txt`.

## 3) Log in to Hugging Face (gated models)

`flux.1-dev` and `sd3.5-large` are gated repos -- accept each model's license on its Hub page
(logged in) once, then log in on the pod non-interactively with a token
(https://huggingface.co/settings/tokens).

**Must set the same `HF_HOME` here as in every training/inference command below.** Logging in
without it saves the token under the default `~/.cache/huggingface/token` (the tiny root disk),
not under `.hf_cache` on `/workspace` -- training then can't find it and fails with
`GatedRepoError`, even though you're "logged in":

```bash
./runpod/scripts/ssh_remote.sh --project finetuning \
  "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache && hf auth login --token <your_hf_token>"
```

## 4) Start training in tmux

tmux runs on the pod itself, not on your laptop -- once started, the job keeps running even if
you close your laptop, lose your connection, or the SSH session drops. `start_tmux.sh` creates
the session, redirects output to `logs/<session>.log`, and returns immediately.

**Always export `HF_HOME`/`HUGGINGFACE_HUB_CACHE` pointing at `/workspace`, as below.** RunPod
containers commonly ship a tiny root disk (e.g. 5-20 GB) separate from the large `/workspace`
volume. Hugging Face's download cache defaults to `~/.cache/huggingface` on that small root
disk -- a single gated model like FLUX.1-dev is ~24 GB total, so downloading it without this
export will fill the root disk and fail with a disk-space or "xet" download error partway
through.

```bash
./runpod/scripts/start_tmux.sh --project finetuning flux_dev_stage1 \
  "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache/hub && python -m src.run --model flux.1-dev --config configs/flux.1-dev.yaml --run-name run1"
```

If a previous attempt already filled the root disk, clear the stale cache first:

```bash
./runpod/scripts/ssh_remote.sh --project finetuning "rm -rf /root/.cache/huggingface"
```

Detach any time with `Ctrl+b` then `d` -- this only detaches your terminal, it does not stop the
job.

### Incremental training on small, balanced batches

See `README.md`'s "Incremental training on small, (class, variant)-balanced batches" section for
what this does and why. The commands there are written generically (assumes an already-activated
environment); on RunPod, wrap them in `start_tmux.sh` with the same `HF_HOME` export as above so
they persist and don't hit the disk-space issue:

```bash
# batch 1 -- fresh LoRA weights, --max-steps omitted (defaults to one epoch over batch_001.json)
./runpod/scripts/start_tmux.sh --project finetuning batch1 \
  "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache/hub && python -m src.run --model flux.1-dev --config configs/flux.1-dev.yaml --dataset ../data/clevr-dataset/finetuning-data/batches/batch_001.json --run-name batch1"
```

After evaluating batch1's checkpoint (path printed at the end of the run, e.g.
`../outputs/checkpoints-finetuning/flux.1-dev/batch1-step001921`), continue on batch 2 from it
with `--init-ckpt`:

```bash
./runpod/scripts/start_tmux.sh --project finetuning batch2 \
  "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache/hub && python -m src.run --model flux.1-dev --config configs/flux.1-dev.yaml --dataset ../data/clevr-dataset/finetuning-data/batches/batch_002.json --run-name batch2 --init-ckpt ../outputs/checkpoints-finetuning/flux.1-dev/batch1-step001921"
```

Repeat with `batch_003.json`, `--init-ckpt` pointing at whatever checkpoint `batch2` printed, and
so on.

**Note:** all four configs set `eval_dataset_path` by default, so you'll see extra
`eval_loss=... (n=198)` lines in the log every 50 steps -- see `README.md`'s "Eval loss" section
for what that measures and why it matters. To train more than one epoch over a batch, use
`--epochs N` instead of computing a step count by hand (e.g. add `--epochs 2` to either command
above); set at most one of `--max-steps`/`--epochs`.

## 5) Monitor progress

```bash
./runpod/scripts/list_tmux.sh --project finetuning                          # is it still running?
./runpod/scripts/tail_log.sh --project finetuning flux_dev_stage1           # live log stream
./runpod/scripts/attach_tmux.sh --project finetuning flux_dev_stage1        # reattach to the session
```

### Stopping a run

```bash
./runpod/scripts/stop_tmux.sh --project finetuning flux_dev_stage1
```

Kills the tmux session immediately. Checkpoints save every `checkpoint_every` steps (default
500) and always at the final step (see `src/common/types.py` / `_diffusers_common.py`), so at
most you lose whatever was trained since the last periodic save, not the whole run.

## 6) Pull checkpoints back -- do this live, not just at the end

**Don't wait until training finishes to pull results back.** If the pod or its Network Volume
is lost in between (wrong volume attached to a later pod, pod deleted, etc.) -- which has
actually happened in this project -- anything not already copied out locally is gone for good,
even though it was "saved" on the pod. Run this in a separate terminal any time after starting
training; it automatically pulls a local copy every time the log reports a new checkpoint, so a
local copy exists within moments of it being written, not only if you remember to fetch it after
the run completes:

```bash
./runpod/scripts/watch_and_pull_checkpoints.sh --project finetuning \
  --folder ../outputs/checkpoints-finetuning flux_dev_stage1
```

Ctrl+C stops watching without stopping training; it exits on its own once the run finishes or
crashes. You can also pull manually at any point:

```bash
./runpod/scripts/sync_folder_from_runpod.sh --project finetuning --folder ../outputs/checkpoints-finetuning
```

Checkpoints write to the repo-root `outputs/checkpoints-finetuning/` (sibling to
`ccig-finetuning/`, shared across pipelines), not `ccig-finetuning/outputs/` -- the explicit
`--folder ../outputs/checkpoints-finetuning` above is what points the pull at that location; it
assumes the remote layout mirrors the local repo (`RUNPOD_REMOTE_ROOT` plays the role of
`REPO_ROOT`), which matches how `bootstrap_remote.sh`/`sync_to_runpod.sh` lay the pod out.

## Other models

Same steps, just change `--model`/`--config`/tmux session name:

```bash
./runpod/scripts/start_tmux.sh --project finetuning sd35_large_stage1 \
  "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache/hub && python -m src.run --model sd3.5-large --config configs/sd3.5-large.yaml --run-name run1"

./runpod/scripts/start_tmux.sh --project finetuning qwen_image_stage1 \
  "source .venv/bin/activate && export HF_HOME=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache && export HUGGINGFACE_HUB_CACHE=/workspace/CCIG_Eval/ccig-finetuning/.hf_cache/hub && python -m src.run --model qwen-image --config configs/qwen-image.yaml --run-name run1"
```

## Troubleshooting

See the [`runpod/README.md`](../runpod/README.md#troubleshooting) troubleshooting section for
connection issues (refused connections, host-key errors, missing `.venv`, etc).
