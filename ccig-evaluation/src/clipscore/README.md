# clipscore

CLIPScore: cosine similarity between a generated image's CLIP embedding and its
prompt's CLIP text embedding, using a CLIP checkpoint pretrained on the target domain
(`--clip-checkpoint`, a local path or HF repo id -- one checkpoint for CLEVR, a
different one for COCO; this method does not ship a checkpoint of its own).

Score range is `[-1, 1]` (raw cosine similarity, not rescaled) -- the CLIP literature's
"CLIPScore" popular rescaling (`2.5 * max(cos, 0)`) is intentionally not applied here so
the raw signal is preserved for downstream analysis; apply any rescaling downstream if
needed.

Output: `clipscore/results.json` — `{method, domain, checkpoint, results: [{id,
prompt_field, prompt, image_path, clipscore, success, error}]}`.

## Retraining the CLIP checkpoint (`clip_clevr_pretraining.py`)

Fine-tunes `zer0int/LongCLIP-GmP-ViT-L-14` on CLEVR image/prompt pairs to produce the
domain-pretrained checkpoint that `--clip-checkpoint` above expects. Run from this
directory (`ccig-evaluation/src/clipscore/`), matching `clipscore_run.sh`'s SBATCH
`-D` -- all defaults below are relative to that directory, three levels up from the
repo root:

```bash
cd ccig-evaluation/src/clipscore
python clip_clevr_pretraining.py
# or override any of the defaults:
python clip_clevr_pretraining.py --image_dir ... --clip_train ... --clip_val ... --checkpoint ...
```

| flag | meaning | default |
|---|---|---|
| `--image_dir` | folder of CLEVR train images | `../../../data/clevr-dataset/images` |
| `--clip_train` | training pairs json (`clip_train_clevr_20k.json`) | `../../../data/clevr-dataset/retraining-data/clip_train_clevr_20k.json` |
| `--clip_val` | validation pairs json (`clip_val_clevr_20k.json`) | `../../../data/clevr-dataset/retraining-data/clip_val_clevr_20k.json` |
| `--checkpoint` | folder checkpoints are saved under | `../../../outputs/checkpoints-retraining` |

`--image_dir`/`--clip_train`/`--clip_val` are training data that must already exist --
the script fails fast with a clear error naming the missing flag if one isn't found,
rather than creating an empty placeholder. `--checkpoint`, on the other hand, is an
output dir and is created automatically (along with any missing parents) if it
doesn't exist yet.

Whenever validation loss improves, the current model is saved to a new subfolder
under `--checkpoint`, named after the model and how far training had gotten, e.g.:

```
outputs/checkpoints-retraining/longclip-clevr-epoch01-step050000/
```

Only the latest improved checkpoint is kept on disk -- the previous one is deleted
each time a new one is saved, the same single-checkpoint-at-a-time convention
`ccig-finetuning` uses, rather than accumulating one folder per improvement.
