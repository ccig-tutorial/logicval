'''Command: 
python clip_clevr_pretraining.py --image_dir /users/sbsh670/data/clevr/CLEVR_v1.0/image_generative_model_training/images/train --clip_train /users/sbsh670/data/ccig_finetune/clip_train_clevr_20k.json --clip_val /users/sbsh670/data/ccig_finetune/clip_val_clevr_20k.json --checkpoint /users/sbsh670/data/ccig-models/clip-clevr
--image_dir : path to folder that has the Clevr train images
--clip_train: path to json file - clip_train_clevr_20k
--clip_val: path to json file - clip_val_clevr_20k
--checkpoint: path to folder where the checkpoint model and procesor saved
'''  
import os

# Must be set before any tokenizer runs, and before DataLoader workers fork (now
# that num_workers>0 below) -- HF's Rust-based fast tokenizers otherwise warn
# ("process just got forked, after parallelism has already been used") and can
# deadlock when a forked worker touches a tokenizer that already tokenized
# something in the parent process.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import shutil
import json, argparse
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
import torch
from transformers import CLIPModel
from tqdm import tqdm
from torch.utils.data import Subset

from transformers import CLIPTokenizer
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from collections import defaultdict
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device:', device)
model_name = "zer0int/LongCLIP-GmP-ViT-L-14" #openai/clip-vit-base-patch32

tokenizer = AutoTokenizer.from_pretrained(model_name)

#print('Max len tokenizer:', tokenizer.model_max_length)#77 for normal clip, 248 for long clip

#json_file = '/users/sbsh670/data/ccig_finetune/clip_pretraining_clevr_train_20k.json'
#image_dir = '/users/sbsh670/data/clevr/CLEVR_v1.0/image_generative_model_training/images/train'

#clip_train = '/users/sbsh670/data/ccig_finetune/clip_train_clevr_20k.json'
#clip_val = '/users/sbsh670/data/ccig_finetune/clip_val_clevr_20k.json'

#save_path= '/users/sbsh670/data/ccig-models/clip-clevr'
#save_model = save_path+'/longclip_clevr_best'

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

'''
with open(json_file) as f:
    data = json.load(f)

# group all prompts by image
image_to_samples = defaultdict(list)

for sample in data:
    image_to_samples[sample["image_id"]].append(sample)

image_ids = list(image_to_samples.keys())

random.seed(42)
random.shuffle(image_ids)

split_ratio = 0.9

split_idx = int(len(image_ids) * split_ratio)

train_ids = set(image_ids[:split_idx])
val_ids = set(image_ids[split_idx:])

train_samples = []
val_samples = []

for image_id, samples in image_to_samples.items():

    if image_id in train_ids:
        train_samples.extend(samples)
    else:
        val_samples.extend(samples)


print("Train images:", len(train_ids))
print("Val images:", len(val_ids))

print("Train samples:", len(train_samples))
print("Val samples:", len(val_samples))

with open(clip_train,"w") as f:
    json.dump(train_samples,f)

with open(clip_val,"w") as f:
    json.dump(val_samples,f)


with open(json_file) as f:
    data = json.load(f)

lengths = []

for sample in tqdm(data):
    tokens = tokenizer(
        sample["prompt"],
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    lengths.append(len(tokens))

print("Maximum tokens:", max(lengths)) #94
print("Average tokens:", sum(lengths)/len(lengths)) #36.55

'''

def evaluate(model, loader, device):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for batch in loader:

            batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
            }

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                outputs = model(
                    **batch,
                    return_loss=True
                )

            total_loss += outputs.loss.item()


    return total_loss / len(loader)


class CLEVRClipDataset(Dataset):
    def __init__(self, json_file, image_dir):

        with open(json_file, "r") as f:
            self.samples = json.load(f)

        self.image_dir = image_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image_path = os.path.join(
            self.image_dir,
            sample["image_filename"]
        )

        image = Image.open(image_path).convert("RGB")

        text = sample["prompt"]

        return image, text, sample["image_id"], sample["n_objects"]

from collections import defaultdict
import random
from torch.utils.data import BatchSampler, Sampler


class NObjectsBatchSampler(Sampler):

    def __init__(self, dataset, batch_size):

        self.batch_size = batch_size

        self.groups = defaultdict(list)

        for idx, sample in enumerate(dataset.samples):
            self.groups[sample["n_objects"]].append(idx)

        self.n_values = list(self.groups.keys())


    def __iter__(self):

        pools = {
            k: random.sample(v, len(v))
            for k, v in self.groups.items()
        }

        while True:

            batch = []

            # cycle through different n_objects
            while len(batch) < self.batch_size:

                random.shuffle(self.n_values)

                added = False

                for n in self.n_values:

                    if pools[n]:

                        batch.append(
                            pools[n].pop()
                        )

                        added = True

                    if len(batch) == self.batch_size:
                        break

                if not added:
                    break

            if len(batch) == 0:
                break

            yield batch


    def __len__(self):

        total = sum(
            len(v)
            for v in self.groups.values()
        )

        return total // self.batch_size

def collate_fn(batch):

    images = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    image_ids = [b[2] for b in batch]
    n_objects = [b[3] for b in batch]
    
    return processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=248
    )

def main(args):
    image_dir = args.image_dir
    clip_train = args.clip_train
    clip_val = args.clip_val
    save_path = args.checkpoint

    # image_dir/clip_train/clip_val are training data that must already exist --
    # fail fast with a clear message instead of an opaque error deep inside the
    # dataset/dataloader if a path is wrong.
    for path, label in [(image_dir, "--image_dir"), (clip_train, "--clip_train"), (clip_val, "--clip_val")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # --checkpoint is an output dir -- create it (and any missing parents) if this
    # is the first run, same as ccig-finetuning's checkpoint saving.
    os.makedirs(save_path, exist_ok=True)

    train_dataset = CLEVRClipDataset(json_file=clip_train, image_dir=image_dir)

    val_dataset = CLEVRClipDataset(json_file=clip_val,image_dir=image_dir)
    val_indices = random.sample(range(len(val_dataset)),10000)
    val_subset = Subset(val_dataset, val_indices)

    sampler = NObjectsBatchSampler(train_dataset, 8)

    # num_workers/pin_memory/persistent_workers: the original single-process loader
    # (num_workers=0) left the GPU waiting on CPU-side image decode + tokenization
    # between every batch (~36% GPU util at batch_size=8) -- this pipeline is CPU-
    # bound, not compute-bound, so parallelizing data loading (not raising batch
    # size) is what actually speeds it up. persistent_workers avoids respawning the
    # worker pool every epoch; prefetch_factor keeps a few batches queued ahead so
    # workers stay busy while the GPU is training on the current one.
    NUM_WORKERS = min(16, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(),lr=5e-6, weight_decay=0.01)

    epochs = 1
    step = 0
    validate_every = 50000

    best_val_loss = float("inf")
    last_ckpt_dir = None

    for epoch in range(epochs):

        model.train()
        running_loss = 0
        for batch in tqdm(train_loader):

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # bf16 autocast: Ada (RTX 4090) tensor cores run bf16 matmuls much faster
            # than fp32, and unlike fp16 it doesn't need a GradScaler (same dynamic
            # range as fp32, just less mantissa precision) -- free speedup, batch
            # size / training dynamics otherwise unchanged.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                outputs = model(**batch, return_loss=True)
                loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step += 1
            if step % validate_every == 0:

                train_loss = running_loss / validate_every
                val_loss = evaluate(model, val_loader, device)

                print(
                    f"""
                    Step: {step}
                    Train loss: {train_loss:.4f}
                    Validation loss: {val_loss:.4f}
                    """
                )


                if val_loss < best_val_loss:

                    best_val_loss = val_loss

                    # Name the checkpoint after the model + how far training got
                    # (epoch/step), e.g. longclip-clevr-epoch01-step050000/ -- and
                    # keep only the latest best on disk (delete the previous one)
                    # rather than accumulating one per improvement.
                    ckpt_name = f"longclip-clevr-epoch{epoch + 1:02d}-step{step:06d}"
                    ckpt_dir = os.path.join(save_path, ckpt_name)
                    os.makedirs(ckpt_dir, exist_ok=True)

                    model.save_pretrained(ckpt_dir)

                    processor.save_pretrained(ckpt_dir)

                    if last_ckpt_dir is not None and last_ckpt_dir != ckpt_dir:
                        shutil.rmtree(last_ckpt_dir, ignore_errors=True)
                    last_ckpt_dir = ckpt_dir

                    print(f"Saved best model -> {ckpt_dir}")


                running_loss = 0
                model.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Process image and CLIP feature paths.")

    parser.add_argument("--image_dir", type=str, default="../../../data/clevr-dataset/images")
    parser.add_argument(
        "--clip_train", type=str, default="../../../data/clevr-dataset/retraining-data/clip_train_clevr_20k.json"
    )
    parser.add_argument(
        "--clip_val", type=str, default="../../../data/clevr-dataset/retraining-data/clip_val_clevr_20k.json"
    )
    parser.add_argument("--checkpoint", type=str, default="../../../outputs/checkpoints-retraining")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


