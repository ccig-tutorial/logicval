from __future__ import annotations

from pathlib import Path

from ..common.io import write_json

from ..common.types import ClipScoreResult, MatchedItem

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMMON_DIR = _REPO_ROOT / "ccig-image-generation" / "src" / "common"
import sys
sys.path.insert(0, str(_COMMON_DIR))

from scene_setup import scene_setup_text, scene_unsat_text
import json

def run_clipscore(items: list[MatchedItem], domain: str, checkpoint: str, out_path: str | Path, manifest:str|Path, is_closed_model:bool) -> None:
    """Compute CLIPScore (CLIP image-text embedding cosine similarity) for each
    generated image against the prompt that generated it.

    checkpoint: path (or HF repo id) to a CLIP model pretrained on `domain` --
    one checkpoint for clevr, a different one for coco, passed explicitly by the
    caller since neither is a general-purpose CLIP checkpoint.
    """
    # Imported lazily so selecting `--method vlm-judge` / `--method perception` doesn't
    # force-load torch/transformers when clipscore isn't requested.
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    
    from transformers import AutoProcessor, AutoModel

    #checkpoint "openai/clip-vit-large-patch14"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: list[ClipScoreResult] = []
    #find error in generated from manifest - no image generated cases
    with open(manifest, "r") as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    for item in manifest:
        if item["error"] is not None:
            print('No image generated:', item['id'])
            results.append(
                    ClipScoreResult(
                    clipmodel=checkpoint,
                    id=item['id'],
                    prompt_field=item['prompt_field'],
                    prompt=item['prompt'],
                    image_path=str(item['image_path']),
                    clipscore=0,
                    cosine=-1,
                    success=False,
                    error='No image generated',))
    
    model = CLIPModel.from_pretrained(checkpoint).to(device).eval()
    processor = CLIPProcessor.from_pretrained(checkpoint)
     
    
    len_77 = 0
    for item in items:
        try:
            image = Image.open(item.image_path).convert("RGB")
            # to study 
            setup_text = scene_setup_text(item.record.number_of_objects, item.record.domain, with_background=is_closed_model)
            unsat_text = scene_unsat_text(item.record.domain, with_unsat=is_closed_model)
            text = "A photo depicts an image where: " + setup_text+ item.prompt_text + unsat_text
            inputs = processor(text=["A photo depicts an image where: " + text], images=image, return_tensors="pt", truncation= True, max_length=77, padding=True) #
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            
            full = processor.tokenizer(text, truncation=False, return_tensors="pt",)
            truncated = processor.tokenizer(text, truncation=True, max_length=77, return_tensors="pt",) #
            
            if full.input_ids.shape[1] > 77:
                len_77 = len_77+1
                print("Len greater - hence truncated item:", item.id, len_77)
               
            
            with torch.no_grad():
                out = model(**inputs)
            image_embed = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            text_embed = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            #768d embeddings
            cosine = (image_embed @ text_embed.T).item()
            clipscore = 2.5 * max(cosine, 0.0)
            results.append(
                ClipScoreResult(
                    clipmodel=checkpoint,
                    id=item.id,
                    prompt_field=item.prompt_field,
                    prompt=item.prompt_text,
                    image_path=str(item.image_path),
                    clipscore=clipscore,
                    cosine=cosine,
                    success=True,
                    error=None,
                )
            )
            print(f"[ok]   {item.id}: {clipscore:.4f}")
        except Exception as e:
            results.append(
                ClipScoreResult(
                    clipmodel = checkpoint,
                    id=item.id,
                    prompt_field=item.prompt_field,
                    prompt=item.prompt_text,
                    image_path=str(item.image_path),
                    clipscore=None,
                    cosine=None,
                    success=False,
                    error=repr(e),
                )
            )
            print(f"[fail] {item.id}: {e}")
        

    write_json(
        out_path,
        {
            "method": "clipscore",
            "domain": domain,
            "checkpoint": checkpoint,
            "results": [r.to_json() for r in results],
        },
    )
