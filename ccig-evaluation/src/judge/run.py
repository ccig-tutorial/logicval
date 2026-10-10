from __future__ import annotations

from pathlib import Path

from ..common.io import write_json
from ..common.types import JudgeResult, MatchedItem
from .base import VLMJudge
_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMMON_DIR = _REPO_ROOT / "ccig-image-generation" / "src" / "common"
import sys
sys.path.insert(0, str(_COMMON_DIR))

from scene_setup import scene_setup_text, scene_unsat_text
import json


def run_judge(items: list[MatchedItem], judge: VLMJudge, out_path: str | Path, manifest:str|Path, is_closed_model:bool) -> None:
    from PIL import Image

    results: list[JudgeResult] = []
    with open(manifest, "r") as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    for item in manifest:
        if item["error"] is not None:
            print('No image generated:', item['id'])
            results.append(
                    JudgeResult(
                    id=item['id'],
                    prompt_field=item['prompt_field'],
                    prompt=item['prompt'],
                    image_path=str(item['image_path']),
                    score=0,
                    raw_score=0,
                    rationale='No image generated',
                    success=False,
                    error='No image generated',
                ))
    
    for item in items:
        try:
            image = Image.open(item.image_path).convert("RGB")
            # to study 
            setup_text = scene_setup_text(item.record.number_of_objects, item.record.domain, with_background=is_closed_model)
            unsat_text = scene_unsat_text(item.record.domain, with_unsat=is_closed_model)
            text = setup_text+ item.prompt_text + unsat_text
            print(text)
            verdict = judge.judge(image, text)
            results.append(
                JudgeResult(
                    id=item.id,
                    prompt_field=item.prompt_field,
                    prompt=item.prompt_text,
                    image_path=str(item.image_path),
                    score=verdict.score,
                    raw_score=verdict.raw_score,
                    rationale=verdict.rationale,
                    success=True,
                    error=None,
                )
            )
            print(f"[ok]   {item.id}: {verdict.score:.2f}")
        except Exception as e:
            results.append(
                JudgeResult(
                    id=item.id,
                    prompt_field=item.prompt_field,
                    prompt=item.prompt_text,
                    image_path=str(item.image_path),
                    score=None,
                    raw_score=None,
                    rationale=None,
                    success=False,
                    error=repr(e),
                )
            )
            print(f"[fail] {item.id}: {e}")

    write_json(
        out_path,
        {
            "method": "vlm-judge",
            "backend": judge.name,
            "results": [r.to_json() for r in results],
        },
    )
