from __future__ import annotations

import argparse
from pathlib import Path

from src.common.io import match_images_to_prompts
from src.judge.registry import JUDGE_REGISTRY
from src.perception.attributes.registry import ATTRIBUTE_REGISTRY
from src.perception.detectors.registry import DETECTOR_REGISTRY
from src.soft_tifa.registry import VQA_REGISTRY

#Example - to get clipscore:
'''
python -m src.run --images-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT --prompts-file /users/sbsh670/data/ccig_evalData/clevr_1_scenes_SAT.jsonl --method clipscore --domain clevr --out-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT --clip-checkpoint openai/clip-vit-large-patch14 
'''
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated images against the CCIG constraints/prompts that produced them."
    )
    parser.add_argument("--images-dir",  help="Folder of generated images from one model")
    parser.add_argument("--prompts-file", required=True, help="Path to ccig_eval_dataset_{SAT,UNSAT}.jsonl")
    parser.add_argument(
        "--method", nargs="+", required = True, choices=["clipscore", "vlm-judge", "perception", "soft-tifa", "human-eval", "analysis"]
    )
    parser.add_argument("--domain", required=True, choices=["clevr", "coco"])
    parser.add_argument("--out-dir", default=None, help="Default: outputs/<images-dir-name>/")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifest", default=None, help="path to manifest file from generation")
    parser.add_argument("--is_closed_model",action="store_true",)

    # clipscore
    parser.add_argument("--clip-checkpoint", help="CLIP checkpoint path/repo pretrained on --domain")

    # vlm-judge
    parser.add_argument("--judge-backend", default="gpt-4o", choices=list(JUDGE_REGISTRY))

    # soft-tifa
    parser.add_argument("--vqa-backend", default="gpt-4o", choices=list(VQA_REGISTRY))
    parser.add_argument("--sat",action="store_true",)

    # perception
    parser.add_argument("--detector", default="grounding-dino", choices=list(DETECTOR_REGISTRY))
    parser.add_argument("--attribute-classifier", default="clip-zero-shot", choices=list(ATTRIBUTE_REGISTRY))
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu'; default: auto-detect")
    
    #human_eval
    parser.add_argument("--annotation_file", default=None, help="path to human annotated file")
    
    #analysis
    parser.add_argument("--vlm_judge_results", default=None, help="path to vlm_judge evaluation json")
    parser.add_argument("--perception_results", default=None, help="path to perception evaluation json")
    parser.add_argument("--soft_tifa_results", default=None, help="path to soft-tifa evaluation json")
    parser.add_argument("--clipscore_results", default=None, help="path to clipscore evaluation json")
    parser.add_argument("--human_results", default=None, help="path to human evaluation json")
    parser.add_argument("--analysis_out", default=None, help="path where analysis outputs have to be saved")
    parser.add_argument("--combo", type = int, default=0, help="whether combination ds or not")
    

    
    args = parser.parse_args()
    print('args:', args.combo)
    items = match_images_to_prompts(args.images_dir, args.prompts_file)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        parser.error("No images matched to prompt records -- nothing to evaluate.")

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / Path(args.images_dir).name
    
    if "clipscore" in args.method:
        if not args.clip_checkpoint:
            parser.error("--clip-checkpoint is required for --method clipscore")
        #from src.clipscore.run import run_clipscore
        from src.clipscore.run_L import run_clipscore

        #run_clipscore(items, args.domain, args.clip_checkpoint, out_dir / "clipscore" / "results-gen.json", args.manifest, args.is_closed_model)
        run_clipscore(items, args.domain, args.clip_checkpoint, out_dir / "clipscore" / "results-L-gen.json", args.manifest, args.is_closed_model)


    if "vlm-judge" in args.method:
        from src.judge.registry import build_judge
        from src.judge.run import run_judge

        run_judge(items, build_judge(args.judge_backend, device=args.device), out_dir / "vlm_judge" / "results-gen.json", args.manifest, args.is_closed_model,)

    if "perception" in args.method:
        from src.perception.run import run_perception

        run_perception(
            items,
            args.domain,
            args.detector,
            args.attribute_classifier,
            args.device,
            out_dir / "perception" / "results-gen.json", args.manifest,
        )

    if "soft-tifa" in args.method:
        from src.common.dataset_gen import load_domain
        from src.soft_tifa.registry import build_vqa_backend
        from src.soft_tifa.run import run_soft_tifa

        run_soft_tifa(
            items,
            load_domain(args.domain),
            build_vqa_backend(args.vqa_backend, device=args.device),
            out_dir / "soft_tifa" / "results-gen.json", args.manifest, args.is_closed_model, args.sat,
        )
    if "human-eval" in args.method:
        from src.common.dataset_gen import load_domain
        from src.human_evaluation.run import run_human_eval

        run_human_eval(
            items,
            args.domain,
            args.annotation_file,
            out_dir / "human_eval" / "results-gen.json", args.manifest,
        )
    if "analysis" in args.method and args.combo == 0:
        from src.common.dataset_gen import load_domain
        from src.analysis_plots import run_analysis

        run_analysis(
            args.prompts_file,
            args.domain,
            args.clipscore_results,
            args.vlm_judge_results,
            args.soft_tifa_results,
            args.perception_results,
            args.human_results,
            Path(args.analysis_out) /"analysis.json", 
        )
    
    if "analysis" in args.method and args.combo == 1:
        from src.common.dataset_gen import load_domain
        from src.analysis_combo import run_analysis

        run_analysis(
            args.prompts_file,
            args.domain,
            args.clipscore_results,
            args.vlm_judge_results,
            args.soft_tifa_results,
            args.perception_results,
            args.human_results,
            Path(args.analysis_out) /"analysis.json", 
        )


if __name__ == "__main__":
    main()
