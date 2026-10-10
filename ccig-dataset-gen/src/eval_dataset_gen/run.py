#!/usr/bin/env python3
"""
CCIG Dataset Generation Pipeline
=================================
Reads ASP constraint template files, instantiates them with random valid
placeholder assignments, solves the resulting ASP programs with clingo,
verbalizes each constraint in short/medium/long natural language, and writes
records to two JSONL files — one for SAT, one for UNSAT.

Usage
-----
Run as a module from clevr-ccig-dataset-gen/ (needed for the src.common / src.eval_dataset_gen
package imports to resolve):

# Generate 10 random instances per template (default):
    python -m src.eval_dataset_gen.run

# 20 samples per template, 6 scene objects, custom output:
    python -m src.eval_dataset_gen.run --samples 20 --n_objects 6 --output ccig_eval_dataset

# Only C1 and C3 templates:
    python -m src.eval_dataset_gen.run --classes C1 C3

# Dry run (verbalize without solving):
    python -m src.eval_dataset_gen.run --no_solve

# Exhaustive enumeration instead of random sampling:
    python -m src.eval_dataset_gen.run --mode exhaustive
"""

import argparse
import json
import sys
import random
import hashlib
from pathlib import Path
from collections import defaultdict
from .domain import background_asp
from .instantiate import load_template, sample_assignment, apply_assignment
from ..common.verbalize import verbalize
from .solve import solve


# ── Defaults ────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_DIR = Path(__file__).resolve().parent / "constraint_templates"
DEFAULT_OUTPUT = _REPO_ROOT / "data" / "ccig_eval_dataset"

#template files not valid for coco domain
not_coco = ['C1_4prop_mix_neg', 'C1_4prop', 'C2_4prop_neg', 'C2_4prop',
            'C3_1propA_3propC', 'C3_2propA_2propC', 'C3_2propA_neg_2propC',
            'C3_3propA_1propC', 'C3_3propA_1propC_neg']

# ── Record construction ──────────────────────────────────────────────────────

def build_record(
    template_path: Path,
    assignment: dict[str, str],
    rule_text: str,
    status: str,
    record_id: str,
    n_objects: int = 4,
    domain: str = "clevr",
) -> dict:
    stem = template_path.stem

    cls = stem.split("_")[0]

    family = stem.split("_", 1)[1] if "_" in stem else ""

    instantiated_rule = apply_assignment(rule_text, assignment)
    texts = verbalize(stem, assignment)
    # verbalize() returns {"short", "medium", "long", "subqa"} -- subqa is a
    # dict[str, str] (question -> expected answer), not a str, so it can't live
    # inside "prompts" (typed dict[str, str] by ccig-evaluation's PromptRecord).
    # It gets its own top-level field instead; see build_combo_record below.
    subqa = texts.pop("subqa", {})

    return {
        "id": record_id,
        "domain": domain,
        "complexity_class": cls,
        "constraint_family": family,
        "prompts": texts,
        "subqa": subqa,
        "instantiated_rule": instantiated_rule,
        "asp_template_file": template_path.name,
        "status": status,
        "number_of_objects": n_objects,
    }

def build_combo_record(
    cls: list[str],
    template_pth: list[Path],
    prompts: list[dict[str,str]],
    instantiated_rules: list[str],
    status: str,
    record_id: str,
    n_objects: int = 4,
    domain: str = "clevr",
    subqas: list[dict[str, str]] | None = None,
) -> dict:

    name_files = []
    for p in template_pth:
        name_files.append(p.name)

    families = []
    for p in template_pth:
        stem = p.stem
        family = stem.split("_", 1)[1] if "_" in stem else ""
        families.append(family)
    if not(isinstance(cls[0], int)):
        complexity_classes = cls
    else:
        complexity_classes = [f"C{c}" for c in cls]

    # Merge each constraint's subqa dict into one, prefixing questions with
    # "Constraint N: " when there's more than one constraint -- same convention
    # already used for the concatenated prompt text below -- so identical
    # questions from different constraints don't collide as duplicate keys.
    subqa: dict[str, str] = {}
    if subqas:
        multi = len(subqas) > 1
        for i, sq in enumerate(subqas, start=1):
            for question, answer in (sq or {}).items():
                key = f"Constraint {i}: {question}" if multi else question
                subqa[key] = answer

    return {
        "id": record_id,
        "domain": domain,
        "complexity_class": complexity_classes,
        "constraint_family": families,
        "prompts": prompts,
        "subqa": subqa,
        "instantiated_rule": instantiated_rules,
        "asp_template_file": name_files,
        "status": status,
        "number_of_objects": n_objects,
    }


def _record_id(template_name: str, assignment: dict, idx: int) -> str:
    content = json.dumps({"t": template_name, "a": assignment}, sort_keys=True)
    h = hashlib.md5(content.encode()).hexdigest()[:6]
    stem = template_name.replace(".txt", "").replace("_", "-")
    return f"{stem}-{h}-{idx:03d}"


# ── Main pipeline ────────────────────────────────────────────────────────────

def run(
    template_dir: Path,
    output_path: Path,
    n_samples: int,
    classes: list[str] | None,
    solve_programs: bool,
    n_models: int,
    time_limit: int,
    clingo_bin: str,
    seed: int,
    verbose: bool,
    domain: str = "clevr",
    combo: int = 1
) -> None:
    rng = random.Random(seed)

    
    template_files = sorted(
        p for p in template_dir.glob("*.txt")
        if "OLD" not in p.name and p.name != "README.txt"
    )


    if classes:
        template_files = [p for p in template_files
                          if any(p.name.startswith(c + "_") for c in classes)]

        if domain=='coco':
            template_files = [p for p in template_files if p.stem not in not_coco]
    
    if not template_files:
        print(f"No template files found in {template_dir}", file=sys.stderr)
        sys.exit(1)

    class_files = defaultdict(list)
    for p in template_files:
        class_num = int(p.name.split("_")[0][1:])  
        class_files[class_num].append(p)
    class_files = dict(class_files)
    output_dir = Path(output_path)  # directory path provided by user
    output_dir.mkdir(parents=True, exist_ok=True)

    sat_path = output_dir / f"{domain}_{combo}_scenes_SAT.jsonl"
    unsat_path = output_dir / f"{domain}_{combo}_scenes_UNSAT.jsonl"
    
    sat_count = unsat_count = total_count = 0
    
    max_attempts = 1000
    attempts = 0
    with open(sat_path, "w") as sat_f, open(unsat_path, "w") as unsat_f:
        #for combinations of templates 
        while(sat_count<n_samples and attempts<max_attempts):
                if combo==1:
                    break
                attempts  =attempts+1
                n_objects = random.randint(3, 9)
                classes = random.sample(range(1, 10), combo)
                selected_templates = []
                for c in classes:
                    file = random.choice(class_files[c])
                    selected_templates.append(file)
                
                
                asgns = []
                prompts = {'short': '', 'medium': '', 'long': ''}
                subqas = []
                instantiated_rules = []
                background = background_asp(domain, n_objects)
                full_program = background
                cons = 0
                for tpl_path in selected_templates:
                    cons = cons+1
                    stem = tpl_path.stem
                    _, rule_text = load_template(tpl_path)
                    asgn = sample_assignment(domain, rule_text, rng)
                    instantiated_rule = apply_assignment(domain, rule_text, asgn)
                    text = verbalize(stem, asgn)
                    prompts['short'] = prompts['short'] + ' Constraint ' + str(cons) + ':' + text['short'] + '\n'
                    prompts['medium'] = prompts['medium'] + ' Constraint ' + str(cons) + ':' + text['medium'] + '\n'
                    prompts['long'] = prompts['long'] + ' Constraint ' + str(cons) + ':' + text['long'] + '\n'
                    subqas.append(text.get('subqa', {}))
                    asgns.append(asgn)
                    instantiated_rules.append(instantiated_rule)
                    full_program = full_program + instantiated_rule + "\n"

                
                try:
                    status, raw_models = solve(
                            full_program,
                            n_models=n_models,
                            time_limit=time_limit,
                            clingo_bin=clingo_bin,
                            )
                except RuntimeError as e:
                    print(f"  Solver error: {e}", file=sys.stderr)
                    status = "ERROR"

                if status == "SAT":
                    #record_id = _record_id(tpl_path.name, asgn, tpl_sat_count)
                    sat_count += 1
                    record_id = sat_count
                    total_count+=1
                    #for r_m in raw_models:
                    #    scene = format_scene(r_m)
                    #    print(scene)


                elif status == "UNSAT":
                    #record_id = _record_id(tpl_path.name, asgn, tpl_unsat_count)
                    unsat_count += 1
                    record_id = unsat_count
                    total_count+=1

                
                record = build_combo_record(classes, selected_templates, prompts, instantiated_rules, status, record_id, n_objects, domain, subqas=subqas)
                line = json.dumps(record) + "\n"

                if status == "SAT":
                    sat_f.write(line)
                elif status == "UNSAT":
                    unsat_f.write(line)
        
        if (combo>1 and sat_count < n_samples):
                print('Tried max_attempts =2000 times but could produce:', sat_count, ' satisfiable and ', unsat_count, ' unsatisfiable prompts.')
        
        
        for tpl_path in template_files: #for each template of that class - generate 5 sat plus 2 unsat per file
            if combo>1: 
                break
            if verbose:
                print(f"Processing {tpl_path.name} …")

            _, rule_text = load_template(tpl_path)

            stem = tpl_path.stem
            cls = stem.split("_")[0]
            family = stem.split("_", 1)[1] if "_" in stem else ""

            history_asgn = []
            max_attempts = 2000 
            attempts = 0
            tpl_sat_count = 0
            tpl_unsat_count = 0
            while (tpl_sat_count<n_samples and attempts<max_attempts):
                
                attempts  =attempts+1
                flag = True
                n_objects = random.randint(3, 9)
                while (flag):
                    asgn = sample_assignment(domain, rule_text, rng) #generate an assignment for the rule class - rule_text
                    if (asgn, n_objects) not in history_asgn:
                        history_asgn.append((asgn, n_objects))
                        flag  = False


                instantiated_rule = apply_assignment(domain, rule_text, asgn)
                background = background_asp(domain, n_objects)
                full_program = background + instantiated_rule + "\n"
                text = verbalize(stem, asgn)
                try:
                    status, raw_models = solve(
                            full_program,
                            n_models=n_models,
                            time_limit=time_limit,
                            clingo_bin=clingo_bin,
                            )
                except RuntimeError as e:
                    print(f"  Solver error: {e}", file=sys.stderr)
                    status = "ERROR"

                if status == "SAT":
                    #record_id = _record_id(tpl_path.name, asgn, tpl_sat_count)
                    tpl_sat_count += 1
                    sat_count += 1
                    record_id = sat_count
                    #for r_m in raw_models:
                    #    scene = format_scene(r_m)
                    #    print(scene)


                elif status == "UNSAT":
                    #_record_id(tpl_path.name, asgn, tpl_unsat_count)
                    tpl_unsat_count += 1
                    unsat_count += 1
                    record_id = unsat_count
                else:
                    if verbose:
                        mark = {"TIMEOUT": "⏱", "UNKNOWN": "?", "ERROR": "!"}.get(status, "?")
                        print(f"  [{mark}] skipped ({status})")
                    continue

                total_count += 1

                # `text` is verbalize()'s full {"short","medium","long","subqa"} dict --
                # split subqa out into its own list so it lands in the record's top-level
                # "subqa" field instead of inside "prompts" (see build_combo_record).
                prompts_only = {k: text[k] for k in ("short", "medium", "long")}
                record = build_combo_record(
                    [cls], [tpl_path], prompts_only, [instantiated_rule], status, record_id,
                    n_objects, domain, subqas=[text.get("subqa", {})],
                )

                line = json.dumps(record) + "\n"

                if status == "SAT":
                    sat_f.write(line)
                elif status == "UNSAT":
                    unsat_f.write(line)

                if verbose:
                    mark = {"SAT": "✓", "UNSAT": "✗"}.get(status, "?")
                    print(f"  [{mark}] {record_id}")
            if (tpl_sat_count < n_samples):
                print('Tried max_attempts =1000 times but could produce:', tpl_sat_count, ' satisfiable and ', tpl_unsat_count, ' unsatisfiable prompts.')
    print(
        f"\nDone. {total_count} records written.\n"
        f"  SAT={sat_count}  → {sat_path}\n"
        f"  UNSAT={unsat_count}  → {unsat_path}"
    )




# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the CCIG constraint dataset from ASP template files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--domain", choices=["clevr", "coco"], default="clevr",
                        help="Which domain's property/value vocabulary to generate against.")
    parser.add_argument("--combo", type=int, default=1,
                        help="the number of constraints per prompt to be satisfied.")
    parser.add_argument("--template_dir", type=Path, default=TEMPLATE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Base output path; _SAT.jsonl and _UNSAT.jsonl are derived from it.")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--n_objects", type=int, default=4)
    parser.add_argument("--mode", choices=["random", "exhaustive"], default="random")
    parser.add_argument("--classes", nargs="+", metavar="C1")
    parser.add_argument("--no_solve", action="store_true")
    parser.add_argument("--n_models", type=int, default=1) 
    parser.add_argument("--time_limit", type=int, default=10)
    parser.add_argument("--clingo", type=str, default="clingo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        template_dir=args.template_dir,
        output_path=args.output,
        n_samples=args.samples, # is nuber of prompts generated?
        classes=args.classes, #complexity
        solve_programs=not args.no_solve,
        n_models=args.n_models, #is this no of plausible scene configurations for a setting? we probably do not need this now
        time_limit=args.time_limit,
        clingo_bin=args.clingo,
        seed=args.seed,
        verbose=args.verbose,
        domain=args.domain,
        combo = args.combo,
    )
