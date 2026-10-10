from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from  .metric_alignment_combo import compute_human_clip_alignment, compute_human_vlm_alignment, compute_human_soft_alignment, compute_human_perception_alignment
from .common.io import load_prompt_records


# =============================================================================
# I/O
# =============================================================================

def setup_plot_style() -> None:
    """Configure a clean publication-style plot."""

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def add_bar_labels(
    ax: plt.Axes,
    bars: Any,
) -> None:
    """Add percentage labels above bars."""

    for bar in bars:

        height = bar.get_height()

        if np.isnan(height):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )




def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save analysis results as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# Statistics helpers
# =============================================================================

def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """Return numerator / denominator, or 0 if denominator is zero."""
    return numerator / denominator if denominator else 0.0


def add_result(
    stats: dict[str, Any],
    correct: bool,
    no_count_correct: bool,
) -> None:
    """
    Add one evaluation result for short prompts.

    Statistics are stored as:

        total
        correct
        no_count_correct
    """

    stats["total"] += 1

    if correct:
        stats["correct"] += 1

    if no_count_correct:
        stats["no_count_correct"] += 1


# =============================================================================
# Human evaluation
# =============================================================================

def analyse_human_results(
    results: list[dict[str, Any]],
    records: dict[int, Any],
) -> dict[str, Any]:
    """
    Analyse human evaluation results for short prompts only.

    Returns overall accuracy and NoCount accuracy.
    """

    stats = {
        "total": 0,
        "correct": 0,
        "no_count_correct": 0,
    }

    for result in results:

        result_id = int(
            result["id"]
        )

        record = records.get(result_id)

        if record is None:
            print(
                f"Warning: no prompt record found for id={result_id}"
            )
            continue

        # --------------------------------------------------------------
        # Only evaluate short prompts.
        # --------------------------------------------------------------

        length = str(
            result["prompt_field"]
        ).lower()

        if length != "short":
            continue

        # --------------------------------------------------------------
        # Normal score.
        # --------------------------------------------------------------

        correct = (
            result["score"] == 1
        )

        # --------------------------------------------------------------
        # NoCount score.
        #
        # score == 1 is always counted.
        #
        # Otherwise, if both the dataset and prediction are SAT,
        # count it as correct while ignoring object-count mismatch.
        # --------------------------------------------------------------

        no_count_correct = (
            correct
            or (
                result.get("dataset_status") == "SAT"
                and result.get("predicted_status") == "SAT"
            )
        )

        # --------------------------------------------------------------
        # Update statistics.
        # --------------------------------------------------------------

        stats["total"] += 1

        if correct:
            stats["correct"] += 1

        if no_count_correct:
            stats["no_count_correct"] += 1

    # --------------------------------------------------------------
    # Convert counts to scores.
    # --------------------------------------------------------------

    return {
        "total": stats["total"],
        "correct": stats["correct"],
        "score": safe_divide(
            stats["correct"],
            stats["total"],
        ),
        "no_count_correct": stats["no_count_correct"],
        "score_no_count": safe_divide(
            stats["no_count_correct"],
            stats["total"],
        ),
    }


def run_analysis(
    prompt_file: str,
    domain: str,
    clipscore_results: str | Path,
    vlm_judge_results: str | Path,
    soft_tifa_results: str | Path,
    perception_results: str | Path,
    human_results: str | Path,
    analysis_out: str | Path,
) -> None:
    """
    Run the complete analysis.

    Outputs:

        analysis.json

        plots/
            complexity_vs_score.png
            complexity_class_1_families.png
            complexity_class_2_families.png
            complexity_class_3_families.png
            ...
            complexity_class_10_families.png    """

    setup_plot_style()

    # =========================================================================
    # Load data
    # =========================================================================

    records = load_prompt_records(prompt_file)

    clip_res = load_json(clipscore_results)
    vlm_judge_res = load_json(vlm_judge_results)
    tifa_res = load_json(soft_tifa_results)
    perception_res = load_json(perception_results)
    human_res = load_json(human_results)

    
    clip_human_alignment = compute_human_clip_alignment(
    human_res=human_res,
    clip_res=clip_res,)

    
    
    vlm_human_alignment = compute_human_vlm_alignment(
    human_res=human_res,
    vlm_judge_res=vlm_judge_res,)

    
    soft_human_alignment = compute_human_soft_alignment(
    human_res=human_res,
    soft_tifa_res=tifa_res,
    )
    
    roc_perc_plot = (
    Path(analysis_out).parent
    / "plots"
    / "perc_roc.png"
    )
    perc_human_alignment =  compute_human_perception_alignment(
    human_res=human_res,
    perception_res=perception_res,
    roc_plot_path=roc_perc_plot,)
    
    clip_alignment_out = Path(analysis_out).with_name(
    "clip_human_alignment.json")

    vlm_alignment_out = Path(analysis_out).with_name(
    "vlm_human_alignment.json")

    soft_alignment_out = Path(analysis_out).with_name(
    "soft_human_alignment.json")

    perc_alignment_out = Path(analysis_out).with_name(
    "perc_human_alignment.json")

    with open(clip_alignment_out, "w") as f:
        json.dump(
            clip_human_alignment,
            f,
      indent=4,)

    with open(vlm_alignment_out, "w") as f:
        json.dump(
            vlm_human_alignment,
            f,
            indent=4,)
    
    with open(soft_alignment_out, "w") as f:
        json.dump(
            soft_human_alignment,
            f,
            indent=4,)

    with open(perc_alignment_out, "w") as f:
        json.dump(
            perc_human_alignment,
            f,
            indent=4,)
    # These are loaded for the complete evaluation pipeline.
    # Add automated metric analysis here if required.
    #_ = (
    #    clip_res,
    #    vlm_judge_res,
    #    tifa_res,
    #    perception_res,
    #)

    results = human_res["results"]

    # =========================================================================
    # Human analysis
    # =========================================================================

    human_analysis = analyse_human_results(
        results,
        records,
    )
    # =========================================================================
    # Final JSON
    # =========================================================================

    analysis_path = Path(analysis_out)

    if analysis_path.suffix.lower() != ".json":
        analysis_path = analysis_path / "analysis.json"

    with open(analysis_path, "w") as f:
        json.dump(human_analysis, f, indent=2)

