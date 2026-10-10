from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from  .metric_alignment import compute_human_clip_alignment, compute_human_vlm_alignment, compute_human_soft_alignment, compute_human_perception_alignment
from .common.io import load_prompt_records


# =============================================================================
# I/O
# =============================================================================

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
    complexity_class: str,
    family: str,
    correct: bool,
    no_count_correct: bool,
) -> None:
    """
    Add one evaluation result.

    Statistics are stored as:

        complexity_class
            ├── total
            ├── correct
            └── no_count_correct

        complexity_class
            └── family
                ├── total
                ├── correct
                └── no_count_correct
    """

    # -------------------------------------------------------------------------
    # Complexity-class statistics
    # -------------------------------------------------------------------------

    class_stats = stats["complexity"].setdefault(
        complexity_class,
        {
            "total": 0,
            "correct": 0,
            "no_count_correct": 0,
        },
    )

    class_stats["total"] += 1

    if correct:
        class_stats["correct"] += 1

    if no_count_correct:
        class_stats["no_count_correct"] += 1

    # -------------------------------------------------------------------------
    # Family statistics within complexity class
    # -------------------------------------------------------------------------

    family_stats = (
        stats["family"]
        .setdefault(complexity_class, {})
        .setdefault(
            family,
            {
                "total": 0,
                "correct": 0,
                "no_count_correct": 0,
            },
        )
    )

    family_stats["total"] += 1

    if correct:
        family_stats["correct"] += 1

    if no_count_correct:
        family_stats["no_count_correct"] += 1


def convert_counts_to_scores(
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Convert raw counts into scores."""

    output = {
        "complexity": {},
        "family": {},
    }

    # -------------------------------------------------------------------------
    # Complexity
    # -------------------------------------------------------------------------

    for complexity, values in stats["complexity"].items():
        total = values["total"]

        output["complexity"][complexity] = {
            "total": total,
            "correct": values["correct"],
            "no_count_correct": values["no_count_correct"],
            "score": safe_divide(values["correct"], total),
            "score_no_count": safe_divide(
                values["no_count_correct"],
                total,
            ),
        }

    # -------------------------------------------------------------------------
    # Family within complexity
    # -------------------------------------------------------------------------

    for complexity, families in stats["family"].items():
        output["family"][complexity] = {}

        for family, values in families.items():
            total = values["total"]

            output["family"][complexity][family] = {
                "total": total,
                "correct": values["correct"],
                "no_count_correct": values["no_count_correct"],
                "score": safe_divide(
                    values["correct"],
                    total,
                ),
                "score_no_count": safe_divide(
                    values["no_count_correct"],
                    total,
                ),
            }

    return output


# =============================================================================
# Human evaluation
# =============================================================================

def analyse_human_results(
    results: list[dict[str, Any]],
    records: dict[int, Any],
) -> dict[str, Any]:
    """
    Analyse human evaluation results separately for short and long prompts.
    """

    stats = {
        "short": {
            "total": 0,
            "correct": 0,
            "no_count_correct": 0,
            "complexity": {},
            "family": {},
        },
        "long": {
            "total": 0,
            "correct": 0,
            "no_count_correct": 0,
            "complexity": {},
            "family": {},
        },
    }

    for result in results:

        result_id = int(result["id"])
        record = records.get(result_id)

        if record is None:
            print(
                f"Warning: no prompt record found for id={result_id}"
            )
            continue

        # Your prompt records appear to contain lists.
        complexity_class = str(
            record.complexity_class[0]
        )

        family = str(
            record.constraint_family[0]
        )

        length = str(
            result["prompt_field"]
        ).lower()

        if length not in {"short", "long"}:
            print(
                f"Warning: unknown prompt_field={length!r} "
                f"for id={result_id}"
            )
            continue

        # Normal score.
        correct = result["score"] == 1

        # ---------------------------------------------------------------------
        # NoCount score
        #
        # score == 1 is always counted.
        #
        # Otherwise, if both the dataset and prediction are SAT, count it as
        # correct while ignoring the object-count mismatch.
        # ---------------------------------------------------------------------

        no_count_correct = (
            correct
            or (
                result.get("dataset_status") == "SAT"
                and result.get("predicted_status") == "SAT"
            )
        )

        # ---------------------------------------------------------------------
        # Overall
        # ---------------------------------------------------------------------

        stats[length]["total"] += 1

        if correct:
            stats[length]["correct"] += 1

        if no_count_correct:
            stats[length]["no_count_correct"] += 1

        # ---------------------------------------------------------------------
        # Complexity + family
        # ---------------------------------------------------------------------

        add_result(
            stats=stats[length],
            complexity_class=complexity_class,
            family=family,
            correct=correct,
            no_count_correct=no_count_correct,
        )

    # -------------------------------------------------------------------------
    # Convert counts to scores
    # -------------------------------------------------------------------------

    analysis = {}

    for length in ("short", "long"):

        length_stats = stats[length]

        complexity_scores = convert_counts_to_scores(
            {
                "complexity": length_stats["complexity"],
                "family": {},
            }
        )["complexity"]

        family_scores = convert_counts_to_scores(
            {
                "complexity": {},
                "family": length_stats["family"],
            }
        )["family"]

        analysis[length] = {
            "total": length_stats["total"],
            "correct": length_stats["correct"],
            "score": safe_divide(
                length_stats["correct"],
                length_stats["total"],
            ),
            "no_count_correct": length_stats["no_count_correct"],
            "score_no_count": safe_divide(
                length_stats["no_count_correct"],
                length_stats["total"],
            ),
            "complexity": complexity_scores,
            "family": family_scores,
        }

    return analysis


# =============================================================================
# Short / Long consistency
# =============================================================================

def calculate_short_long_consistency(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare short and long scores for each prompt ID.

    This replaces the original nested O(N^2) loop.
    """

    by_id: dict[int, dict[str, int]] = {}

    for result in results:

        result_id = int(result["id"])
        length = str(result["prompt_field"]).lower()

        if length not in {"short", "long"}:
            continue

        by_id.setdefault(result_id, {})[length] = result["score"]

    complete_pairs = 0
    short_correct_long_incorrect = 0
    long_correct_short_incorrect = 0
    both_correct = 0
    both_incorrect = 0

    for scores in by_id.values():

        if "short" not in scores or "long" not in scores:
            continue

        complete_pairs += 1

        short_score = scores["short"]
        long_score = scores["long"]

        if short_score == 1 and long_score != 1:
            short_correct_long_incorrect += 1

        elif long_score == 1 and short_score != 1:
            long_correct_short_incorrect += 1

        elif short_score == 1 and long_score == 1:
            both_correct += 1

        else:
            both_incorrect += 1

    return {
        "complete_pairs": complete_pairs,
        "short_correct_long_incorrect": (
            short_correct_long_incorrect
        ),
        "long_correct_short_incorrect": (
            long_correct_short_incorrect
        ),
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "short_correct_long_incorrect_rate": safe_divide(
            short_correct_long_incorrect,
            complete_pairs,
        ),
        "long_correct_short_incorrect_rate": safe_divide(
            long_correct_short_incorrect,
            complete_pairs,
        ),
    }


# =============================================================================
# Plot styling
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


# =============================================================================
# Plot 1
# Complexity class: SHORT vs LONG
# =============================================================================

def plot_complexity_short_long(
    analysis: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    One plot containing:

        Complexity 1: Short | Long
        Complexity 2: Short | Long
        Complexity 3: Short | Long
        ...

    Normal score is plotted.
    """

    short_data = analysis["short"]["complexity"]
    long_data = analysis["long"]["complexity"]

    # Include every complexity class appearing in either dataset.
    complexity_classes = list(
        dict.fromkeys(
            list(short_data.keys()) +
            list(long_data.keys())
        )
    )

    if not complexity_classes:
        return

    short_scores = [
        short_data.get(c, {}).get("score", np.nan)
        for c in complexity_classes
    ]

    long_scores = [
        long_data.get(c, {}).get("score", np.nan)
        for c in complexity_classes
    ]

    x = np.arange(len(complexity_classes))

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    short_bars = ax.bar(
        x - width / 2,
        short_scores,
        width,
        label="Short",
        color="#4C78A8",
    )

    long_bars = ax.bar(
        x + width / 2,
        long_scores,
        width,
        label="Long",
        color="#F58518",
    )

    add_bar_labels(ax, short_bars)
    add_bar_labels(ax, long_bars)

    ax.set_title(
        "Score by Complexity Class: Short vs Long",
        pad=15,
    )

    ax.set_xlabel("Complexity class")
    ax.set_ylabel("Score")

    ax.set_xticks(x)
    ax.set_xticklabels(complexity_classes)

    ax.set_ylim(0, 1.10)

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda y, _: f"{y:.0%}"
        )
    )

    ax.legend()

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# Plot 2
# Family scores for EACH complexity class
# =============================================================================

def plot_family_scores_by_complexity(
    analysis: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """
    Create one PNG for each complexity class.

    For every complexity class, the plot shows:

        Family A     Short | Long
        Family B     Short | Long
        Family C     Short | Long
        ...

    Example output:

        family_complexity_1.png
        family_complexity_2.png
        family_complexity_3.png
        ...
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    short_data = analysis["short"]["family"]
    long_data = analysis["long"]["family"]

    # -------------------------------------------------------------------------
    # Get all complexity classes appearing in either short or long.
    # -------------------------------------------------------------------------

    complexity_classes = list(
        dict.fromkeys(
            list(short_data.keys())
            + list(long_data.keys())
        )
    )

    if not complexity_classes:
        print("No family data found.")
        return

    # -------------------------------------------------------------------------
    # Create ONE figure for EACH complexity class.
    # -------------------------------------------------------------------------

    for complexity in complexity_classes:

        short_families = short_data.get(
            complexity,
            {},
        )

        long_families = long_data.get(
            complexity,
            {},
        )

        # Families appearing in either Short or Long.
        families = list(
            dict.fromkeys(
                list(short_families.keys())
                + list(long_families.keys())
            )
        )

        if not families:
            continue

        # ---------------------------------------------------------------------
        # Get scores
        # ---------------------------------------------------------------------

        short_scores = []

        long_scores = []

        for family in families:

            short_score = short_families.get(
                family,
                {},
            ).get(
                "score",
                np.nan,
            )

            long_score = long_families.get(
                family,
                {},
            ).get(
                "score",
                np.nan,
            )

            short_scores.append(short_score)
            long_scores.append(long_score)

        # ---------------------------------------------------------------------
        # Plot
        # ---------------------------------------------------------------------

        x = np.arange(len(families))

        width = 0.36

        # Automatically make the figure wider if there are many families.
        fig_width = max(
            8,
            len(families) * 1.5,
        )

        fig, ax = plt.subplots(
            figsize=(fig_width, 6),
        )

        short_bars = ax.bar(
            x - width / 2,
            short_scores,
            width,
            label="Short",
            color="#4C78A8",
        )

        long_bars = ax.bar(
            x + width / 2,
            long_scores,
            width,
            label="Long",
            color="#F58518",
        )

        # ---------------------------------------------------------------------
        # Add percentage labels
        # ---------------------------------------------------------------------

        add_bar_labels(
            ax,
            short_bars,
        )

        add_bar_labels(
            ax,
            long_bars,
        )

        # ---------------------------------------------------------------------
        # Labels / title
        # ---------------------------------------------------------------------

        ax.set_title(
            f"Constraint Family Scores — Complexity Class {complexity}",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )

        ax.set_xlabel(
            "Constraint family",
        )

        ax.set_ylabel(
            "Score",
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            families,
            rotation=35,
            ha="right",
        )

        # Scores are between 0 and 1.
        ax.set_ylim(
            0,
            1.10,
        )

        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda y, _: f"{y:.0%}"
            )
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        ax.legend()

        # ---------------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------------

        output_path = (
            output_dir
            / f"family_complexity_{complexity}.png"
        )

        fig.tight_layout()

        fig.savefig(
            output_path,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved family plot: {output_path}"
        )

# =============================================================================
# Main
# =============================================================================

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

    roc_clip_plot = (Path(analysis_out).parent/ "plots"/ "clip_roc.png")

    clip_human_alignment = compute_human_clip_alignment(
    human_res=human_res,
    clip_res=clip_res,
    roc_plot_path=roc_clip_plot,)

    roc_vlm_plot = (
    Path(analysis_out).parent
    / "plots"
    / "vlm_roc.png"
    )
    
    vlm_human_alignment = compute_human_vlm_alignment(
    human_res=human_res,
    vlm_judge_res=vlm_judge_res,
    roc_plot_path=roc_vlm_plot,)

    roc_soft_plot = (
    Path(analysis_out).parent
    / "plots"
    / "soft_roc.png"
    )
    
    soft_human_alignment = compute_human_soft_alignment(
    human_res=human_res,
    soft_tifa_res=tifa_res,
    roc_plot_path=roc_soft_plot,)
    
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
    # Short / long consistency
    # =========================================================================

    consistency = calculate_short_long_consistency(
        results,
    )

    # =========================================================================
    # Final JSON
    # =========================================================================

    analysis = {
        "domain": domain,

        "summary": {
            "short": {
                "total": human_analysis["short"]["total"],
                "correct": human_analysis["short"]["correct"],
                "score": human_analysis["short"]["score"],
                "no_count_correct": (
                    human_analysis["short"]["no_count_correct"]
                ),
                "score_no_count": (
                    human_analysis["short"]["score_no_count"]
                ),
            },

            "long": {
                "total": human_analysis["long"]["total"],
                "correct": human_analysis["long"]["correct"],
                "score": human_analysis["long"]["score"],
                "no_count_correct": (
                    human_analysis["long"]["no_count_correct"]
                ),
                "score_no_count": (
                    human_analysis["long"]["score_no_count"]
                ),
            },
        },

        "short": human_analysis["short"],
        "long": human_analysis["long"],

        "short_long_consistency": consistency,#
        "clip_human_alignment": clip_human_alignment,
        "vlm_human_alignment": vlm_human_alignment,
        "soft_human_alignment": soft_human_alignment
    }

    # =========================================================================
    # Save JSON
    # =========================================================================

    analysis_path = Path(analysis_out)

    if analysis_path.suffix.lower() != ".json":
        analysis_path = analysis_path / "analysis.json"

    save_json(
        analysis,
        analysis_path,
    )

    # =========================================================================
    # Save plots
    # =========================================================================

    plot_dir = analysis_path.parent / "plots"
    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Plot 1:
    # Complexity class with Short vs Long side-by-side
    # -------------------------------------------------------------------------

    plot_complexity_short_long(
        analysis,
        plot_dir / "complexity_vs_score.png",
    )

    # -------------------------------------------------------------------------
    # Plot 2:
    # One subplot for each complexity class, with family Short vs Long
    # -------------------------------------------------------------------------

    plot_family_scores_by_complexity(
        analysis,
        plot_dir / "family_scores_by_complexity.png",
    )

    # =========================================================================
    # Console output
    # =========================================================================

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    for length in ("short", "long"):

        data = human_analysis[length]

        print(f"\n{length.upper()}")
        print("-" * 70)

        print(f"Total:              {data['total']}")
        print(f"Correct:            {data['correct']}")
        print(f"Score:              {data['score']:.2%}")
        print(
            f"NoCount correct:    "
            f"{data['no_count_correct']}"
        )
        print(
            f"NoCount score:      "
            f"{data['score_no_count']:.2%}"
        )

        print("\nComplexity classes:")

        for complexity, values in data["complexity"].items():

            print(
                f"  {complexity:<20}"
                f"{values['score']:.2%}"
                f"  (n={values['total']})"
            )

    print("\nSHORT/LONG CONSISTENCY")
    print("-" * 70)

    for key, value in consistency.items():

        if key.endswith("_rate"):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")

    print("\nOUTPUT")
    print("-" * 70)

    print(f"JSON:   {analysis_path}")
    print(f"Plots:  {plot_dir}")
    print("=" * 70)
