from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _point_biserial_correlation(
    cosine_scores: np.ndarray,
    human_scores: np.ndarray,
) -> float | None:
    """
    Compute point-biserial correlation between continuous CLIP
    cosine scores and binary human scores (0/1).

    Point-biserial correlation is equivalent to Pearson
    correlation when the binary variable is represented as 0/1.

    Returns None if the correlation is undefined.
    """

    if len(cosine_scores) < 2:
        return None

    if len(cosine_scores) != len(human_scores):
        raise ValueError(
            "cosine_scores and human_scores must have "
            "the same length."
        )

    # Both human classes must be present.
    if len(np.unique(human_scores)) < 2:
        return None

    # CLIP scores must have non-zero variance.
    if np.std(cosine_scores) == 0:
        return None

    return float(
        np.corrcoef(
            cosine_scores,
            human_scores,
        )[0, 1]
    )


def _roc_curve_and_auc(
    cosine_scores: np.ndarray,
    human_scores: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float | None,
]:
    """
    Compute ROC curve and ROC-AUC manually.

    Human score:
        1 = positive
        0 = negative

    CLIP cosine is used as the prediction score.

    Returns
    -------
    fpr:
        False-positive rate values.

    tpr:
        True-positive rate values.

    auc:
        Area under the ROC curve.
    """

    positives = np.sum(human_scores == 1)
    negatives = np.sum(human_scores == 0)

    if positives == 0 or negatives == 0:
        return (
            np.array([]),
            np.array([]),
            None,
        )

    # ------------------------------------------------------------------
    # Sort by CLIP score from highest to lowest.
    # ------------------------------------------------------------------

    order = np.argsort(
        -cosine_scores,
        kind="stable",
    )

    sorted_scores = cosine_scores[order]
    sorted_human = human_scores[order]

    # ------------------------------------------------------------------
    # Evaluate thresholds at every distinct cosine value.
    # ------------------------------------------------------------------

    thresholds = np.r_[
        np.inf,
        np.unique(sorted_scores)[::-1],
    ]

    tpr = []
    fpr = []

    for threshold in thresholds:

        predicted_positive = (
            cosine_scores >= threshold
        )

        true_positive = np.sum(
            predicted_positive
            & (human_scores == 1)
        )

        false_positive = np.sum(
            predicted_positive
            & (human_scores == 0)
        )

        current_tpr = (
            true_positive / positives
        )

        current_fpr = (
            false_positive / negatives
        )

        tpr.append(current_tpr)
        fpr.append(current_fpr)

    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    # ------------------------------------------------------------------
    # Calculate area under the curve using trapezoidal integration.
    # ------------------------------------------------------------------

    auc = float(
        np.trapezoid(tpr, fpr)
    )

    return fpr, tpr, auc


def compute_human_clip_alignment(
    human_res: dict[str, Any],
    clip_res: dict[str, Any],
    roc_plot_path: str | Path,
) -> dict[str, Any]:
    """
    Compute alignment between human evaluation and CLIP cosine.

    Human evaluation:
        score = 1 -> correct
        score = 0 -> incorrect

    CLIP:
        cosine in [-1, 1]

    Metrics:
        1. Mean cosine
        2. Mean cosine for human=0
        3. Mean cosine for human=1
        4. Point-biserial correlation
        5. ROC-AUC

    Also saves a ROC curve comparing Short and Long prompts.

    Parameters
    ----------
    human_res:
        Loaded human evaluation JSON.

    clip_res:
        Loaded CLIP evaluation JSON.

    roc_plot_path:
        Path where the ROC plot should be saved.

    Returns
    -------
    dict
        Alignment statistics for short, long, and overall.
    """

    roc_plot_path = Path(roc_plot_path)
    

    # ==================================================================
    # Build human lookup
    # ==================================================================

    human_by_key: dict[
        tuple[int, str],
        int,
    ] = {}

    for result in human_res["results"]:

        result_id = int(result["id"])

        prompt_field = str(
            result["prompt_field"]
        ).lower()

        human_score = int(
            result["score"]
        )

        human_by_key[
            (result_id, prompt_field)
        ] = human_score

    # ==================================================================
    # Build CLIP lookup
    # ==================================================================

    clip_by_key: dict[
        tuple[int, str],
        float,
    ] = {}

    for result in clip_res["results"]:

        result_id = int(result["id"])

        prompt_field = str(
            result["prompt_field"]
        ).lower()

        cosine = result.get("cosine")

        if cosine is None:
            continue

        clip_by_key[
            (result_id, prompt_field)
        ] = float(cosine)

    # ==================================================================
    # Match human and CLIP results
    # ==================================================================

    matched: dict[
        str,
        list[tuple[float, int]],
    ] = {
        "short": [],
        "long": [],
    }

    for key, human_score in human_by_key.items():

        clip_score = clip_by_key.get(key)

        if clip_score is None:
            continue

        result_id, prompt_field = key

        if prompt_field not in matched:
            continue

        matched[prompt_field].append(
            (
                clip_score,
                human_score,
            )
        )

    # ==================================================================
    # Calculate metrics for one subset
    # ==================================================================

    def calculate_metrics(
        pairs: list[tuple[float, int]],
    ) -> dict[str, Any]:

        if not pairs:
            return {
                "n": 0,
                "n_human_0": 0,
                "n_human_1": 0,
                "mean_cosine": None,
                "mean_cosine_human_0": None,
                "mean_cosine_human_1": None,
                "mean_cosine_difference_1_minus_0": None,
                "point_biserial": None,
                "roc_auc": None,
            }

        cosine_scores = np.asarray(
            [pair[0] for pair in pairs],
            dtype=float,
        )

        human_scores = np.asarray(
            [pair[1] for pair in pairs],
            dtype=int,
        )

        # --------------------------------------------------------------
        # Mean cosine
        # --------------------------------------------------------------

        mean_cosine = float(
            np.mean(cosine_scores)
        )

        # --------------------------------------------------------------
        # Mean cosine by human judgment
        # --------------------------------------------------------------

        cosine_human_0 = cosine_scores[
            human_scores == 0
        ]

        cosine_human_1 = cosine_scores[
            human_scores == 1
        ]

        mean_cosine_human_0 = (
            float(np.mean(cosine_human_0))
            if len(cosine_human_0) > 0
            else None
        )

        mean_cosine_human_1 = (
            float(np.mean(cosine_human_1))
            if len(cosine_human_1) > 0
            else None
        )

        # --------------------------------------------------------------
        # Difference between human-positive and human-negative means
        # --------------------------------------------------------------

        if (
            mean_cosine_human_0 is not None
            and mean_cosine_human_1 is not None
        ):
            mean_difference = (
                mean_cosine_human_1
                - mean_cosine_human_0
            )
        else:
            mean_difference = None

        # --------------------------------------------------------------
        # Point-biserial correlation
        # --------------------------------------------------------------

        point_biserial = (
            _point_biserial_correlation(
                cosine_scores,
                human_scores,
            )
        )

        # --------------------------------------------------------------
        # ROC curve + AUC
        # --------------------------------------------------------------

        _, _, roc_auc = _roc_curve_and_auc(
            cosine_scores,
            human_scores,
        )

        return {
            "n": len(pairs),

            "n_human_0": int(
                np.sum(human_scores == 0)
            ),

            "n_human_1": int(
                np.sum(human_scores == 1)
            ),

            "mean_cosine": mean_cosine,

            "mean_cosine_human_0": (
                mean_cosine_human_0
            ),

            "mean_cosine_human_1": (
                mean_cosine_human_1
            ),

            "mean_cosine_difference_1_minus_0": (
                mean_difference
            ),

            "point_biserial": point_biserial,

            "roc_auc": roc_auc,
        }

    # ==================================================================
    # Short
    # ==================================================================

    short_metrics = calculate_metrics(
        matched["short"]
    )

    # ==================================================================
    # Long
    # ==================================================================

    long_metrics = calculate_metrics(
        matched["long"]
    )

    # ==================================================================
    # Overall
    # ==================================================================

    overall_pairs = (
        matched["short"]
        + matched["long"]
    )

    overall_metrics = calculate_metrics(
        overall_pairs
    )

    # ==================================================================
    # ROC plot
    # ==================================================================

    plt.figure(
        figsize=(8, 7)
    )

    # ------------------------------------------------------------------
    # Short ROC
    # ------------------------------------------------------------------

    short_pairs = matched["short"]

    if short_pairs:

        short_cosine = np.asarray(
            [pair[0] for pair in short_pairs],
            dtype=float,
        )

        short_human = np.asarray(
            [pair[1] for pair in short_pairs],
            dtype=int,
        )

        (
            short_fpr,
            short_tpr,
            short_auc,
        ) = _roc_curve_and_auc(
            short_cosine,
            short_human,
        )

        if short_auc is not None:

            plt.plot(
                short_fpr,
                short_tpr,
                linewidth=2.5,
                label=f"Short (AUC = {short_auc:.3f})",
            )

    # ------------------------------------------------------------------
    # Long ROC
    # ------------------------------------------------------------------

    long_pairs = matched["long"]

    if long_pairs:

        long_cosine = np.asarray(
            [pair[0] for pair in long_pairs],
            dtype=float,
        )

        long_human = np.asarray(
            [pair[1] for pair in long_pairs],
            dtype=int,
        )

        (
            long_fpr,
            long_tpr,
            long_auc,
        ) = _roc_curve_and_auc(
            long_cosine,
            long_human,
        )

        if long_auc is not None:

            plt.plot(
                long_fpr,
                long_tpr,
                linewidth=2.5,
                label=f"Long (AUC = {long_auc:.3f})",
            )

    # ------------------------------------------------------------------
    # Random classifier
    # ------------------------------------------------------------------

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Random (AUC = 0.500)",
    )

    plt.xlabel(
        "False Positive Rate",
        fontsize=12,
    )

    plt.ylabel(
        "True Positive Rate",
        fontsize=12,
    )

    plt.title(
        "CLIP Cosine Similarity vs Human Evaluation",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlim(
        0,
        1,
    )

    plt.ylim(
        0,
        1,
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    plt.legend(
        loc="lower right",
        frameon=True,
    )

    plt.tight_layout()

    plt.savefig(
        roc_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ==================================================================
    # Return analysis
    # ==================================================================

    return {
        "metric": "clip_cosine_vs_human_score",

        "clip_score": {
            "field": "cosine",
            "range": [-1, 1],
            "interpretation": (
                "Higher cosine indicates stronger "
                "text-image alignment."
            ),
        },

        "human_score": {
            "field": "score",
            "values": [0, 1],
            "interpretation": (
                "1 = human judged the generation correct; "
                "0 = incorrect."
            ),
        },

        "short": short_metrics,

        "long": long_metrics,

        "overall": overall_metrics,

        "roc_plot": str(
            roc_plot_path
        ),
    }


def compute_human_vlm_alignment(
    human_res: dict[str, Any],
    vlm_judge_res: dict[str, Any],
    roc_plot_path: str | Path,
) -> dict[str, Any]:
    """
    Compute alignment between VLM-judge scores and human evaluation.

    Human evaluation:
        score = 1 -> correct
        score = 0 -> incorrect

    VLM judge:
        score in [0, 1]
        higher score = stronger text-image alignment.

    Metrics:
        - Mean VLM score
        - Mean VLM score for human=0
        - Mean VLM score for human=1
        - Difference between human=1 and human=0 means
        - Point-biserial correlation
        - ROC-AUC

    Also saves one ROC plot containing Short and Long curves.

    Parameters
    ----------
    human_res:
        Loaded human evaluation JSON.

    vlm_judge_res:
        Loaded VLM-judge evaluation JSON.

    roc_plot_path:
        Path where the ROC PNG should be saved.

    Returns
    -------
    dict
        Alignment statistics for short, long, and overall.
    """

    roc_plot_path = Path(roc_plot_path)

    

    # ==================================================================
    # Human results lookup
    # ==================================================================

    human_by_key: dict[
        tuple[int, str],
        int,
    ] = {}

    for result in human_res["results"]:

        result_id = int(
            result["id"]
        )

        prompt_field = str(
            result["prompt_field"]
        ).lower()

        human_score = int(
            result["score"]
        )

        human_by_key[
            (result_id, prompt_field)
        ] = human_score

    # ==================================================================
    # VLM results lookup
    # ==================================================================

    vlm_by_key: dict[
        tuple[int, str],
        float,
    ] = {}

    for result in vlm_judge_res["results"]:

        result_id = int(
            result["id"]
        )

        prompt_field = str(
            result["prompt_field"]
        ).lower()

        vlm_score = result.get("score")

        if vlm_score is None:
            continue

        vlm_by_key[
            (result_id, prompt_field)
        ] = float(vlm_score)

    # ==================================================================
    # Match human and VLM results
    # ==================================================================

    matched: dict[
        str,
        list[tuple[float, int]],
    ] = {
        "short": [],
        "long": [],
    }

    for key, human_score in human_by_key.items():

        vlm_score = vlm_by_key.get(key)

        if vlm_score is None:
            continue

        result_id, prompt_field = key

        if prompt_field not in matched:
            continue

        matched[prompt_field].append(
            (
                vlm_score,
                human_score,
            )
        )

    # ==================================================================
    # Calculate metrics
    # ==================================================================

    def calculate_metrics(
        pairs: list[tuple[float, int]],
    ) -> dict[str, Any]:

        if not pairs:
            return {
                "n": 0,
                "n_human_0": 0,
                "n_human_1": 0,
                "mean_vlm_score": None,
                "mean_vlm_score_human_0": None,
                "mean_vlm_score_human_1": None,
                "mean_vlm_difference_1_minus_0": None,
                "point_biserial": None,
                "roc_auc": None,
            }

        vlm_scores = np.asarray(
            [pair[0] for pair in pairs],
            dtype=float,
        )

        human_scores = np.asarray(
            [pair[1] for pair in pairs],
            dtype=int,
        )

        # --------------------------------------------------------------
        # Mean VLM score
        # --------------------------------------------------------------

        mean_vlm_score = float(
            np.mean(vlm_scores)
        )

        # --------------------------------------------------------------
        # Mean VLM score for human=0
        # --------------------------------------------------------------

        vlm_human_0 = vlm_scores[
            human_scores == 0
        ]

        mean_vlm_human_0 = (
            float(np.mean(vlm_human_0))
            if len(vlm_human_0) > 0
            else None
        )

        # --------------------------------------------------------------
        # Mean VLM score for human=1
        # --------------------------------------------------------------

        vlm_human_1 = vlm_scores[
            human_scores == 1
        ]

        mean_vlm_human_1 = (
            float(np.mean(vlm_human_1))
            if len(vlm_human_1) > 0
            else None
        )

        # --------------------------------------------------------------
        # Difference between human-positive and human-negative means
        # --------------------------------------------------------------

        if (
            mean_vlm_human_0 is not None
            and mean_vlm_human_1 is not None
        ):
            mean_difference = (
                mean_vlm_human_1
                - mean_vlm_human_0
            )
        else:
            mean_difference = None

        # --------------------------------------------------------------
        # Point-biserial correlation
        # --------------------------------------------------------------

        point_biserial = (
            _point_biserial_correlation(
                vlm_scores,
                human_scores,
            )
        )

        # --------------------------------------------------------------
        # ROC-AUC
        # --------------------------------------------------------------

        _, _, roc_auc = _roc_curve_and_auc(
            vlm_scores,
            human_scores,
        )

        return {
            "n": len(pairs),

            "n_human_0": int(
                np.sum(human_scores == 0)
            ),

            "n_human_1": int(
                np.sum(human_scores == 1)
            ),

            "mean_vlm_score": mean_vlm_score,

            "mean_vlm_score_human_0": (
                mean_vlm_human_0
            ),

            "mean_vlm_score_human_1": (
                mean_vlm_human_1
            ),

            "mean_vlm_difference_1_minus_0": (
                mean_difference
            ),

            "point_biserial": point_biserial,

            "roc_auc": roc_auc,
        }

    # ==================================================================
    # Short
    # ==================================================================

    short_metrics = calculate_metrics(
        matched["short"]
    )

    # ==================================================================
    # Long
    # ==================================================================

    long_metrics = calculate_metrics(
        matched["long"]
    )

    # ==================================================================
    # Overall
    # ==================================================================

    overall_pairs = (
        matched["short"]
        + matched["long"]
    )

    overall_metrics = calculate_metrics(
        overall_pairs
    )

    # ==================================================================
    # ROC plot
    # ==================================================================

    plt.figure(
        figsize=(8, 7)
    )

    # --------------------------------------------------------------
    # Short
    # --------------------------------------------------------------

    if matched["short"]:

        short_scores = np.asarray(
            [pair[0] for pair in matched["short"]],
            dtype=float,
        )

        short_human = np.asarray(
            [pair[1] for pair in matched["short"]],
            dtype=int,
        )

        (
            short_fpr,
            short_tpr,
            short_auc,
        ) = _roc_curve_and_auc(
            short_scores,
            short_human,
        )

        if short_auc is not None:

            plt.plot(
                short_fpr,
                short_tpr,
                linewidth=2.5,
                label=(
                    f"Short (AUC = {short_auc:.3f})"
                ),
            )

    # --------------------------------------------------------------
    # Long
    # --------------------------------------------------------------

    if matched["long"]:

        long_scores = np.asarray(
            [pair[0] for pair in matched["long"]],
            dtype=float,
        )

        long_human = np.asarray(
            [pair[1] for pair in matched["long"]],
            dtype=int,
        )

        (
            long_fpr,
            long_tpr,
            long_auc,
        ) = _roc_curve_and_auc(
            long_scores,
            long_human,
        )

        if long_auc is not None:

            plt.plot(
                long_fpr,
                long_tpr,
                linewidth=2.5,
                label=(
                    f"Long (AUC = {long_auc:.3f})"
                ),
            )

    # --------------------------------------------------------------
    # Random baseline
    # --------------------------------------------------------------

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Random (AUC = 0.500)",
    )

    plt.xlabel(
        "False Positive Rate",
        fontsize=12,
    )

    plt.ylabel(
        "True Positive Rate",
        fontsize=12,
    )

    plt.title(
        "VLM-Judge Score vs Human Evaluation",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    plt.legend(
        loc="lower right",
        frameon=True,
    )

    plt.tight_layout()

    plt.savefig(
        roc_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ==================================================================
    # Return
    # ==================================================================

    return {
        "metric": "vlm_judge_vs_human_score",

        "vlm_score": {
            "field": "score",
            "range": [0, 1],
            "interpretation": (
                "Higher score indicates stronger "
                "text-image alignment."
            ),
        },

        "human_score": {
            "field": "score",
            "values": [0, 1],
            "interpretation": (
                "1 = human judged the generation correct; "
                "0 = incorrect."
            ),
        },

        "short": short_metrics,

        "long": long_metrics,

        "overall": overall_metrics,

        "roc_plot": str(
            roc_plot_path
        ),
    }

def compute_human_soft_alignment(
    human_res: dict[str, Any],
    soft_tifa_res: dict[str, Any],
    roc_plot_path: str | Path,
) -> dict[str, Any]:
    """
    Compute alignment between soft tifa scores and human evaluation.

    Human evaluation:
        score = 1 -> correct
        score = 0 -> incorrect

    Soft-tifa:
        score in [0, 1]
        higher score = stronger text-image alignment.

    Metrics:
        - Mean soft tifa score
        - Mean soft tifa score for human=0
        - Mean soft tifa score for human=1
        - Difference between human=1 and human=0 means
        - Point-biserial correlation
        - ROC-AUC

    Also saves one ROC plot containing Short and Long curves.

    Parameters
    ----------
    human_res:
        Loaded human evaluation JSON.

    soft_tifa_res:
        Loaded Soft-tifa evaluation JSON.

    roc_plot_path:
        Path where the ROC PNG should be saved.

    Returns
    -------
    dict
        Alignment statistics for short, long, and overall.
    """

    roc_plot_path = Path(roc_plot_path)

    

    # ==================================================================
    # Human results lookup
    # ==================================================================

    human_by_key: dict[
        tuple[int, str],
        int,
    ] = {}

    for result in human_res["results"]:

        result_id = int(
            result["id"]
        )

        prompt_field = str(
            result["prompt_field"]
        ).lower()

        human_score = int(
            result["score"]
        )

        human_by_key[
            (result_id, prompt_field)
        ] = human_score

    # ==================================================================
    # soft tifa results lookup
    # ==================================================================

    soft_tifa_by_key: dict[
        tuple[int, str],
        float,
    ] = {}

    for result in soft_tifa_res["results"]:

        result_id = int(
            result["id"]
        )

        prompt_field = str(
            result["prompt_field"]
        ).lower()

        soft_tifa_score = result.get("score_am")

        if soft_tifa_score is None:
            continue

        soft_tifa_by_key[
            (result_id, prompt_field)
        ] = float(soft_tifa_score)

    # ==================================================================
    # Match human and soft tifa results
    # ==================================================================

    matched: dict[
        str,
        list[tuple[float, int]],
    ] = {
        "short": [],
        "long": [],
    }

    for key, human_score in human_by_key.items():

        soft_tifa_score = soft_tifa_by_key.get(key)

        if soft_tifa_score is None:
            continue

        result_id, prompt_field = key

        if prompt_field not in matched:
            continue

        matched[prompt_field].append(
            (
                soft_tifa_score,
                human_score,
            )
        )

    # ==================================================================
    # Calculate metrics
    # ==================================================================

    def calculate_metrics(
        pairs: list[tuple[float, int]],
    ) -> dict[str, Any]:

        if not pairs:
            return {
                "n": 0,
                "n_human_0": 0,
                "n_human_1": 0,
                "mean_soft_tifa_score": None,
                "mean_soft_tifa_score_human_0": None,
                "mean_soft_tifa_score_human_1": None,
                "mean_soft_tifa_difference_1_minus_0": None,
                "point_biserial": None,
                "roc_auc": None,
            }

        soft_tifa_scores = np.asarray(
            [pair[0] for pair in pairs],
            dtype=float,
        )

        human_scores = np.asarray(
            [pair[1] for pair in pairs],
            dtype=int,
        )

        # --------------------------------------------------------------
        # Mean soft tifa score
        # --------------------------------------------------------------

        mean_soft_tifa_score = float(
            np.mean(soft_tifa_scores)
        )

        # --------------------------------------------------------------
        # Mean VLM score for human=0
        # --------------------------------------------------------------

        soft_tifa_human_0 = soft_tifa_scores[
            human_scores == 0
        ]

        mean_soft_tifa_human_0 = (
            float(np.mean(soft_tifa_human_0))
            if len(soft_tifa_human_0) > 0
            else None
        )

        # --------------------------------------------------------------
        # Mean soft tifa score for human=1
        # --------------------------------------------------------------

        soft_tifa_human_1 = soft_tifa_scores[
            human_scores == 1
        ]

        mean_soft_tifa_human_1 = (
            float(np.mean(soft_tifa_human_1))
            if len(soft_tifa_human_1) > 0
            else None
        )

        # --------------------------------------------------------------
        # Difference between human-positive and human-negative means
        # --------------------------------------------------------------

        if (
            mean_soft_tifa_human_0 is not None
            and mean_soft_tifa_human_1 is not None
        ):
            mean_difference = (
                mean_soft_tifa_human_1
                - mean_soft_tifa_human_0
            )
        else:
            mean_difference = None

        # --------------------------------------------------------------
        # Point-biserial correlation
        # --------------------------------------------------------------

        point_biserial = (
            _point_biserial_correlation(
                soft_tifa_scores,
                human_scores,
            )
        )

        # --------------------------------------------------------------
        # ROC-AUC
        # --------------------------------------------------------------

        _, _, roc_auc = _roc_curve_and_auc(
            soft_tifa_scores,
            human_scores,
        )

        return {
            "n": len(pairs),

            "n_human_0": int(
                np.sum(human_scores == 0)
            ),

            "n_human_1": int(
                np.sum(human_scores == 1)
            ),

            "mean_soft_tifa_score": mean_soft_tifa_score,

            "mean_soft_tifa_score_human_0": (
                mean_soft_tifa_human_0
            ),

            "mean_soft_tifa_score_human_1": (
                mean_soft_tifa_human_1
            ),

            "mean_soft_tifa_difference_1_minus_0": (
                mean_difference
            ),

            "point_biserial": point_biserial,

            "roc_auc": roc_auc,
        }

    # ==================================================================
    # Short
    # ==================================================================

    short_metrics = calculate_metrics(
        matched["short"]
    )

    # ==================================================================
    # Long
    # ==================================================================

    long_metrics = calculate_metrics(
        matched["long"]
    )

    # ==================================================================
    # Overall
    # ==================================================================

    overall_pairs = (
        matched["short"]
        + matched["long"]
    )

    overall_metrics = calculate_metrics(
        overall_pairs
    )

    # ==================================================================
    # ROC plot
    # ==================================================================

    plt.figure(
        figsize=(8, 7)
    )

    # --------------------------------------------------------------
    # Short
    # --------------------------------------------------------------

    if matched["short"]:

        short_scores = np.asarray(
            [pair[0] for pair in matched["short"]],
            dtype=float,
        )

        short_human = np.asarray(
            [pair[1] for pair in matched["short"]],
            dtype=int,
        )

        (
            short_fpr,
            short_tpr,
            short_auc,
        ) = _roc_curve_and_auc(
            short_scores,
            short_human,
        )

        if short_auc is not None:

            plt.plot(
                short_fpr,
                short_tpr,
                linewidth=2.5,
                label=(
                    f"Short (AUC = {short_auc:.3f})"
                ),
            )

    # --------------------------------------------------------------
    # Long
    # --------------------------------------------------------------

    if matched["long"]:

        long_scores = np.asarray(
            [pair[0] for pair in matched["long"]],
            dtype=float,
        )

        long_human = np.asarray(
            [pair[1] for pair in matched["long"]],
            dtype=int,
        )

        (
            long_fpr,
            long_tpr,
            long_auc,
        ) = _roc_curve_and_auc(
            long_scores,
            long_human,
        )

        if long_auc is not None:

            plt.plot(
                long_fpr,
                long_tpr,
                linewidth=2.5,
                label=(
                    f"Long (AUC = {long_auc:.3f})"
                ),
            )

    # --------------------------------------------------------------
    # Random baseline
    # --------------------------------------------------------------

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Random (AUC = 0.500)",
    )

    plt.xlabel(
        "False Positive Rate",
        fontsize=12,
    )

    plt.ylabel(
        "True Positive Rate",
        fontsize=12,
    )

    plt.title(
        "Soft-TIFA Score vs Human Evaluation",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    plt.legend(
        loc="lower right",
        frameon=True,
    )

    plt.tight_layout()

    plt.savefig(
        roc_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ==================================================================
    # Return
    # ==================================================================

    return {
        "metric": "soft_tifa_vs_human_score",

        "soft_tifa_score": {
            "field": "score",
            "range": [0, 1],
            "interpretation": (
                "Higher score indicates stronger "
                "text-image alignment."
            ),
        },

        "human_score": {
            "field": "score",
            "values": [0, 1],
            "interpretation": (
                "1 = human judged the generation correct; "
                "0 = incorrect."
            ),
        },

        "short": short_metrics,

        "long": long_metrics,

        "overall": overall_metrics,

        "roc_plot": str(
            roc_plot_path
        ),
    }

from collections import Counter


def _object_signature(
    obj: dict[str, Any],
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
]:
    """
    Extract the attributes used for exact object matching.

    Matches:
        color
        shape
        material
        region

    BBox is intentionally ignored.
    """

    properties = obj.get(
        "properties",
        obj,
    )

    return (
        properties.get("color"),
        properties.get("shape"),
        properties.get("material"),
        obj.get("region", properties.get("region")),
    )


def _objects_match_exactly(
    actual_objects: list[dict[str, Any]],
    perceived_objects: list[dict[str, Any]],
) -> bool:
    """
    Return 1 iff the perceived object list exactly matches
    the actual object list.

    Matching attributes:
        color + shape + material + region

    Object order does not matter.

    If the number of objects differs, return False.
    """

    # --------------------------------------------------------------
    # Different number of objects => automatically incorrect.
    # --------------------------------------------------------------

    if len(actual_objects) != len(perceived_objects):
        return False

    # --------------------------------------------------------------
    # Compare the complete multiset of object attributes.
    # --------------------------------------------------------------

    actual_signatures = Counter(
        _object_signature(obj)
        for obj in actual_objects
    )

    perceived_signatures = Counter(
        _object_signature(obj)
        for obj in perceived_objects
    )
    
    return actual_signatures == perceived_signatures


def _get_objects(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Normalize the object representation.

    Supports:

        objects: [...]

    and:

        objects: {
            "0": {...},
            "1": {...}
        }
    """

    objects = result.get(
        "objects",
        [],
    )

    if isinstance(objects, list):
        return objects

    if isinstance(objects, dict):
        return list(
            objects.values()
        )

    return []


def compute_human_perception_alignment(
    human_res: dict[str, Any],
    perception_res: dict[str, Any],
    roc_plot_path: str | Path,
) -> dict[str, Any]:
    """
    Compute alignment between perception-model scores and human evaluation.

    Human evaluation:
        score = 1 -> correct/aligned
        score = 0 -> incorrect/not aligned

    Perception:
        score = 1 -> correct/aligned
        score = 0 -> incorrect/not aligned

    Metrics:
        - Mean perception score
        - Mean perception score for human=0
        - Mean perception score for human=1
        - Difference between human=1 and human=0 means
        - Exact agreement
        - Point-biserial correlation
        - ROC-AUC

    Also saves one ROC plot containing Short and Long curves.

    Parameters
    ----------
    human_res:
        Loaded human evaluation JSON.

    perception_res:
        Loaded perception evaluation JSON.

    roc_plot_path:
        Path where the ROC PNG should be saved.

    Returns
    -------
    dict
        Alignment statistics for short, long, and overall.
    """

    roc_plot_path = Path(roc_plot_path)

    # ==================================================================
    # Human results lookup
    # ==================================================================

    human_by_key: dict[
        tuple[int, str],
        int,
    ] = {}

    for result in human_res["results"]:

        result_id = int(
            result["id"]
        )

        prompt_field = str(
            result["prompt_field"]
        ).lower()

        human_score = int(
            result["score"]
        )

        human_by_key[
            (result_id, prompt_field)
        ] = {"score": human_score, "objects": _get_objects(result),}

    # ==================================================================
    # Perception results lookup
    # ==================================================================

    perception_by_key: dict[
        tuple[int, str],
        int,
    ] = {}

    for result in perception_res["results"]:

        result_id = int(
            result["id"]
        )

        prompt_field = str(
            result["prompt_field"]
        ).lower()

        perception_score = result.get("score")

        if perception_score is None:
            continue

        perception_by_key[
            (result_id, prompt_field)
        ] = {"score": int(perception_score), "objects": _get_objects(result),}

    # ==================================================================
    # Match human and perception results
    # ==================================================================

    matched: dict[
        str,
        list[tuple[int, int]],
    ] = {
        "short": [],
        "long": [],
    }

    for key, human_result in human_by_key.items():

        perception_result = perception_by_key.get(key)

        if perception_result is None:
            continue

        result_id, prompt_field = key

        if prompt_field not in matched:
            continue

        human_score = human_result["score"]
        perception_score = perception_result["score"]

        actual_objects = human_result["objects"]
        perceived_objects = perception_result["objects"]
        
        perception_correct = _objects_match_exactly(
        actual_objects,
        perceived_objects,
        )

        matched[prompt_field].append(
            (
                perception_score,
                human_score,
                perception_correct,
            )
        )

    # ==================================================================
    # Calculate metrics
    # ==================================================================

    def calculate_metrics(
        pairs: list[tuple[int, int]],
    ) -> dict[str, Any]:

        if not pairs:
            return {
                "n": 0,
                "n_human_0": 0,
                "n_human_1": 0,
                "n_perception_0": 0,
                "n_perception_1": 0,
                "mean_perception_score": None,
                "mean_perception_score_human_0": None,
                "mean_perception_score_human_1": None,
                "mean_perception_difference_1_minus_0": None,
                "exact_agreement": None,
                "point_biserial": None,
                "roc_auc": None,
            }

        perception_scores = np.asarray(
            [pair[0] for pair in pairs],
            dtype=int,
        )

        human_scores = np.asarray(
            [pair[1] for pair in pairs],
            dtype=int,
        )

        # --------------------------------------------------------------
        # Mean perception score
        #
        # Since perception scores are binary, this is simply the
        # proportion of images judged aligned by the perception model.
        # --------------------------------------------------------------

        mean_perception_score = float(
            np.mean(perception_scores)
        )

        # --------------------------------------------------------------
        # Mean perception score for human=0
        #
        # This is the proportion of human-negative examples that the
        # perception model classified as aligned.
        #
        # This is equivalent to the false-positive rate.
        # --------------------------------------------------------------

        perception_human_0 = perception_scores[
            human_scores == 0
        ]

        mean_perception_human_0 = (
            float(np.mean(perception_human_0))
            if len(perception_human_0) > 0
            else None
        )

        # --------------------------------------------------------------
        # Mean perception score for human=1
        #
        # This is the proportion of human-positive examples that the
        # perception model classified as aligned.
        #
        # This is equivalent to the true-positive rate / sensitivity.
        # --------------------------------------------------------------

        perception_human_1 = perception_scores[
            human_scores == 1
        ]

        mean_perception_human_1 = (
            float(np.mean(perception_human_1))
            if len(perception_human_1) > 0
            else None
        )

        # --------------------------------------------------------------
        # Difference between human-positive and human-negative means
        # --------------------------------------------------------------

        if (
            mean_perception_human_0 is not None
            and mean_perception_human_1 is not None
        ):
            mean_difference = (
                mean_perception_human_1
                - mean_perception_human_0
            )
        else:
            mean_difference = None

        # --------------------------------------------------------------
        # Exact agreement
        #
        # Percentage of examples where:
        #
        #     perception_score == human_score
        # --------------------------------------------------------------

        exact_agreement = float(
            np.mean(
                perception_scores == human_scores
            )
        )

        # --------------------------------------------------------------
        # Point-biserial correlation
        # --------------------------------------------------------------

        point_biserial = (
            _point_biserial_correlation(
                perception_scores.astype(float),
                human_scores,
            )
        )

        # --------------------------------------------------------------
        # ROC-AUC
        #
        # Perception score is binary, so AUC is based on the binary
        # ranking produced by the perception model.
        # --------------------------------------------------------------

        _, _, roc_auc = _roc_curve_and_auc(
            perception_scores.astype(float),
            human_scores,
        )

        # --------------------------------------------------------------
        # Confusion matrix
        # --------------------------------------------------------------

        true_positive = int(
            np.sum(
                (perception_scores == 1)
                & (human_scores == 1)
            )
        )

        true_negative = int(
            np.sum(
                (perception_scores == 0)
                & (human_scores == 0)
            )
        )

        false_positive = int(
            np.sum(
                (perception_scores == 1)
                & (human_scores == 0)
            )
        )

        false_negative = int(
            np.sum(
                (perception_scores == 0)
                & (human_scores == 1)
            )
        )

        # --------------------------------------------------------------
        # Precision
        # --------------------------------------------------------------

        precision = (
            true_positive
            / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else None
        )

        # --------------------------------------------------------------
        # Recall / sensitivity
        # --------------------------------------------------------------

        recall = (
            true_positive
            / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else None
        )

        # --------------------------------------------------------------
        # Specificity
        # --------------------------------------------------------------

        specificity = (
            true_negative
            / (true_negative + false_positive)
            if (true_negative + false_positive) > 0
            else None
        )

        # --------------------------------------------------------------
        # F1
        # --------------------------------------------------------------

        f1 = (
            2 * precision * recall / (precision + recall)
            if (
                precision is not None
                and recall is not None
                and (precision + recall) > 0
            )
            else None
        )

        object_matches = np.asarray([pair[2] for pair in pairs],dtype=bool,)

        perception_accuracy = float(np.mean(object_matches))
        
        return {
            "n": len(pairs),

            "n_human_0": int(
                np.sum(human_scores == 0)
            ),

            "n_human_1": int(
                np.sum(human_scores == 1)
            ),

            "n_perception_0": int(
                np.sum(perception_scores == 0)
            ),

            "n_perception_1": int(
                np.sum(perception_scores == 1)
            ),

            "mean_perception_score": (
                mean_perception_score
            ),

            "mean_perception_score_human_0": (
                mean_perception_human_0
            ),

            "mean_perception_score_human_1": (
                mean_perception_human_1
            ),

            "mean_perception_difference_1_minus_0": (
                mean_difference
            ),

            "exact_agreement": (
                exact_agreement
            ),

            "point_biserial": (
                point_biserial
            ),

            "roc_auc": (
                roc_auc
            ),

            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,

            "precision": (
                float(precision)
                if precision is not None
                else None
            ),

            "recall": (
                float(recall)
                if recall is not None
                else None
            ),

            "specificity": (
                float(specificity)
                if specificity is not None
                else None
            ),

            "f1": (
                float(f1)
                if f1 is not None
                else None
            ),
            "perception_accuracy": (perception_accuracy),

            "n_perception_correct": int(np.sum(object_matches)),

            "n_perception_incorrect": int(np.sum(~object_matches)),
        }

    # ==================================================================
    # Short
    # ==================================================================

    short_metrics = calculate_metrics(
        matched["short"]
    )

    # ==================================================================
    # Long
    # ==================================================================

    long_metrics = calculate_metrics(
        matched["long"]
    )

    # ==================================================================
    # Overall
    # ==================================================================

    overall_pairs = (
        matched["short"]
        + matched["long"]
    )

    overall_metrics = calculate_metrics(
        overall_pairs
    )


    # ==================================================================
    # ROC plot
    # ==================================================================

    plt.figure(
        figsize=(8, 7)
    )

    for prompt_field, label in [
        ("short", "Short"),
        ("long", "Long"),
    ]:

        pairs = matched[prompt_field]

        if not pairs:
            continue

        perception_scores = np.asarray(
            [pair[0] for pair in pairs],
            dtype=float,
        )

        human_scores = np.asarray(
            [pair[1] for pair in pairs],
            dtype=int,
        )

        (
            fpr,
            tpr,
            auc,
        ) = _roc_curve_and_auc(
            perception_scores,
            human_scores,
        )

        if auc is None:
            continue

        plt.plot(
            fpr,
            tpr,
            marker="o",
            linewidth=2.5,
            label=(
                f"{label} (AUC = {auc:.3f})"
            ),
        )

    # --------------------------------------------------------------
    # Random baseline
    # --------------------------------------------------------------

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Random (AUC = 0.500)",
    )

    plt.xlabel(
        "False Positive Rate",
        fontsize=12,
    )

    plt.ylabel(
        "True Positive Rate",
        fontsize=12,
    )

    plt.title(
        "Perception Score vs Human Evaluation",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlim(
        0,
        1,
    )

    plt.ylim(
        0,
        1,
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    plt.legend(
        loc="lower right",
        frameon=True,
    )

    plt.tight_layout()

    plt.savefig(
        roc_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ==================================================================
    # Return
    # ==================================================================

    return {
        "metric": ("perception_score_vs_human_score"),

        "perception_score": {
            "field": "score",
            "range": [0, 1],
            "interpretation": (
                "1 = perception model judged the "
                "text-image pair aligned; "
                "0 = not aligned."
            ),
        },

        "human_score": {
            "field": "score",
            "values": [0, 1],
            "interpretation": (
                "1 = human judged the generation correct; "
                "0 = incorrect."
            ),
        },

        "perception_accuracy": {
            "fields": [
                "color",
                "shape",
                "material",
                "region",
            ],
        
            "range": [0, 1],
            "interpretation": (
            "1 = the perceived object list exactly "
            "matches the actual object list in number "
            "and in color, shape, material, and region; "
            "0 = otherwise."
            ),
        },

        "short": short_metrics,

        "long": long_metrics,

        "overall": overall_metrics,

        "roc_plot": str(
        roc_plot_path
        ),
    }

    