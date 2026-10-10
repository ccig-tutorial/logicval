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
    
) -> dict[str, Any]:
    """
    Compute alignment between human evaluation and CLIP cosine
    for short prompts only.

    Human evaluation:
        score = 1 -> correct
        score = 0 -> incorrect

    CLIP:
        cosine in [-1, 1]
        higher cosine = stronger text-image alignment.

    Metrics:
        - Mean cosine
        - Mean cosine for human=0
        - Mean cosine for human=1
        - Difference between human=1 and human=0 means
        - Point-biserial correlation
        - ROC-AUC
    """

    # ==================================================================
    # Build human lookup
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

        # Only short prompts.
        if prompt_field != "short":
            continue

        human_by_key[
            (result_id, prompt_field)
        ] = int(
            result["score"]
        )

    # ==================================================================
    # Build CLIP lookup
    # ==================================================================

    clip_by_key: dict[
        tuple[int, str],
        float,
    ] = {}

    for result in clip_res["results"]:

        result_id = int(
            result["id"]
        )

        prompt_field = str(
            result["prompt_field"]
        ).lower()

        # Only short prompts.
        if prompt_field != "short":
            continue

        cosine = result.get("cosine")

        if cosine is None:
            continue

        clip_by_key[
            (result_id, prompt_field)
        ] = float(cosine)

    # ==================================================================
    # Match human and CLIP results
    # ==================================================================

    pairs: list[
        tuple[float, int]
    ] = []

    for key, human_score in human_by_key.items():

        clip_score = clip_by_key.get(key)

        if clip_score is None:
            continue

        pairs.append(
            (
                clip_score,
                human_score,
            )
        )

    # ==================================================================
    # No matched results
    # ==================================================================

    if not pairs:
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

    # ==================================================================
    # Convert to arrays
    # ==================================================================

    cosine_scores = np.asarray(
        [pair[0] for pair in pairs],
        dtype=float,
    )

    human_scores = np.asarray(
        [pair[1] for pair in pairs],
        dtype=int,
    )

    # ==================================================================
    # Mean cosine
    # ==================================================================

    mean_cosine = float(
        np.mean(cosine_scores)
    )

    # ==================================================================
    # Mean cosine for human=0
    # ==================================================================

    cosine_human_0 = cosine_scores[
        human_scores == 0
    ]

    mean_cosine_human_0 = (
        float(
            np.mean(cosine_human_0)
        )
        if len(cosine_human_0) > 0
        else None
    )

    # ==================================================================
    # Mean cosine for human=1
    # ==================================================================

    cosine_human_1 = cosine_scores[
        human_scores == 1
    ]

    mean_cosine_human_1 = (
        float(
            np.mean(cosine_human_1)
        )
        if len(cosine_human_1) > 0
        else None
    )

    # ==================================================================
    # Difference between human-positive and human-negative means
    # ==================================================================

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

    # ==================================================================
    # Point-biserial correlation
    # ==================================================================

    point_biserial = (
        _point_biserial_correlation(
            cosine_scores,
            human_scores,
        )
    )

    # ==================================================================
    # ROC-AUC
    # ==================================================================

    _, _, roc_auc = _roc_curve_and_auc(
        cosine_scores,
        human_scores,
    )

    # ==================================================================
    # Return
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


def compute_human_vlm_alignment(
    human_res: dict[str, Any],
    vlm_judge_res: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute alignment between VLM-judge scores and human evaluation
    for short prompts only.

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
    """

    # ==================================================================
    # Build human lookup
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

        # Only short prompts.
        if prompt_field != "short":
            continue

        human_by_key[
            (result_id, prompt_field)
        ] = int(
            result["score"]
        )

    # ==================================================================
    # Build VLM lookup
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

        # Only short prompts.
        if prompt_field != "short":
            continue

        vlm_score = result.get("score")

        if vlm_score is None:
            continue

        vlm_by_key[
            (result_id, prompt_field)
        ] = float(vlm_score)

    # ==================================================================
    # Match human and VLM results
    # ==================================================================

    pairs: list[
        tuple[float, int]
    ] = []

    for key, human_score in human_by_key.items():

        vlm_score = vlm_by_key.get(key)

        if vlm_score is None:
            continue

        pairs.append(
            (
                vlm_score,
                human_score,
            )
        )

    # ==================================================================
    # No matched results
    # ==================================================================

    if not pairs:
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

    # ==================================================================
    # Convert to arrays
    # ==================================================================

    vlm_scores = np.asarray(
        [pair[0] for pair in pairs],
        dtype=float,
    )

    human_scores = np.asarray(
        [pair[1] for pair in pairs],
        dtype=int,
    )

    # ==================================================================
    # Mean VLM score
    # ==================================================================

    mean_vlm_score = float(
        np.mean(vlm_scores)
    )

    # ==================================================================
    # Mean VLM score for human=0
    # ==================================================================

    vlm_human_0 = vlm_scores[
        human_scores == 0
    ]

    mean_vlm_human_0 = (
        float(
            np.mean(vlm_human_0)
        )
        if len(vlm_human_0) > 0
        else None
    )

    # ==================================================================
    # Mean VLM score for human=1
    # ==================================================================

    vlm_human_1 = vlm_scores[
        human_scores == 1
    ]

    mean_vlm_human_1 = (
        float(
            np.mean(vlm_human_1)
        )
        if len(vlm_human_1) > 0
        else None
    )

    # ==================================================================
    # Difference between human-positive and human-negative means
    # ==================================================================

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

    # ==================================================================
    # Point-biserial correlation
    # ==================================================================

    point_biserial = (
        _point_biserial_correlation(
            vlm_scores,
            human_scores,
        )
    )

    # ==================================================================
    # ROC-AUC
    # ==================================================================

    _, _, roc_auc = _roc_curve_and_auc(
        vlm_scores,
        human_scores,
    )

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


def compute_human_soft_alignment(
    human_res: dict[str, Any],
    soft_tifa_res: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute alignment between Soft-TIFA scores and human evaluation
    for short prompts only.

    Human evaluation:
        score = 1 -> correct
        score = 0 -> incorrect

    Soft-TIFA:
        score in [0, 1]
        higher score = stronger text-image alignment.

    Metrics:
        - Mean Soft-TIFA score
        - Mean Soft-TIFA score for human=0
        - Mean Soft-TIFA score for human=1
        - Difference between human=1 and human=0 means
        - Point-biserial correlation
        - ROC-AUC
    """

    # ==================================================================
    # Build human lookup
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

        # Only short prompts.
        if prompt_field != "short":
            continue

        human_by_key[
            (result_id, prompt_field)
        ] = int(
            result["score"]
        )

    # ==================================================================
    # Build Soft-TIFA lookup
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

        # Only short prompts.
        if prompt_field != "short":
            continue

        soft_tifa_score = result.get(
            "score_am"
        )

        if soft_tifa_score is None:
            continue

        soft_tifa_by_key[
            (result_id, prompt_field)
        ] = float(
            soft_tifa_score
        )

    # ==================================================================
    # Match human and Soft-TIFA results
    # ==================================================================

    pairs: list[
        tuple[float, int]
    ] = []

    for key, human_score in human_by_key.items():

        soft_tifa_score = soft_tifa_by_key.get(
            key
        )

        if soft_tifa_score is None:
            continue

        pairs.append(
            (
                soft_tifa_score,
                human_score,
            )
        )

    # ==================================================================
    # No matched results
    # ==================================================================

    if not pairs:
        return {
            "metric": "soft_tifa_vs_human_score",

            "soft_tifa_score": {
                "field": "score_am",
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

    # ==================================================================
    # Convert to arrays
    # ==================================================================

    soft_tifa_scores = np.asarray(
        [pair[0] for pair in pairs],
        dtype=float,
    )

    human_scores = np.asarray(
        [pair[1] for pair in pairs],
        dtype=int,
    )

    # ==================================================================
    # Mean Soft-TIFA score
    # ==================================================================

    mean_soft_tifa_score = float(
        np.mean(soft_tifa_scores)
    )

    # ==================================================================
    # Mean Soft-TIFA score for human=0
    # ==================================================================

    soft_tifa_human_0 = soft_tifa_scores[
        human_scores == 0
    ]

    mean_soft_tifa_human_0 = (
        float(
            np.mean(soft_tifa_human_0)
        )
        if len(soft_tifa_human_0) > 0
        else None
    )

    # ==================================================================
    # Mean Soft-TIFA score for human=1
    # ==================================================================

    soft_tifa_human_1 = soft_tifa_scores[
        human_scores == 1
    ]

    mean_soft_tifa_human_1 = (
        float(
            np.mean(soft_tifa_human_1)
        )
        if len(soft_tifa_human_1) > 0
        else None
    )

    # ==================================================================
    # Difference between human-positive and human-negative means
    # ==================================================================

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

    # ==================================================================
    # Point-biserial correlation
    # ==================================================================

    point_biserial = (
        _point_biserial_correlation(
            soft_tifa_scores,
            human_scores,
        )
    )

    # ==================================================================
    # ROC-AUC
    # ==================================================================

    _, _, roc_auc = _roc_curve_and_auc(
        soft_tifa_scores,
        human_scores,
    )

    # ==================================================================
    # Return
    # ==================================================================

    return {
        "metric": "soft_tifa_vs_human_score",

        "soft_tifa_score": {
            "field": "score_am",
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

        "n": len(pairs),

        "n_human_0": int(
            np.sum(human_scores == 0)
        ),

        "n_human_1": int(
            np.sum(human_scores == 1)
        ),

        "mean_soft_tifa_score": (
            mean_soft_tifa_score
        ),

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
    Compute alignment between perception-model scores and human evaluation
    for short prompts only.

    Human evaluation:
        score = 1 -> correct/aligned
        score = 0 -> incorrect/not aligned

    Perception:
        score = 1 -> correct/aligned
        score = 0 -> incorrect/not aligned

    Object-level perception accuracy:
        The perceived object list must have the same number of objects
        as the actual object list, and every object must match in:

            - color
            - shape
            - material
            - region

        If the object count differs, perception correctness = 0.

    Metrics:
        - Mean perception score
        - Mean perception score for human=0
        - Mean perception score for human=1
        - Difference between human=1 and human=0 means
        - Exact agreement
        - Confusion matrix
        - Precision
        - Recall
        - Specificity
        - F1
        - Point-biserial correlation
        - ROC-AUC
        - Object-level perception accuracy

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
        Alignment statistics for short prompts.
    """

    roc_plot_path = Path(roc_plot_path)

    # ==================================================================
    # Human results lookup
    # ==================================================================

    human_by_key: dict[
        tuple[int, str],
        dict[str, Any],
    ] = {}

    for result in human_res["results"]:

        result_id = int(result["id"])

        prompt_field = str(
            result["prompt_field"]
        ).lower()

        if prompt_field != "short":
            continue

        human_by_key[
            (result_id, prompt_field)
        ] = {
            "score": int(result["score"]),
            "objects": _get_objects(result),
        }

    # ==================================================================
    # Perception results lookup
    # ==================================================================

    perception_by_key: dict[
        tuple[int, str],
        dict[str, Any],
    ] = {}

    for result in perception_res["results"]:

        result_id = int(result["id"])

        prompt_field = str(
            result["prompt_field"]
        ).lower()

        if prompt_field != "short":
            continue

        perception_score = result.get("score")

        if perception_score is None:
            continue

        perception_by_key[
            (result_id, prompt_field)
        ] = {
            "score": int(perception_score),
            "objects": _get_objects(result),
        }

    # ==================================================================
    # Match human and perception results
    # ==================================================================

    matched: list[
        tuple[int, int, bool]
    ] = []

    for key, human_result in human_by_key.items():

        perception_result = perception_by_key.get(key)

        if perception_result is None:
            continue

        human_score = human_result["score"]
        perception_score = perception_result["score"]

        actual_objects = human_result["objects"]
        perceived_objects = perception_result["objects"]

        # --------------------------------------------------------------
        # Object-level perception correctness
        # --------------------------------------------------------------

        perception_correct = _objects_match_exactly(
            actual_objects,
            perceived_objects,
        )

        matched.append(
            (
                perception_score,
                human_score,
                perception_correct,
            )
        )

    # ==================================================================
    # Calculate metrics
    # ==================================================================

    if not matched:
        metrics = {
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
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "precision": None,
            "recall": None,
            "specificity": None,
            "f1": None,
            "perception_accuracy": None,
            "n_perception_correct": 0,
            "n_perception_incorrect": 0,
        }

    else:

        perception_scores = np.asarray(
            [pair[0] for pair in matched],
            dtype=int,
        )

        human_scores = np.asarray(
            [pair[1] for pair in matched],
            dtype=int,
        )

        # --------------------------------------------------------------
        # Mean perception score
        # --------------------------------------------------------------

        mean_perception_score = float(
            np.mean(perception_scores)
        )

        # --------------------------------------------------------------
        # Mean perception score for human=0
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
        # Difference
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
        # Recall
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

        # --------------------------------------------------------------
        # Object-level perception accuracy
        # --------------------------------------------------------------

        object_matches = np.asarray(
            [pair[2] for pair in matched],
            dtype=bool,
        )

        perception_accuracy = float(
            np.mean(object_matches)
        )

        n_perception_correct = int(
            np.sum(object_matches)
        )

        n_perception_incorrect = int(
            np.sum(~object_matches)
        )

        metrics = {
            "n": len(matched),

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

            "true_positive": (
                true_positive
            ),

            "true_negative": (
                true_negative
            ),

            "false_positive": (
                false_positive
            ),

            "false_negative": (
                false_negative
            ),

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

            "perception_accuracy": (
                perception_accuracy
            ),

            "n_perception_correct": (
                n_perception_correct
            ),

            "n_perception_incorrect": (
                n_perception_incorrect
            ),
        }

    # ==================================================================
    # ROC plot
    # ==================================================================

    plt.figure(
        figsize=(8, 7)
    )

    if matched:

        perception_scores = np.asarray(
            [pair[0] for pair in matched],
            dtype=float,
        )

        human_scores = np.asarray(
            [pair[1] for pair in matched],
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

        if auc is not None:

            plt.plot(
                fpr,
                tpr,
                marker="o",
                linewidth=2.5,
                label=f"Short (AUC = {auc:.3f})",
            )

    # ==================================================================
    # Random baseline
    # ==================================================================

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
        "metric": (
            "perception_score_vs_human_score"
        ),

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

        "short": metrics,

        "roc_plot": str(
            roc_plot_path
        ),
    }


