"""
Score one image's full `subqa` dict against a VQA backend.

Question order matters: symbols (see answer_spec.py) are resolved against whatever
was bound by an *earlier* question in the same dict, so this walks `subqa` in
insertion order (verbalize.py's dicts are always written in the order they must be
read -- a defining question before anything that refers back to it) rather than
sorting or parallelizing across questions.
"""

from __future__ import annotations

from PIL import Image

from ..common.types import SubQAScore
from .answer_spec import CountSpec, OpenValueSpec, TrueFalseSpec, YesNoSpec, compare, parse_answer
from .base import VQABackend

_YES_NO_CANDIDATES = ["Yes", "No"]
_TRUE_FALSE_CANDIDATES = ["True", "False"]


def _open_value_candidates(expected: str, domain_module) -> list[str]:
    """The forced-choice candidate set for an open-value question, e.g. "blue" ->
    all 8 CLEVR colors. Falls back to just [expected] (an unavoidably easy,
    degenerate forced choice) if `expected` isn't found in the domain vocabulary --
    this only happens if a future verbalize.py answer uses a word outside the
    property/direction vocab, and failing soft rather than crashing keeps one
    unexpected value from taking down a whole scoring run.
    """
    for values in domain_module.PROPERTIES.values():
        if expected in values:
            return list(values)
    if expected in domain_module.DIRECTIONS:
        return list(domain_module.DIRECTIONS)
    return [expected]


def score_subqa(
    image: Image.Image,
    subqa: dict[str, str],
    backend: VQABackend,
    domain_module,
    max_count: int,
) -> list[SubQAScore]:
    # symbol -> the VQA model's own answer to whichever earlier question defined that
    # symbol (its highest-probability candidate -- exactly what the model would have
    # said if asked to just answer with one number). Not a distribution: once a
    # question defines "c", later questions compare against that one received value,
    # the same way you'd read the model's answer off the screen and reuse it by hand.
    symbol_values: dict[str, int] = {}
    results: list[SubQAScore] = []

    for question, raw_answer in subqa.items():
        spec = parse_answer(raw_answer)
        if isinstance(spec, YesNoSpec):
            dist = backend.answer_distribution(image, question, _YES_NO_CANDIDATES)
            score = dist.get("Yes" if spec.expected == "yes" else "No", 0.0)
            results.append(SubQAScore(question, raw_answer, "yes_no", dist, score, False))
            continue

        if isinstance(spec, TrueFalseSpec):
            dist = backend.answer_distribution(image, question, _TRUE_FALSE_CANDIDATES)
            score = dist.get(spec.expected, 0.0)
            results.append(SubQAScore(question, raw_answer, "true_false", dist, score, False))
            continue

        if isinstance(spec, CountSpec):
            candidates = [str(k) for k in range(max_count + 1)]
            dist = backend.answer_distribution(image, question, candidates)
            count_dist = {int(k): v for k, v in dist.items()}
            if spec.binds_symbol:
                symbol_values[spec.binds_symbol] = max(count_dist, key=count_dist.get)
            if spec.op is None:
                # Anchor question ("How many objects are in the image?": "n") -- no
                # expected answer of its own, just defines a symbol for later use.
                results.append(SubQAScore(question, raw_answer, "count_comparison", dist, None, True))
                continue

            rhs = spec.rhs
            if isinstance(rhs, str):
                if rhs not in symbol_values:
                    # Referenced a symbol no earlier question in this dict bound --
                    # a malformed subqa dict, not something to crash the run over.
                    results.append(
                        SubQAScore(question, raw_answer, "count_comparison", dist, None, True)
                    )
                    continue
                rhs = symbol_values[rhs]
                
            score = sum(p for k, p in count_dist.items() if compare(k, spec.op, rhs))
            results.append(SubQAScore(question, raw_answer, "count_comparison", dist, score, False))
            continue

        # OpenValueSpec
        candidates = _open_value_candidates(spec.expected, domain_module)
        dist = backend.answer_distribution(image, question, candidates)
        score = dist.get(spec.expected, 0.0)
        print('Open:', score, dist, candidates )
        results.append(SubQAScore(question, raw_answer, "open_value", dist, score, False))

    return results
