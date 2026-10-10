from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class VQABackend(ABC):
    """A VQA model that answers a forced-choice question about an image with a
    per-candidate probability, for soft-TIFA scoring."""

    name: str

    @abstractmethod
    def answer_distribution(
        self, image: Image.Image, question: str, candidates: list[str]
    ) -> dict[str, float]:
        """Ask `question` about `image`, forced to a single next generated token, and
        return {candidate: probability} -- each candidate's raw softmax probability
        mass from the model's *full* vocabulary distribution over that one token
        (summed across the candidate's tokenizations, e.g. "Yes" and " Yes").

        Deliberately NOT renormalized to sum to 1 over just `candidates`: if the model
        spreads probability mass onto tokens outside the candidate set (hedging,
        punctuation, an unexpected continuation), that uncertainty should show up as
        low scores for every candidate rather than being hidden by renormalization.
        This mirrors soft-TIFA's own description: the score *is* "the VQA model's
        probability assigned to the correct answer", not a renormalized posterior.
        """
        raise NotImplementedError
