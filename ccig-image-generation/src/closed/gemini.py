from __future__ import annotations

import os
from io import BytesIO

from PIL import Image

from .base import ClosedImageModel


class GeminiFlashImage(ClosedImageModel):
    name = "gemini-2.0-flash"
    api_model = "gemini-2.5-flash-image"

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> Image.Image:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.api_model,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return Image.open(BytesIO(part.inline_data.data)).convert("RGB")
        raise RuntimeError("Gemini response did not contain image data.")


class Gemini31FlashImage(ClosedImageModel):
    """Google's Gemini 3.1 Flash Image, the cheaper/faster sibling of gemini-3-pro-image.

    Weaker prompt-following than the Pro tier but ~2x cheaper at 1K -- the closest Google
    equivalent to gpt-image-2's "low" quality tier (Nano Banana Pro has no such tier: its
    1K and 2K resolutions are priced the same).

    NOTE: same caveat as Gemini3ProImage -- request shape pieced together from docs that
    post-date this assistant's training data, not verified against a live call. Smoke-test
    with a single prompt first; if `image_config`/`ImageConfig` is rejected, drop it.
    """

    name = "gemini-3.1-flash-image"
    api_model = "gemini-3.1-flash-image-preview"
    image_size = "1K"  # 512px | 1K | 2K | 4K -- see README pricing note

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> Image.Image:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.api_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(image_size=self.image_size),
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return Image.open(BytesIO(part.inline_data.data)).convert("RGB")
        raise RuntimeError("Gemini response did not contain image data.")


class Gemini3ProImage(ClosedImageModel):
    """Google's Gemini 3 Pro Image ("Nano Banana Pro"), successor to gemini-2.5-flash-image.

    NOTE: the exact request shape for this model was pieced together from current docs
    (post-dates this assistant's training data) rather than verified against a live call.
    Smoke-test with a single prompt before trusting it at scale -- if `image_size` in
    GenerateContentConfig is rejected, drop it and let the model use its default resolution.
    """

    name = "gemini-3-pro-image"
    api_model = "gemini-3-pro-image"
    image_size = "1K"  # 1K | 2K | 4K -- NOTE: 1K and 2K are priced the same on this model
    # (only 4K costs more), so unlike gpt-image-2's `quality = "low"` this does NOT reduce
    # cost -- it only shrinks output resolution/file size. For an actually cheaper run, use
    # the Gemini31FlashImage model instead. See README pricing note.

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> Image.Image:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.api_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(image_size=self.image_size),
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return Image.open(BytesIO(part.inline_data.data)).convert("RGB")
        raise RuntimeError("Gemini response did not contain image data.")
