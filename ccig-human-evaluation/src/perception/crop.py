from __future__ import annotations

from PIL import Image

from .detectors.base import BBox


def crop_and_neutralize(
    image: Image.Image,
    bbox: BBox,
    bg_color: tuple[int, int, int] = (245, 245, 245),
    pad_frac: float = 0.15,
) -> Image.Image:
    """Crop to bbox, padded on a solid background -- gives CLIP a little context
    around the object without letting neighboring objects dominate the crop.

    bg_color defaults to off-white (245,245,245), not the gray CLEVR-perception
    papers typically use -- CLEVR's own color vocabulary includes "gray", so a gray
    background would bias the color classifier toward/against that label.

    No segmentation mask is available from a box-only detector, so pixels inside the
    bbox itself are not masked to the object's silhouette -- a known limitation,
    worth revisiting if a segmentation-capable detector is swapped in.
    """
    crop = image.crop((bbox.x0, bbox.y0, bbox.x1, bbox.y1)).convert("RGB")
    pad_x, pad_y = int(crop.width * pad_frac), int(crop.height * pad_frac)
    canvas = Image.new("RGB", (crop.width + 2 * pad_x, crop.height + 2 * pad_y), bg_color)
    canvas.paste(crop, (pad_x, pad_y))
    return canvas
