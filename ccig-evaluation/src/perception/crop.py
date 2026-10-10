from __future__ import annotations

from PIL import Image

from .detectors.base import BBox


def crop_and_neutralize(
    image: Image.Image,
    bbox: BBox,
    bg_color: tuple[int, int, int] = (245, 245, 245),
    pad_frac: float = 0.30, #0.15
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

def crop_with_padding(
    image: Image.Image,
    bbox: BBox,
    pad_frac: float = 0.15,
    bg_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    """
    Crop to bbox with fractional padding.

    If bg_color is None, the original image is used and the padded
    area is taken from the image where possible.

    If bg_color is specified, the bbox crop is placed on a solid
    background.
    """
    image = image.convert("RGB")

    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1

    crop = image.crop((x0, y0, x1, y1))

    pad_x = int(crop.width * pad_frac)
    pad_y = int(crop.height * pad_frac)

    if bg_color is None:
        # Expand the crop within image boundaries.
        new_x0 = max(0, x0 - pad_x)
        new_y0 = max(0, y0 - pad_y)
        new_x1 = min(image.width, x1 + pad_x)
        new_y1 = min(image.height, y1 + pad_y)

        return image.crop((new_x0, new_y0, new_x1, new_y1))

    canvas = Image.new(
        "RGB",
        (crop.width + 2 * pad_x, crop.height + 2 * pad_y),
        bg_color,
    )
    canvas.paste(crop, (pad_x, pad_y))

    return canvas

def make_object_views(
    image: Image.Image,
    bbox: BBox,
) -> dict[str, Image.Image]:

    return {
        # Small context, useful for shape.
        "tight": crop_with_padding(
            image,
            bbox,
            pad_frac=0.05,
            bg_color=(245, 245, 245),
        ),

        # Your current preprocessing.
        "medium": crop_with_padding(
            image,
            bbox,
            pad_frac=0.15,
            bg_color=(245, 245, 245),
        ),

        # More context, useful when the detector box is slightly tight.
        "wide": crop_with_padding(
            image,
            bbox,
            pad_frac=0.30,
            bg_color=(245, 245, 245),
        ),

        # Original image context. Particularly useful for material.
        "rgb": crop_with_padding(
            image,
            bbox,
            pad_frac=0.15,
            bg_color=None,
        ),
    }