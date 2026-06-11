"""Visualization utilities for detection results."""

from __future__ import annotations

import logging

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Default color palette for up to 8 classes
COLORS_RGB = [
    (255, 59, 48),
    (255, 204, 0),
    (255, 149, 0),
    (255, 214, 10),
    (175, 82, 222),
    (52, 199, 89),
    (90, 200, 250),
    (164, 28, 48),
]


def load_label_font(size: int, font_path: str = "") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font, falling back to the default bitmap font."""
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            logger.warning("Could not load font: %s", font_path)
    return ImageFont.load_default()


def draw_detections(
    img: Image.Image,
    boxes: list[list[float]],
    labels: list[str],
    scores: list[float],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
) -> Image.Image:
    """Draw bounding boxes and labels on an image.

    Args:
        img: PIL Image to annotate (modified in-place)
        boxes: list of ``[x1, y1, x2, y2]`` boxes
        labels: list of class name strings
        scores: list of confidence scores
        font: optional font for labels

    Returns:
        The annotated image
    """
    draw = ImageDraw.Draw(img)
    orig_h = img.height

    if font is None:
        font = load_label_font(max(12, int(orig_h * 0.02)))

    thickness = max(2, int(orig_h * 0.005))

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box
        color = COLORS_RGB[i % len(COLORS_RGB)]

        for t in range(thickness):
            draw.rectangle([x1 + t, y1 + t, x2 - t, y2 - t], outline=color)

        label_text = f"{label} {score:.0%}"
        if hasattr(draw, "textbbox"):
            text_w, text_h = draw.textbbox((0, 0), label_text, font=font)[2:4]
        else:
            text_w, text_h = draw.textsize(label_text, font=font)  # type: ignore[attr-defined]

        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 6, y1], fill=color)
        draw.text(
            (x1 + 3, y1 - text_h - 2),
            label_text,
            fill=(255, 255, 255),
            font=font,
        )

    return img
