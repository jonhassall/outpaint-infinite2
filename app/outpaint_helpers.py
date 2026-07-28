from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
from PIL import Image


class Direction(str, Enum):
    left = "left"
    right = "right"
    up = "up"
    down = "down"


@dataclass(frozen=True)
class PreparedSource:
    condition: Image.Image
    placed_source: Image.Image
    canvas_size: tuple[int, int]
    bbox: tuple[int, int, int, int]
    seam_px: int

    @property
    def bbox_normalized(self) -> list[float]:
        width, height = self.canvas_size
        x0, y0, x1, y1 = self.bbox
        return [x0 / width, y0 / height, x1 / width, y1 / height]


@dataclass(frozen=True)
class CanvasPlan:
    source: Image.Image
    canvas_size: tuple[int, int]
    bbox: tuple[int, int, int, int]
    direction: Direction
    requested_expand: int
    actual_expand: int
    source_was_resized: bool


def align_up(value: int, alignment: int = 16) -> int:
    return max(alignment, int(math.ceil(value / alignment) * alignment))


def align_down(value: int, alignment: int = 16) -> int:
    return max(alignment, int(math.floor(value / alignment) * alignment))


def resize_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    if max(image.size) <= max_edge:
        return image.copy()
    scale = max_edge / max(image.size)
    size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def _resize_for_horizontal_growth(
    source: Image.Image, expand: int, max_canvas: int
) -> tuple[Image.Image, int, int]:
    ratio = source.width / source.height
    max_source_w = max(16, max_canvas - expand)
    max_height_from_width = max_source_w / ratio
    target_h = min(source.height, max_canvas, max_height_from_width)
    target_h = align_down(int(target_h))
    target_w = max(16, int(round(target_h * ratio)))

    # Rounding can still push the final canvas over the cap. Reduce one block at a time.
    while align_up(target_w + expand) > max_canvas and target_h > 16:
        target_h -= 16
        target_w = max(16, int(round(target_h * ratio)))

    resized = source.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return resized, target_w, target_h


def _resize_for_vertical_growth(
    source: Image.Image, expand: int, max_canvas: int
) -> tuple[Image.Image, int, int]:
    ratio = source.width / source.height
    max_source_h = max(16, max_canvas - expand)
    max_width_from_height = max_source_h * ratio
    target_w = min(source.width, max_canvas, max_width_from_height)
    target_w = align_down(int(target_w))
    target_h = max(16, int(round(target_w / ratio)))

    while align_up(target_h + expand) > max_canvas and target_w > 16:
        target_w -= 16
        target_h = max(16, int(round(target_w / ratio)))

    resized = source.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return resized, target_w, target_h


def plan_directional_canvas(
    source: Image.Image,
    direction: Direction | str,
    expand_pixels: int,
    max_canvas: int,
) -> CanvasPlan:
    direction = Direction(direction)
    source = source.convert("RGB")
    expand = align_up(max(16, int(expand_pixels)))

    if max_canvas < 64:
        raise ValueError("max_canvas must be at least 64")
    if expand >= max_canvas:
        raise ValueError("expand_pixels must be smaller than max_canvas")

    original_size = source.size

    if direction in {Direction.left, Direction.right}:
        source, sw, sh = _resize_for_horizontal_growth(source, expand, max_canvas)
        canvas_w = align_up(sw + expand)
        canvas_h = sh  # source spans the full canvas height
        actual_expand = canvas_w - sw
        if direction is Direction.left:
            bbox = (actual_expand, 0, actual_expand + sw, sh)
        else:
            bbox = (0, 0, sw, sh)
    else:
        source, sw, sh = _resize_for_vertical_growth(source, expand, max_canvas)
        canvas_w = sw  # source spans the full canvas width
        canvas_h = align_up(sh + expand)
        actual_expand = canvas_h - sh
        if direction is Direction.up:
            bbox = (0, actual_expand, sw, actual_expand + sh)
        else:
            bbox = (0, 0, sw, sh)

    if canvas_w % 16 or canvas_h % 16:
        raise AssertionError("canvas dimensions must be multiples of 16")
    if max(canvas_w, canvas_h) > max_canvas:
        raise AssertionError("planned canvas exceeds max_canvas")

    return CanvasPlan(
        source=source,
        canvas_size=(canvas_w, canvas_h),
        bbox=bbox,
        direction=direction,
        requested_expand=int(expand_pixels),
        actual_expand=actual_expand,
        source_was_resized=source.size != original_size,
    )


def prepare_source(
    source: Image.Image,
    canvas_size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    source_max_edge: int = 384,
    seam_px: int = 32,
) -> PreparedSource:
    width, height = canvas_size
    x0, y0, x1, y1 = bbox
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise ValueError("Canvas dimensions must be positive multiples of 16")
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Source bbox is outside the canvas: {bbox}")

    source = source.convert("RGB")
    box_w, box_h = x1 - x0, y1 - y0
    source_ratio = source.width / source.height
    box_ratio = box_w / box_h
    tolerance = max(0.025, 2.0 / min(box_w, box_h))
    if abs(box_ratio / source_ratio - 1.0) > tolerance:
        raise ValueError("Source bbox must preserve the source image aspect ratio")

    placed = source.resize((box_w, box_h), Image.Resampling.LANCZOS)
    return PreparedSource(
        condition=resize_max_edge(placed, source_max_edge),
        placed_source=placed,
        canvas_size=canvas_size,
        bbox=bbox,
        seam_px=max(0, int(seam_px)),
    )


def composite(generated: Image.Image, prepared: PreparedSource) -> Image.Image:
    generated = generated.convert("RGB")
    if generated.size != prepared.canvas_size:
        raise ValueError("Generated image size does not match the canvas")

    output = np.asarray(generated, dtype=np.float32).copy()
    source = np.asarray(prepared.placed_source, dtype=np.float32)
    x0, y0, x1, y1 = prepared.bbox
    height, width = source.shape[:2]

    if prepared.seam_px <= 0:
        alpha = np.ones((height, width), dtype=np.float32)
    else:
        yy, xx = np.mgrid[:height, :width]
        edge_distance = np.minimum.reduce(
            [xx, width - 1 - xx, yy, height - 1 - yy]
        ).astype(np.float32)
        alpha = np.clip(edge_distance / float(prepared.seam_px), 0.0, 1.0)
        # Smoothstep avoids a visibly linear seam.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

    alpha = alpha[..., None]
    region = output[y0:y1, x0:x1]
    output[y0:y1, x0:x1] = source * alpha + region * (1.0 - alpha)
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")
