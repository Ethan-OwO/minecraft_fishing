"""合成「仿 Minecraft 字幕」畫面，用來在沒有遊戲的環境下測試偵測器。

刻意模仿原版 SubtitleOverlay 的幾個關鍵行為：
  - GUI 以整數倍放大（最近鄰），像素字邊緣銳利
  - 背景框是 80% 不透明的黑色，會透出一點後面的遊戲畫面
  - 文字不透明，顏色是灰階 gray（新字幕 255，淡出到 75），陰影為 1/4 亮度、往右下偏移 1px
  - 多行字幕從右下角往上堆疊，框寬由最長那行決定，文字置中
"""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 1.0
PAD = 2  # 框內邊距（GUI 像素）


def _text_size(text: str) -> tuple[int, int]:
    (w, h), base = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
    return w, h + base


def _glyph_mask(text: str, org: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    """把文字畫成「二值」遮罩。

    OpenCV 5 的 putText 一律反鋸齒（LINE_8 不再生效），邊緣會有半透明灰階，
    跟 Minecraft 銳利的點陣字不同。所以先畫到空白圖上再二值化，得到像素字形。
    """
    canvas = np.zeros(shape, np.uint8)
    cv2.putText(canvas, text, org, FONT, FONT_SCALE, 255, 1, cv2.LINE_8)
    return canvas >= 128


def make_scene(
    lines: list[tuple[str, int]],
    gui_size: tuple[int, int] = (120, 220),
    gui_scale: int = 2,
    world: np.ndarray | int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """lines：由下往上的 (文字, 灰階)。回傳 (灰階畫面, {文字: 螢幕座標 (x, y, w, h)})。"""
    gh, gw = gui_size
    rng = np.random.default_rng(seed)
    if world is None:
        # 有起伏的「遊戲畫面」：漸層 + 雜訊
        yy, xx = np.mgrid[0:gh, 0:gw]
        world = 80 + 60 * np.sin(xx / 17.0) * np.cos(yy / 11.0) + rng.normal(0, 20, (gh, gw))
    elif np.isscalar(world):
        world = np.full((gh, gw), float(world))
    scene = np.clip(np.asarray(world, dtype=np.float64), 0, 255)

    box_w = max(_text_size(t)[0] for t, _ in lines) + 2 * PAD
    line_h = max(_text_size(t)[1] for t, _ in lines) + 2 * PAD
    right, bottom = gw - 2, gh - 20

    boxes: dict[str, tuple[int, int, int, int]] = {}
    for i, (text, gray) in enumerate(lines):
        y1 = bottom - i * (line_h + 1)
        y0 = y1 - line_h
        x0 = right - box_w
        scene[y0:y1, x0:right] *= 0.2  # 80% 黑色背景框
        tw, _ = _text_size(text)
        tx = x0 + (box_w - tw) // 2
        # putText 的 org 是「基線」位置（文字左下角，不含下伸部），
        # 所以基線 = 框頂 + 邊距 + 基線以上的字高。
        (_, ascent), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
        baseline_y = y0 + PAD + ascent
        glyphs = _glyph_mask(text, (tx, baseline_y), scene.shape)
        shadow = np.roll(np.roll(glyphs, 1, axis=0), 1, axis=1)
        scene[shadow] = gray // 4  # 陰影先畫，文字蓋在上面
        scene[glyphs] = gray  # 不透明：直接覆蓋，不和背景混色
        boxes[text] = (tx * gui_scale, y0 * gui_scale, (tw + 1) * gui_scale, line_h * gui_scale)

    img = scene.astype(np.uint8)
    img = cv2.resize(img, (gw * gui_scale, gh * gui_scale), interpolation=cv2.INTER_NEAREST)
    return img, boxes


def make_template(text: str = "Fishing Bobber splashes", gui_scale: int = 2, seed: int = 1) -> np.ndarray:
    """模擬使用者校正：在新字幕畫面上框出這行文字（含一點邊距）。"""
    img, boxes = make_scene([(text, 255)], gui_scale=gui_scale, seed=seed)
    x, y, w, h = boxes[text]
    return img[y : y + h, x : x + w].copy()
