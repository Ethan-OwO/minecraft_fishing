import cv2
import numpy as np
import pytest
from synth import make_scene, make_template

from detector import SubtitleDetector, split_segments, text_mask

SPLASH = "Fishing Bobber splashes"


@pytest.fixture(scope="module", params=[2, 3], ids=lambda s: f"gui{s}")
def gui_scale(request):
    return request.param


@pytest.fixture(scope="module")
def detector(gui_scale):
    return SubtitleDetector(make_template(SPLASH, gui_scale=gui_scale))


def test_fresh_subtitle_is_a_bite(detector, gui_scale):
    img, boxes = make_scene([(SPLASH, 255)], gui_scale=gui_scale)
    det = detector.detect(img)
    assert det.bite
    assert det.score > 0.95
    assert det.brightness == pytest.approx(255, abs=1)
    assert det.loc == boxes[SPLASH][:2]


@pytest.mark.parametrize("world", [None, 0, 255], ids=["textured", "black", "white-sky"])
def test_found_among_stacked_subtitles(detector, gui_scale, world):
    # 字幕往上堆疊、框寬由最長那行決定、文字置中 → 位置會變，但滑動比對仍找得到
    lines = [("Zombie groans", 255), (SPLASH, 250), ("Water flows", 180)]
    img, boxes = make_scene(lines, gui_scale=gui_scale, world=world, seed=7)
    det = detector.detect(img)
    assert det.bite
    assert det.loc == boxes[SPLASH][:2]


@pytest.mark.parametrize("gray", [255, 230, 210])
def test_recent_subtitle_is_fresh(detector, gui_scale, gray):
    img, _ = make_scene([(SPLASH, gray)], gui_scale=gui_scale)
    det = detector.detect(img)
    assert det.matched and det.fresh


@pytest.mark.parametrize("gray", [190, 150, 120])
def test_old_subtitle_still_matches_but_is_not_fresh(detector, gui_scale, gray):
    """NCC 對淡出不敏感（仍 matched），所以才需要亮度判斷新鮮度。"""
    img, _ = make_scene([(SPLASH, gray)], gui_scale=gui_scale)
    det = detector.detect(img)
    assert det.matched
    assert not det.fresh
    assert not det.bite
    assert det.brightness == pytest.approx(gray, abs=1)


@pytest.mark.parametrize(
    "text",
    [
        "Fishing Bobber thrown",
        "Fishing Bobber retrieved",  # 和目標共用前綴，最容易誤判
        "Bobber splashes",
        "Water splashes",
        "Splashing",
    ],
)
def test_other_subtitles_do_not_match(detector, gui_scale, text):
    img, _ = make_scene([(text, 255)], gui_scale=gui_scale, seed=5)
    det = detector.detect(img)
    assert not det.matched
    assert det.score < detector.match_threshold - 0.2  # 留足夠餘裕


def test_segmented_matching_beats_single_ncc_on_shared_prefix(gui_scale):
    """分段取最小值，能大幅壓低「共用前綴」字幕的分數，而真正的匹配分數幾乎不變。"""
    template = make_template(SPLASH, gui_scale=gui_scale)
    single = SubtitleDetector(template, segments=1)
    segmented = SubtitleDetector(template, segments=3)

    decoy, _ = make_scene([("Fishing Bobber retrieved", 255)], gui_scale=gui_scale, seed=5)
    real, _ = make_scene([(SPLASH, 255)], gui_scale=gui_scale, seed=5)

    assert segmented.detect(decoy).score < single.detect(decoy).score - 0.2
    assert segmented.detect(real).score > 0.95


def test_empty_region_does_not_match(detector, gui_scale):
    for value in (0, 128, 255):
        img = np.full((200 * gui_scale // 2, 400 * gui_scale // 2), value, np.uint8)
        det = detector.detect(img)
        assert np.isfinite(det.score)
        assert not det.matched


def test_accepts_bgra_and_bgr(detector, gui_scale):
    img, _ = make_scene([(SPLASH, 255)], gui_scale=gui_scale)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bgra = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)  # mss 回傳的格式
    assert detector.detect(bgr).bite
    assert detector.detect(bgra).bite


def test_text_mask_excludes_shadow_and_background(gui_scale):
    template = make_template(SPLASH, gui_scale=gui_scale)
    mask = text_mask(template)
    assert template[mask].min() == 255  # 遮罩內只有文字本體
    assert template[~mask].max() < 255 // 4 + 1  # 陰影 (≈64) 與背景都在遮罩外


def test_second_best_score_is_low_for_unique_template(detector, gui_scale):
    img, _ = make_scene([("Zombie groans", 255), (SPLASH, 255)], gui_scale=gui_scale)
    assert detector.second_best_score(img) < 0.6


def test_split_segments_skips_blank_columns():
    t = np.zeros((10, 30), np.uint8)
    t[:, 0:10] = np.tile([0, 255], (10, 5))  # 只有第一段有內容
    assert split_segments(t, 3) == [(0, 10)]
    assert split_segments(np.zeros((10, 30), np.uint8), 3) == [(0, 30)]  # 全空 → 退回整張


@pytest.mark.parametrize(
    "template, message",
    [
        (np.zeros((2, 2), np.uint8), "太小"),
        (np.full((20, 80), 40, np.uint8), "單一顏色"),
        (np.tile(np.r_[np.zeros(1), np.full(60, 255)], (20, 1)).astype(np.uint8), "比例"),
    ],
)
def test_rejects_invalid_templates(template, message):
    with pytest.raises(ValueError, match=message):
        SubtitleDetector(template)


def test_region_smaller_than_template_raises(detector):
    with pytest.raises(ValueError, match="比模板"):
        detector.detect(np.zeros((5, 5), np.uint8))
