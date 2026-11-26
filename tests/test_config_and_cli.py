import json

import pytest

import mc_fishing
from config import Config, Region


def test_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    cfg = Config.load(path)  # 不存在 → 預設值
    assert cfg.region is None
    cfg.region = Region(left=-1920, top=0, width=500, height=300)
    cfg.max_wait = 45.0
    cfg.save(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert "_base_dir" not in data
    loaded = Config.load(path)
    assert loaded.region == cfg.region
    assert loaded.max_wait == 45.0
    assert loaded.region.as_mss() == {"left": -1920, "top": 0, "width": 500, "height": 300}


def test_relative_template_path_is_resolved_from_config_dir(tmp_path):
    path = tmp_path / "sub" / "config.json"
    path.parent.mkdir()
    path.write_text('{"template_path": "calib/t.png"}', encoding="utf-8")
    assert Config.load(path).template_file == tmp_path / "sub" / "calib" / "t.png"


def test_unknown_keys_warn(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text('{"max_wait": 30, "typo_key": 1}', encoding="utf-8")
    cfg = Config.load(path)
    assert cfg.max_wait == 30
    assert "typo_key" in capsys.readouterr().out


@pytest.mark.parametrize(
    "w, h, max_w, max_h, expected",
    [
        (3840, 2160, 1600, 900, 900 / 2160),  # 4K 截圖縮小顯示
        (400, 100, 1600, 900, 4.0),  # 小區域用整數倍放大
        (300, 80, 1600, 900, 5.0),
        (1600, 900, 1600, 900, 1.0),
        (100, 10, 5000, 5000, 8.0),  # 放大上限 8 倍
    ],
)
def test_display_scale(w, h, max_w, max_h, expected):
    assert mc_fishing.display_scale(w, h, max_w, max_h) == pytest.approx(expected)


def test_roi_to_original_covers_selected_pixels():
    # 放大 4 倍顯示時框了 (10, 6) 起 21x9 → 原圖要包含所有被框到的像素
    assert mc_fishing.roi_to_original((10, 6, 21, 9), 4.0, 100, 50) == (2, 1, 6, 3)
    # 縮小 0.5 顯示：座標放大回去
    assert mc_fishing.roi_to_original((100, 50, 200, 100), 0.5, 3840, 2160) == (200, 100, 400, 200)
    # 夾在影像範圍內
    assert mc_fishing.roi_to_original((90, 40, 50, 50), 1.0, 100, 50) == (90, 40, 10, 10)
    # 取消（寬或高為 0）
    assert mc_fishing.roi_to_original((0, 0, 0, 0), 1.0, 100, 50) is None


def test_cli_help_works_without_display(capsys):
    # pynput / mss 都是延遲載入，沒有螢幕的環境也能看說明
    with pytest.raises(SystemExit) as e:
        mc_fishing.main(["--help"])
    assert e.value.code == 0
    assert "calibrate" in capsys.readouterr().out


def test_run_without_calibration_exits_with_hint(tmp_path):
    with pytest.raises(SystemExit, match="calibrate"):
        mc_fishing.main(["--config", str(tmp_path / "none.json"), "run"])


def test_unicode_path_image_roundtrip(tmp_path):
    import numpy as np

    img = (np.arange(60 * 40) % 256).astype(np.uint8).reshape(40, 60)
    img = np.dstack([img] * 3)
    path = tmp_path / "校正" / "模板.png"
    mc_fishing.imwrite_unicode(path, img)
    back = mc_fishing.imread_unicode(path)
    assert back is not None and (back == img).all()
    assert mc_fishing.imread_unicode(tmp_path / "missing.png") is None
