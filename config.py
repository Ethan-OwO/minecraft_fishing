"""設定檔讀寫。校正 (calibrate) 會寫入 region 與模板，其他參數可手動改 config.json。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Region:
    """螢幕上的絕對座標（實體像素；多螢幕時可能是負數）。格式與 mss 的 monitor dict 相同。"""

    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class Config:
    region: Region | None = None
    template_path: str = "calibration/template.png"

    # 偵測
    match_threshold: float = 0.85
    fresh_brightness: float = 200.0
    match_segments: int = 3

    # 時間（秒）
    cast_grace: float = 1.5
    recast_delay: float = 0.6
    max_wait: float = 60.0
    resync_after_timeouts: int = 2
    poll_interval: float = 0.03
    click_hold: float = 0.05

    # 熱鍵（pynput 的 Key 名稱，例如 f6、f7、f10）
    toggle_key: str = "f6"
    quit_key: str = "f7"

    # 只有前景視窗標題包含這個字串時才會點擊（目前只在 Windows 上檢查）。
    # 設成 "" 可停用。若用其他啟動器（例如標題不含 Minecraft），請改成對應字串。
    window_title_keyword: str = "Minecraft"

    # 載入時記錄設定檔位置，用來解析相對路徑；不寫回檔案。
    _base_dir: Path = field(default=Path("."), repr=False, compare=False)

    @property
    def template_file(self) -> Path:
        p = Path(self.template_path)
        return p if p.is_absolute() else self._base_dir / p

    @classmethod
    def load(cls, path: Path) -> Config:
        path = Path(path)
        cfg = cls(_base_dir=path.parent)
        if not path.exists():
            return cfg
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls) if not f.name.startswith("_")}
        unknown = set(data) - known
        if unknown:
            print(f"[警告] 設定檔中有未知欄位，已忽略：{', '.join(sorted(unknown))}")
        for name in known & set(data):
            value = data[name]
            if name == "region" and value is not None:
                value = Region(**value)
            setattr(cfg, name, value)
        return cfg

    def save(self, path: Path) -> None:
        path = Path(path)
        data = {
            f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")
        }
        if self.region is not None:
            data["region"] = asdict(self.region)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
