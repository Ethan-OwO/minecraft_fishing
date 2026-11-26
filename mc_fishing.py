"""Minecraft（Java 版）自動釣魚：偵測右下角「浮標濺水」聲音字幕後自動收竿、重拋。

用法：
    python mc_fishing.py calibrate   # 第一次使用：截一次字幕當模板、框選搜尋區域
    python mc_fishing.py test        # 只偵測不點擊，即時顯示分數，用來確認/調參
    python mc_fishing.py run         # 開始掛機（預設 F6 開始/暫停，F7 結束）

原理與設計說明請見 README.md。
"""

from __future__ import annotations

import argparse
import math
import queue
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import Config, Region
from detector import FRESH_GRAY, SubtitleDetector
from state_machine import Action, FishingStateMachine, Phase, Timing

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def enable_dpi_awareness() -> None:
    """讓 Windows 回報「實體像素」座標。

    Windows 開了 125%/150% 顯示縮放時，沒有宣告 DPI awareness 的程式拿到的是
    「虛擬化」後的座標：截圖範圍、滑鼠位置都會和實際畫面對不上。
    必須在任何截圖/視窗 API 呼叫之前設定，所以放在 main() 一開始。
    """
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread 在 Windows 上不支援含中文的路徑（它走 ANSI API），改用 numpy 讀檔再解碼。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise OSError(f"無法編碼影像：{path}")
    buf.tofile(str(path))


def grab(sct, region: Region) -> np.ndarray:
    """擷取搜尋區域，回傳 BGRA uint8 陣列。只抓小區域比抓整個螢幕快很多。"""
    return np.asarray(sct.grab(region.as_mss()))


# ---------------------------------------------------------------------------
# 熱鍵
# ---------------------------------------------------------------------------
def parse_key(name: str):
    from pynput import keyboard

    key = name.strip().lower()
    if hasattr(keyboard.Key, key):
        return getattr(keyboard.Key, key)
    if len(key) == 1:
        return keyboard.KeyCode.from_char(key)
    raise ValueError(f"無法辨識的按鍵名稱：{name!r}（例如 f6、f7、f10）")


class Hotkeys:
    """全域熱鍵。回呼只把指令丟進 queue，真正的處理在主執行緒做。

    為什麼不直接在回呼裡改狀態？
      1) pynput 的回呼跑在另一條執行緒，狀態機不是 thread-safe 的。
      2) Windows 的低階鍵盤 hook 若回呼太慢（超過 LowLevelHooksTimeout），
         系統會默默把 hook 拔掉，熱鍵就失效了，所以回呼必須立刻返回。
    """

    def __init__(self, bindings: dict[str, Callable[[], None]]) -> None:
        from pynput import keyboard

        self._bindings = [(parse_key(k), fn) for k, fn in bindings.items()]
        self._listener = keyboard.Listener(on_press=self._on_press)

    def _on_press(self, key) -> None:
        for target, fn in self._bindings:
            if key == target:
                fn()

    def __enter__(self) -> Hotkeys:
        self._listener.start()
        self._listener.wait()
        return self

    def __exit__(self, *exc) -> None:
        self._listener.stop()


# ---------------------------------------------------------------------------
# 前景視窗檢查（安全機制）
# ---------------------------------------------------------------------------
class FocusChecker:
    """確認前景視窗是 Minecraft 才點擊。

    模擬的右鍵會送到「目前的前景視窗」。如果你切去終端機，Windows Terminal 的
    右鍵是「貼上」，等於把剪貼簿內容塞進終端機 —— 所以不是 Minecraft 就絕不點。
    """

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword.lower()
        self._cache: tuple[float, str | None] = (-math.inf, None)
        self._xdotool = shutil.which("xdotool") if sys.platform.startswith("linux") else None

    @property
    def supported(self) -> bool:
        return sys.platform == "win32" or self._xdotool is not None

    def _title(self) -> str | None:
        if sys.platform == "win32":
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            return buf.value
        if self._xdotool:
            # 開子行程比較貴，快取 0.25 秒
            now = time.monotonic()
            if now - self._cache[0] < 0.25:
                return self._cache[1]
            try:
                out = subprocess.run(
                    [self._xdotool, "getactivewindow", "getwindowname"],
                    capture_output=True, text=True, timeout=0.5,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                out = None
            self._cache = (now, out)
            return out
        return None

    def ok(self) -> bool:
        if not self.keyword or not self.supported:
            return True
        title = self._title()
        return title is not None and self.keyword in title.lower()


def right_click(mouse, hold: float) -> None:
    from pynput.mouse import Button

    # 按下與放開之間留一點時間：遊戲是每幀/每 tick 輪詢輸入，
    # 太短的點擊在少數環境下可能被漏掉。
    mouse.press(Button.right)
    time.sleep(hold)
    mouse.release(Button.right)


# ---------------------------------------------------------------------------
# 載入偵測器
# ---------------------------------------------------------------------------
def load_detector(cfg: Config) -> SubtitleDetector:
    if cfg.region is None:
        sys.exit("尚未校正：請先執行  python mc_fishing.py calibrate")
    template = imread_unicode(cfg.template_file)
    if template is None:
        sys.exit(f"找不到模板圖片 {cfg.template_file}，請重新執行 calibrate")
    try:
        return SubtitleDetector(template, cfg.match_threshold, cfg.fresh_brightness, cfg.match_segments)
    except ValueError as e:
        sys.exit(f"模板無效：{e}")


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------
def pick_monitor(sct, index: int | None) -> dict:
    """預設挑「滑鼠所在」的螢幕：遊戲中 Minecraft 會把游標鎖在視窗中央，
    所以按下擷取鍵那一刻，游標所在的螢幕就是 Minecraft 所在的螢幕。"""
    monitors = sct.monitors  # [0] 是所有螢幕合起來的虛擬桌面，[1:] 是各螢幕
    if index is not None:
        return monitors[index]
    try:
        from pynput.mouse import Controller

        x, y = Controller().position
        for m in monitors[1:]:
            if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]:
                return m
    except Exception:  # noqa: BLE001 - 取不到游標位置就退回主螢幕
        pass
    return monitors[1]


def display_scale(w: int, h: int, max_w: int, max_h: int) -> float:
    """原圖太大就縮小顯示；太小就放大，且放大只用整數倍（搭配最近鄰插值，
    Minecraft 的像素字才不會糊掉，比較好框準）。"""
    scale = min(max_w / w, max_h / h)
    if scale >= 1:
        return float(max(1, min(int(scale), 8)))
    return scale


def roi_to_original(roi: tuple[int, int, int, int], scale: float, w: int, h: int) -> tuple[int, int, int, int] | None:
    """把在「顯示圖」上框的 (x, y, w, h) 換算回原圖座標。

    左上角往下取整、右下角往上取整，確保使用者框到的像素都被包含；再夾在影像範圍內。
    """
    x, y, rw, rh = roi
    if rw <= 0 or rh <= 0:
        return None
    x0, y0 = max(0, int(x / scale)), max(0, int(y / scale))
    x1, y1 = min(w, math.ceil((x + rw) / scale)), min(h, math.ceil((y + rh) / scale))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def select_roi(window: str, img: np.ndarray, max_w: int, max_h: int) -> tuple[int, int, int, int] | None:
    """讓使用者拖曳框選，回傳「原圖座標」的 (x, y, w, h)；取消則回傳 None。

    視窗標題用英文：OpenCV 在 Windows 上用 ANSI API 建視窗，中文標題會變亂碼。
    """
    h, w = img.shape[:2]
    scale = display_scale(w, h, max_w, max_h)
    interp = cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
    disp = img if scale == 1 else cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    except (cv2.error, AttributeError):
        pass
    roi = cv2.selectROI(window, disp, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(window)
    cv2.waitKey(1)
    return roi_to_original(tuple(int(v) for v in roi), scale, w, h)


def wait_for_command(cmds: queue.Queue[str]) -> str:
    # 用帶 timeout 的 get 迴圈而不是阻塞的 get()：Windows 上阻塞中的 Queue.get()
    # 收不到 Ctrl+C。
    while True:
        try:
            return cmds.get(timeout=0.2)
        except queue.Empty:
            continue


def cmd_calibrate(cfg: Config, config_path: Path, monitor: int | None) -> None:
    import mss

    print(
        "\n=== 校正 ===\n"
        "1. 進 Minecraft，確認「選項 → 協助工具設定 → 顯示字幕」已開啟，手上拿釣竿。\n"
        "2. 拋竿等魚咬鉤，右下角出現「浮標濺水」那行字幕（英文：Fishing Bobber splashes）時，\n"
        f"   立刻按 {cfg.toggle_key.upper()} 擷取畫面（字幕約 3 秒後消失，越早按越好）。\n"
        f"   按 {cfg.quit_key.upper()} 取消。\n"
    )
    cmds: queue.Queue[str] = queue.Queue()
    with Hotkeys({cfg.toggle_key: lambda: cmds.put("capture"), cfg.quit_key: lambda: cmds.put("quit")}):
        if wait_for_command(cmds) == "quit":
            print("已取消。")
            return

    with mss.mss() as sct:
        mon = pick_monitor(sct, monitor)
        shot = np.asarray(sct.grab(mon))
    screen = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
    calib_dir = cfg.template_file.parent
    imwrite_unicode(calib_dir / "screen.png", screen)
    log(f"已擷取螢幕 {mon['width']}x{mon['height']}（存於 {calib_dir / 'screen.png'}）")

    max_w, max_h = int(mon["width"] * 0.9), int(mon["height"] * 0.85)

    print(
        "\n[步驟 1/2] 框選「搜尋區域」：把右下角字幕框會出現的範圍框大一點。\n"
        "  - 同時有多個聲音時字幕會往上堆疊，所以高度要涵蓋好幾行。\n"
        "  - 字幕框寬度會隨最長那行改變，左邊也多留一些。\n"
        "  拖曳框選後按 Enter/空白鍵確認，按 c 取消。"
    )
    r = select_roi("Step 1/2: subtitle search region", screen, max_w, max_h)
    if r is None:
        print("已取消。")
        return
    rx, ry, rw, rh = r
    region_img = screen[ry : ry + rh, rx : rx + rw]

    print(
        "\n[步驟 2/2] 框選「浮標濺水」那一行的文字（畫面已放大）。\n"
        "  - 只框文字本身，四周留 1~2 個像素即可。\n"
        "  - 不要框到左右兩側的方向箭頭 < >（箭頭會隨聲音方向出現/消失）。"
    )
    t = select_roi("Step 2/2: bobber splash text", region_img, max_w, max_h)
    if t is None:
        print("已取消。")
        return
    tx, ty, tw, th = t
    template = region_img[ty : ty + th, tx : tx + tw]

    try:
        detector = SubtitleDetector(template, cfg.match_threshold, cfg.fresh_brightness, cfg.match_segments)
    except ValueError as e:
        print(f"\n[失敗] {e}")
        return

    imwrite_unicode(cfg.template_file, template)
    cfg.region = Region(left=mon["left"] + rx, top=mon["top"] + ry, width=rw, height=rh)
    cfg.save(config_path)

    det = detector.detect(region_img)
    second = detector.second_best_score(region_img)
    print(
        f"\n已儲存模板 {cfg.template_file}（{tw}x{th}）與設定 {config_path}\n"
        f"自我檢查：最佳分數 {det.score:.3f}（應接近 1）、"
        f"次佳分數 {second:.3f}（越低越好，門檻 {cfg.match_threshold}）、"
        f"模板文字亮度 {detector.template_brightness:.0f}"
    )
    if second >= cfg.match_threshold - 0.05:
        print("  [注意] 次佳分數接近門檻：畫面上有其他很像的東西，建議框整行字，或調高 match_threshold。")
    if detector.template_brightness < cfg.fresh_brightness:
        print(
            f"  [注意] 擷取時字幕已經淡掉（亮度 {detector.template_brightness:.0f} < {cfg.fresh_brightness:.0f}）。\n"
            "         形狀比對不受影響；但若你的新字幕本來就不是白色（資源包改過字色），\n"
            f"         請把 fresh_brightness 調低。原版新字幕亮度為 {FRESH_GRAY}。"
        )
    if rh < th * 3:
        print("  [注意] 搜尋區域太矮：其他字幕出現時這行會被往上推，可能跑出範圍。")
    print("\n下一步：python mc_fishing.py test  確認偵測正常，再用 run 開始。")


# ---------------------------------------------------------------------------
# test：只偵測不點擊
# ---------------------------------------------------------------------------
def cmd_test(cfg: Config) -> None:
    detector = load_detector(cfg)

    import mss

    print(
        f"測試模式：只偵測、不點擊。去 Minecraft 手動釣魚，觀察咬鉤時是否出現 ★。\n"
        f"按 {cfg.quit_key.upper()} 或 Ctrl+C 結束。\n"
    )
    cmds: queue.Queue[str] = queue.Queue()
    prev_bite = False
    with mss.mss() as sct, Hotkeys({cfg.quit_key: lambda: cmds.put("quit")}):
        try:
            while cmds.empty():
                t0 = time.perf_counter()
                det = detector.detect(grab(sct, cfg.region))
                ms = (time.perf_counter() - t0) * 1000
                status = (
                    f"score={det.score:5.3f}{'✓' if det.matched else ' '}  "
                    f"brightness={det.brightness:5.1f}{'✓' if det.fresh else ' '}  "
                    f"({ms:4.1f} ms/frame)"
                )
                if det.bite and not prev_bite:
                    print(f"\r★ 偵測到新的濺水字幕  {status}")
                else:
                    print(f"\r  {status}", end="", flush=True)
                prev_bite = det.bite
                time.sleep(cfg.poll_interval)
        except KeyboardInterrupt:
            pass
    print("\n結束測試。")


# ---------------------------------------------------------------------------
# run：掛機
# ---------------------------------------------------------------------------
def save_debug_frame(debug_dir: Path, frame: np.ndarray, det, template_shape) -> None:
    img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    th, tw = template_shape
    x, y = det.loc
    cv2.rectangle(img, (x, y), (x + tw - 1, y + th - 1), (0, 255, 0), 1)
    name = f"{datetime.now():%Y%m%d_%H%M%S_%f}_s{det.score:.2f}_b{det.brightness:.0f}.png"
    imwrite_unicode(debug_dir / name, img)


def cmd_run(cfg: Config, debug_dir: Path | None) -> None:
    detector = load_detector(cfg)  # 先檢查校正狀態，再載入 mss / pynput

    import mss
    from pynput.mouse import Controller

    sm = FishingStateMachine(
        Timing(
            cast_grace=cfg.cast_grace,
            recast_delay=cfg.recast_delay,
            max_wait=cfg.max_wait,
            resync_after_timeouts=cfg.resync_after_timeouts,
        )
    )
    mouse = Controller()
    focus = FocusChecker(cfg.window_title_keyword)
    cmds: queue.Queue[str] = queue.Queue()

    print(
        "\n=== 自動釣魚 ===\n"
        f"  {cfg.toggle_key.upper()}：開始 / 暫停    {cfg.quit_key.upper()}：結束\n"
        "  開始前請確認：手上拿釣竿、浮標「沒有」拋出、字幕已開啟、不要按 F1 隱藏 HUD。\n"
    )
    if cfg.window_title_keyword and not focus.supported:
        print("  [注意] 此系統無法檢查前景視窗，請確保執行期間 Minecraft 一直在前景。\n")

    was_focused = True
    with mss.mss() as sct, Hotkeys(
        {cfg.toggle_key: lambda: cmds.put("toggle"), cfg.quit_key: lambda: cmds.put("quit")}
    ):
        try:
            while True:
                # 1) 處理熱鍵指令（只在主執行緒動狀態機）
                quit_requested = False
                while not cmds.empty():
                    cmd = cmds.get_nowait()
                    if cmd == "quit":
                        quit_requested = True
                    elif cmd == "toggle":
                        if sm.phase is Phase.PAUSED:
                            sm.start(time.monotonic())
                            log("開始" + ("（浮標應仍在水中，繼續等待）" if sm.phase is Phase.WAITING else ""))
                        else:
                            sm.pause()
                            log("暫停")
                if quit_requested:
                    break
                if sm.phase is Phase.PAUSED:
                    time.sleep(0.05)
                    continue

                # 2) 安全檢查：Minecraft 不在前景就什麼都不做
                if not focus.ok():
                    if was_focused:
                        log("Minecraft 不在前景，停止點擊（切回遊戲後自動繼續）")
                    was_focused = False
                    time.sleep(0.1)
                    continue
                if not was_focused:
                    log("回到 Minecraft，繼續")
                    was_focused = True

                # 3) 偵測 → 狀態機 → 執行動作
                now = time.monotonic()
                bite = False
                det = frame = None
                if sm.wants_frame(now):
                    frame = grab(sct, cfg.region)
                    det = detector.detect(frame)
                    bite = det.bite

                action = sm.update(now, bite)
                if action is not Action.NONE:
                    right_click(mouse, cfg.click_hold)
                    extra = f"  (score={det.score:.2f}, brightness={det.brightness:.0f})" if bite else ""
                    if action is not Action.CAST:
                        log(f"{sm.last_event}{extra}   [漁獲 {sm.catches} / 逾時 {sm.timeouts}]")
                    if bite and debug_dir is not None:
                        save_debug_frame(debug_dir, frame, det, detector.template.shape)

                time.sleep(cfg.poll_interval)
        except KeyboardInterrupt:
            pass

    log(f"結束。共收竿 {sm.catches} 次、逾時 {sm.timeouts} 次、重新同步 {sm.resyncs} 次。")


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Minecraft 聲音字幕自動釣魚")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="設定檔路徑（預設：腳本旁的 config.json）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="擷取字幕模板與搜尋區域")
    p_cal.add_argument("--monitor", type=int, default=None, help="指定螢幕編號（1 起算；預設為滑鼠所在螢幕）")
    sub.add_parser("test", help="只偵測不點擊，顯示即時分數")
    p_run = sub.add_parser("run", help="開始自動釣魚")
    p_run.add_argument("--debug", type=Path, nargs="?", const=Path("debug"), default=None,
                       help="每次判定咬鉤時存下截圖到此資料夾（預設 debug/），用來查誤判")

    args = parser.parse_args(argv)
    enable_dpi_awareness()
    cfg = Config.load(args.config)

    if args.command == "calibrate":
        cmd_calibrate(cfg, args.config, args.monitor)
    elif args.command == "test":
        cmd_test(cfg)
    elif args.command == "run":
        cmd_run(cfg, args.debug)


if __name__ == "__main__":
    main()
