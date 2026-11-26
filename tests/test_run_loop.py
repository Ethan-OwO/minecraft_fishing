"""整合測試：用假的 mss / pynput 跑真正的 cmd_run 主迴圈。

假世界的規則與 Minecraft 相同：右鍵是切換；浮標真的在水裡一段時間後，
右下角出現新的濺水字幕，接著淡出。驗證的是「熱鍵 → 狀態機 → 截圖比對 → 點擊」
整條串接，而不只是個別元件。
"""

import sys
import threading
import time
import types

import cv2
from synth import make_scene, make_template

import mc_fishing
from config import Config, Region

SPLASH = "Fishing Bobber splashes"
BITE_AFTER = 0.25  # 浮標落水後多久咬鉤（秒）


class FakeWorld:
    def __init__(self):
        self.lock = threading.Lock()
        self.rod_out = False
        self.out_since = None
        self.clicks = []
        self.fresh, boxes = make_scene([("Zombie groans", 255), (SPLASH, 255)], seed=3)
        self.faded, _ = make_scene([("Zombie groans", 255), (SPLASH, 140)], seed=3)
        self.empty, _ = make_scene([("Zombie groans", 255)], seed=3)

    def click(self):
        with self.lock:
            self.rod_out = not self.rod_out
            self.out_since = time.monotonic() if self.rod_out else None
            self.clicks.append((time.monotonic(), self.rod_out))

    def frame(self):
        with self.lock:
            if self.out_since is None:
                return self.empty
            age = time.monotonic() - self.out_since - BITE_AFTER
        if 0 <= age < 0.15:
            return self.fresh
        if 0.15 <= age < 2.0:
            return self.faded
        return self.empty


def install_fakes(monkeypatch, world: FakeWorld):
    listeners = []

    class Listener:
        def __init__(self, on_press):
            self.on_press = on_press
            listeners.append(self)

        def start(self):
            pass

        def wait(self):
            pass

        def stop(self):
            pass

    class Controller:
        position = (0, 0)

        def press(self, button):
            assert button == "right"

        def release(self, button):
            world.click()  # 放開時才算一次點擊

    class Key:
        f6 = "F6"
        f7 = "F7"

    class KeyCode:
        @staticmethod
        def from_char(c):
            return c

    keyboard = types.SimpleNamespace(Key=Key, KeyCode=KeyCode, Listener=Listener)
    mouse = types.SimpleNamespace(Controller=Controller, Button=types.SimpleNamespace(right="right"))
    pynput = types.ModuleType("pynput")
    pynput.keyboard, pynput.mouse = keyboard, mouse
    monkeypatch.setitem(sys.modules, "pynput", pynput)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)
    monkeypatch.setitem(sys.modules, "pynput.mouse", mouse)

    class Sct:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def grab(self, region):
            img = world.frame()
            assert (region["width"], region["height"]) == (img.shape[1], img.shape[0])
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)  # mss 給的是 BGRA

    monkeypatch.setitem(sys.modules, "mss", types.SimpleNamespace(mss=Sct))
    return listeners


def test_run_loop_end_to_end(monkeypatch, tmp_path, capsys):
    world = FakeWorld()
    listeners = install_fakes(monkeypatch, world)

    template_path = tmp_path / "template.png"
    mc_fishing.imwrite_unicode(template_path, cv2.cvtColor(make_template(SPLASH), cv2.COLOR_GRAY2BGR))
    h, w = world.empty.shape
    cfg = Config(
        region=Region(0, 0, w, h),
        template_path=str(template_path),
        cast_grace=0.15,  # 比字幕淡到門檻以下的時間長，但比咬鉤時間短
        recast_delay=0.05,
        max_wait=5.0,
        poll_interval=0.005,
        click_hold=0.0,
        window_title_keyword="",
    )

    t = threading.Thread(target=mc_fishing.cmd_run, args=(cfg, tmp_path / "debug"))
    t.start()
    try:
        deadline = time.monotonic() + 2.0
        while not listeners and time.monotonic() < deadline:
            time.sleep(0.01)
        (listener,) = listeners
        listener.on_press("F6")  # 開始
        time.sleep(2.0)
        listener.on_press("F6")  # 暫停
        time.sleep(0.2)
        clicks_while_paused = len(world.clicks)
        time.sleep(0.5)
        assert len(world.clicks) == clicks_while_paused  # 暫停時不點擊
        listener.on_press("F7")  # 結束
    finally:
        t.join(timeout=3)
    assert not t.is_alive()

    out = capsys.readouterr().out
    catches = out.count("咬鉤")
    # 每輪 ≈ recast 0.05 + 咬鉤 0.25 秒，2 秒內應該釣到好幾次，而且沒有逾時
    assert catches >= 4, out
    assert "逾時 0" in out
    # 點擊序列：拋、收、拋、收…（rod_out 交替），每次咬鉤都存了除錯截圖
    states = [s for _, s in world.clicks]
    assert states == [i % 2 == 0 for i in range(len(states))]
    assert len(list((tmp_path / "debug").glob("*.png"))) == catches
