"""釣魚流程的狀態機 —— 純邏輯，不做任何 I/O。

為什麼要把它獨立出來？
  主迴圈要同時處理「截圖、比對、點擊、熱鍵、時間」，如果把流程判斷也塞進去，
  就只能開著 Minecraft 實測。把「現在該做什麼」寫成一個只吃 (時間, 有沒有咬鉤)
  、吐出 Action 的狀態機，就能用假的時間在單元測試裡驗證每一種情況
  （見 tests/test_state_machine.py）。主迴圈只負責執行 Action。

右鍵在 Minecraft 裡是「切換」：浮標沒拋出時按 = 拋竿，浮標在水裡時按 = 收竿。
腳本沒辦法直接知道浮標在不在水裡，只能靠「自己按了幾次」推算
（開迴路控制，open-loop）。這就是 rod_out 這個「信念」的由來，
以及為什麼需要 RESYNC 這個自我修正機制（見 update() 裡的說明）。

狀態轉移：

    PAUSED ──start()──► CAST_PENDING ──(時間到)──► WAITING
                              ▲         Action.CAST     │
                              │                         │ 咬鉤 → Action.REEL
                              └─────────────────────────┤ 逾時 → Action.REEL
                                                        │ 連續逾時 → Action.RESYNC（留在 WAITING）
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Phase(Enum):
    PAUSED = auto()
    CAST_PENDING = auto()  # 浮標已收回，等 recast_delay 過去後拋竿
    WAITING = auto()  # 浮標在水裡，等咬鉤


class Action(Enum):
    NONE = auto()
    CAST = auto()  # 拋竿（按一次右鍵）
    REEL = auto()  # 收竿（按一次右鍵），之後會自動再拋
    RESYNC = auto()  # 只按一次右鍵來翻轉狀態，修正「信念」與實際不同步


@dataclass(frozen=True)
class Timing:
    # 拋竿後這段時間內不判斷咬鉤。理由：
    #  1) 上一條濺水字幕在這段時間內可能仍很亮（見 README 的時間計算）
    #  2) 浮標落水本身也會有其他濺水類音效
    cast_grace: float = 1.5
    # 收竿後隔多久再拋。Minecraft 使用物品後有 4 tick（0.2 秒）的右鍵冷卻，
    # 太快的第二下會被吃掉，所以要留餘裕。
    recast_delay: float = 0.6
    # 拋竿後超過這麼久沒咬鉤就收竿重拋（浮標掉到地上、被卡住等情況的保險）。
    max_wait: float = 60.0
    # 連續逾時幾次就執行 RESYNC；0 代表停用。
    resync_after_timeouts: int = 2


class FishingStateMachine:
    def __init__(self, timing: Timing | None = None) -> None:
        self.timing = timing or Timing()
        self.phase = Phase.PAUSED
        self.rod_out = False  # 腳本「相信」浮標目前在水裡
        self.catches = 0
        self.timeouts = 0
        self.resyncs = 0
        self._consecutive_timeouts = 0
        self._cast_at = 0.0
        self._next_cast_at = 0.0
        self.last_event = ""

    # ---- 外部控制 --------------------------------------------------------
    def start(self, now: float) -> None:
        """開始/恢復。若暫停前浮標在水裡，就直接回到等待，不要多按一下把它收回來。"""
        if self.phase is not Phase.PAUSED:
            return
        if self.rod_out:
            self.phase = Phase.WAITING
            self._cast_at = now  # 重新計時，避免暫停很久後一恢復就判定逾時
        else:
            self.phase = Phase.CAST_PENDING
            self._next_cast_at = now

    def pause(self) -> None:
        """暫停時不點擊，只記住 rod_out，恢復時才能接續。"""
        self.phase = Phase.PAUSED

    # ---- 查詢 ------------------------------------------------------------
    def wants_frame(self, now: float) -> bool:
        """只有在「等待咬鉤且過了寬限期」才需要截圖比對，其餘時間省下 CPU。"""
        return self.phase is Phase.WAITING and now - self._cast_at >= self.timing.cast_grace

    # ---- 主要邏輯 --------------------------------------------------------
    def update(self, now: float, bite: bool) -> Action:
        t = self.timing

        if self.phase is Phase.CAST_PENDING:
            if now >= self._next_cast_at:
                self.phase = Phase.WAITING
                self.rod_out = True
                self._cast_at = now
                self.last_event = "拋竿"
                return Action.CAST
            return Action.NONE

        if self.phase is not Phase.WAITING:
            return Action.NONE

        elapsed = now - self._cast_at
        if elapsed < t.cast_grace:
            return Action.NONE

        if bite:
            self.catches += 1
            self._consecutive_timeouts = 0
            self.last_event = f"咬鉤！收竿（第 {self.catches} 次）"
            return self._reel(now)

        if elapsed >= t.max_wait:
            self.timeouts += 1
            self._consecutive_timeouts += 1
            if t.resync_after_timeouts and self._consecutive_timeouts >= t.resync_after_timeouts:
                # 為什麼連續逾時要「只按一下」？
                # 若信念與實際不同步（腳本以為浮標在水裡，其實沒有），正常的
                # 「收竿+拋竿」兩下會變成「拋竿+收竿」，浮標永遠在手上、永遠逾時，
                # 掛機一整晚都白費。按一下（奇數次）可以翻轉奇偶性，讓兩者重新一致。
                # 代價：若其實是同步的、只是運氣差連續沒咬鉤，這一下會反過來造成
                # 不同步，但下一輪連續逾時會再翻回來 —— 最差情況只是浪費幾分鐘。
                self._consecutive_timeouts = 0
                self.resyncs += 1
                self._cast_at = now
                self.last_event = f"連續逾時，嘗試重新同步（第 {self.resyncs} 次）"
                return Action.RESYNC
            self.last_event = f"等待 {t.max_wait:.0f} 秒沒咬鉤，收竿重拋"
            return self._reel(now)

        return Action.NONE

    def _reel(self, now: float) -> Action:
        self.phase = Phase.CAST_PENDING
        self.rod_out = False
        self._next_cast_at = now + self.timing.recast_delay
        return Action.REEL
