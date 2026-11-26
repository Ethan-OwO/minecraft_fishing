import pytest

from state_machine import Action, FishingStateMachine, Phase, Timing

T = Timing(cast_grace=1.5, recast_delay=0.6, max_wait=60.0, resync_after_timeouts=2)


@pytest.fixture
def sm():
    m = FishingStateMachine(T)
    m.start(0.0)
    return m


def test_paused_by_default_does_nothing():
    m = FishingStateMachine(T)
    assert m.update(0.0, bite=True) is Action.NONE
    assert not m.wants_frame(100.0)


def test_casts_immediately_on_start(sm):
    assert sm.update(0.0, False) is Action.CAST
    assert sm.phase is Phase.WAITING and sm.rod_out


def test_ignores_bites_during_grace_period(sm):
    sm.update(0.0, False)  # CAST
    assert not sm.wants_frame(1.0)
    assert sm.update(1.0, bite=True) is Action.NONE  # 上一條字幕的殘影 / 落水音效
    assert sm.wants_frame(1.5)


def test_full_cycle_bite_reel_recast(sm):
    assert sm.update(0.0, False) is Action.CAST
    assert sm.update(5.0, False) is Action.NONE
    assert sm.update(8.0, True) is Action.REEL
    assert sm.catches == 1 and not sm.rod_out
    assert sm.update(8.3, False) is Action.NONE  # 尚未到 recast_delay（避免右鍵冷卻吃掉）
    assert sm.update(8.6, False) is Action.CAST
    assert sm.rod_out


def test_timeout_reels_and_recasts(sm):
    sm.update(0.0, False)
    assert sm.update(59.9, False) is Action.NONE
    assert sm.update(60.0, False) is Action.REEL
    assert sm.timeouts == 1 and sm.catches == 0
    assert sm.update(60.6, False) is Action.CAST


def test_consecutive_timeouts_trigger_single_click_resync(sm):
    sm.update(0.0, False)  # CAST
    assert sm.update(60.0, False) is Action.REEL  # 第 1 次逾時：正常收+拋
    assert sm.update(61.0, False) is Action.CAST
    assert sm.update(121.0, False) is Action.RESYNC  # 第 2 次連續逾時：只按一下
    assert sm.phase is Phase.WAITING  # 信念：按完這下浮標在水裡
    assert sm.resyncs == 1
    # RESYNC 後重新計時並重設計數：接著要再連續逾時 2 次才會再 RESYNC
    assert sm.update(150.0, False) is Action.NONE
    assert sm.update(181.0, False) is Action.REEL


def simulate(sm: FishingStateMachine, real_rod_out: bool, bite_after: float, duration: float, dt=0.05):
    """極簡的 Minecraft 世界模型：

    - 右鍵是「切換」：每次點擊翻轉 real_rod_out
    - 浮標「真的」在水裡 bite_after 秒後，出現一條新的濺水字幕（新鮮約 0.9 秒）
    回傳每次 REEL 是否真的釣到魚（收竿當下浮標真的在水裡且字幕新鮮）。
    """
    out_since = 0.0 if real_rod_out else None
    caught = []
    now = 0.0
    while now < duration:
        fresh = out_since is not None and 0 <= now - out_since - bite_after < 0.9
        action = sm.update(now, bite=fresh)
        if action is not Action.NONE:
            if action is Action.REEL and fresh:
                caught.append(now)
            real_rod_out = not real_rod_out
            out_since = now if real_rod_out else None
        now += dt
    return caught, real_rod_out


def test_in_sync_catches_steadily():
    sm = FishingStateMachine(T)
    sm.start(0.0)
    caught, _ = simulate(sm, real_rod_out=False, bite_after=20.0, duration=600)
    assert len(caught) >= 25  # 每輪約 20.6 秒
    assert sm.timeouts == 0


def test_recovers_after_desync():
    """開始前使用者自己先拋了竿（腳本不知道）→ 第一下 CAST 其實是收竿 → 不同步。
    沒有 RESYNC 會永遠釣不到；有 RESYNC 會在兩次逾時後恢復。"""
    sm = FishingStateMachine(T)
    sm.start(0.0)
    caught, real_rod_out = simulate(sm, real_rod_out=True, bite_after=20.0, duration=900)
    assert sm.resyncs == 1
    assert real_rod_out == sm.rod_out  # 信念與實際重新一致
    assert caught and caught[0] < 2 * T.max_wait + 30
    assert len(caught) >= 30

    no_resync = FishingStateMachine(Timing(resync_after_timeouts=0))
    no_resync.start(0.0)
    caught, _ = simulate(no_resync, real_rod_out=True, bite_after=20.0, duration=900)
    assert caught == []  # 開迴路控制一旦失步就無法自己恢復


def test_bite_resets_consecutive_timeout_counter(sm):
    sm.update(0.0, False)
    assert sm.update(60.0, False) is Action.REEL  # 逾時 1
    sm.update(60.6, False)  # CAST
    assert sm.update(70.0, True) is Action.REEL  # 咬鉤 → 計數歸零
    sm.update(70.6, False)  # CAST
    assert sm.update(130.6, False) is Action.REEL  # 又是第 1 次逾時，不是 RESYNC


def test_resync_can_be_disabled():
    m = FishingStateMachine(Timing(max_wait=10, resync_after_timeouts=0))
    m.start(0.0)
    t = 0.0
    m.update(t, False)
    for _ in range(5):
        t += 10
        assert m.update(t, False) is Action.REEL
        t += 1
        assert m.update(t, False) is Action.CAST


def test_pause_and_resume_while_rod_out_does_not_click(sm):
    sm.update(0.0, False)  # CAST
    sm.pause()
    assert sm.update(10.0, True) is Action.NONE
    sm.start(100.0)
    assert sm.phase is Phase.WAITING  # 不會多按一下把浮標收回來
    assert sm.update(100.1, False) is Action.NONE
    assert sm.update(159.0, False) is Action.NONE  # 計時從恢復時重新開始
    assert sm.update(160.0, False) is Action.REEL


def test_pause_and_resume_while_rod_in_casts(sm):
    sm.update(0.0, False)
    sm.update(5.0, True)  # REEL → CAST_PENDING
    sm.pause()
    sm.start(50.0)
    assert sm.update(50.0, False) is Action.CAST


def test_start_twice_is_harmless(sm):
    sm.update(0.0, False)
    sm.start(30.0)  # 已在執行中 → 忽略
    assert sm.update(60.0, False) is Action.REEL
