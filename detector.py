"""字幕偵測核心：模板比對 (template matching) + 字幕新鮮度 (freshness) 判斷。

這個模組只做純影像運算，不碰螢幕擷取、滑鼠、鍵盤，所以可以在沒有螢幕的
環境下單元測試（見 tests/test_detector.py）。

偵測分兩步，各自回答一個問題：

1. 「浮標濺水」那行字幕有沒有出現在畫面上？
   → 用正規化互相關 (NCC, cv2.TM_CCOEFF_NORMED) 在搜尋區域裡滑動比對模板。
   NCC 會先把模板和視窗各自減掉平均值、再除以標準差，因此對「整體變亮/變暗、
   對比變大/變小」（也就是 I' = a*I + b 這種仿射亮度變化）完全不敏感。
   字幕淡出時文字變灰，NCC 分數幾乎不變，所以「形狀」辨識很穩定。

   分段取最小值：整張模板的 NCC 本質上是「所有像素的平均相關」，
   「Fishing Bobber splashes」和「Fishing Bobber retrieved」共用前綴，
   光靠前綴就能拿到 0.8 以上。所以把模板直切成數段、各自算 NCC，
   在同一個位置取「最低的那段」當分數 —— 從「平均吻合」變成「每一段都要吻合」。

2. 這行字幕是「剛剛才出現」的，還是幾秒前的殘影？
   NCC 對亮度不敏感是優點，但也代表它分不出新舊字幕。原版 Minecraft 的
   SubtitleOverlay 會把文字顏色在「3 秒 × 通知顯示時間倍率」內從灰階 255
   線性降到 75，而且文字本身是「不透明」繪製（只有背景框是半透明）。
   所以「文字像素的灰階值」就是一個不受遊戲背景影響的計時器：
   值越接近 255，代表字幕越新。我們用模板的文字遮罩取出比對位置上的文字像素，
   取中位數當作亮度，超過門檻才算「新鮮」。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# 原版字幕文字顏色的範圍（灰階）：剛出現時 255，淡出到最後是 75。
FRESH_GRAY = 255
FADED_GRAY = 75


@dataclass(frozen=True)
class Detection:
    """一次偵測的結果。保留原始數值（score、brightness）方便除錯與調參。"""

    score: float  # NCC 最高分，範圍 [-1, 1]，1 代表完全吻合
    loc: tuple[int, int]  # 最佳比對位置（模板左上角）在搜尋區域內的 (x, y)
    brightness: float  # 比對位置上「文字像素」的灰階中位數
    matched: bool  # score >= match_threshold：形狀吻合
    fresh: bool  # brightness >= fresh_brightness：字幕是剛出現的

    @property
    def bite(self) -> bool:
        """形狀吻合且是新字幕，才視為「魚咬鉤了」。"""
        return self.matched and self.fresh


def to_gray(img: np.ndarray) -> np.ndarray:
    """把 BGRA / BGR / 灰階影像統一轉成 uint8 灰階。

    mss 擷取回來是 BGRA，cv2.imread 讀回來是 BGR，這裡統一處理，
    讓上層不用關心影像來源。
    """
    if img.ndim == 2:
        gray = img
    elif img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"不支援的影像通道數：{img.shape}")
    return np.ascontiguousarray(gray, dtype=np.uint8)


def text_mask(template_gray: np.ndarray) -> np.ndarray:
    """用 Otsu 法自動找門檻，把模板切成「文字」與「非文字」兩類。

    字幕模板大致有三種灰階：深色背景框、文字陰影（約文字亮度的 1/4）、文字本體。
    Otsu 會挑一個讓兩類「類內變異數最小」的門檻；因為白字和其他兩者差距最大，
    門檻會落在陰影與文字之間，於是遮罩只留下文字本體 —— 正是我們要量亮度的像素。
    """
    _, mask = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0


def split_segments(template_gray: np.ndarray, n: int) -> list[tuple[int, int]]:
    """把模板直切成 n 段，回傳每段的 [x0, x1) 欄範圍。

    幾乎沒有內容的段（例如剛好切在字和字之間的空白）NCC 沒有意義
    （分母的標準差趨近 0），直接略過；全部都被略過時退回整張模板。
    """
    w = template_gray.shape[1]
    n = max(1, min(n, w // 4))  # 每段至少 4px 寬
    bounds = np.linspace(0, w, n + 1).round().astype(int)
    segs = [
        (int(x0), int(x1))
        for x0, x1 in zip(bounds[:-1], bounds[1:], strict=True)
        if float(template_gray[:, x0:x1].std()) >= 1.0
    ]
    return segs or [(0, w)]


class SubtitleDetector:
    def __init__(
        self,
        template: np.ndarray,
        match_threshold: float = 0.85,
        fresh_brightness: float = 200.0,
        segments: int = 3,
    ) -> None:
        t = to_gray(template)
        if t.shape[0] < 4 or t.shape[1] < 4:
            raise ValueError(f"模板太小（{t.shape[1]}x{t.shape[0]}），請重新校正並框選整行字幕文字")
        # NCC 的分母是標準差；完全平的模板標準差為 0，分數沒有定義。
        if float(t.std()) < 1.0:
            raise ValueError("模板幾乎是單一顏色，看起來沒有框到字幕文字，請重新校正")

        mask = text_mask(t)
        ratio = float(mask.mean())
        # 正常的一行字，文字像素大約佔框選範圍的 5%~50%。
        # 太少代表幾乎沒框到字；太多代表框到的是大片亮色區域（例如天空）。
        if not 0.02 <= ratio <= 0.8:
            raise ValueError(
                f"模板中文字像素比例 {ratio:.1%} 不像一行字幕，請只框選字幕文字本身"
            )

        self.template = t
        self.mask = mask
        self.segments = split_segments(t, segments)
        self.match_threshold = float(match_threshold)
        self.fresh_brightness = float(fresh_brightness)

    @property
    def template_brightness(self) -> float:
        """模板本身文字像素的亮度，校正時用來判斷模板是不是截到淡掉的字幕。"""
        return float(np.median(self.template[self.mask]))

    def _match(self, gray: np.ndarray) -> np.ndarray:
        """回傳分數圖 S，S[y, x] = 模板左上角放在 (x, y) 時「各段 NCC 的最小值」。"""
        th, tw = self.template.shape
        H, W = gray.shape
        if H < th or W < tw:
            raise ValueError(
                f"搜尋區域 {W}x{H} 比模板 {tw}x{th} 還小，請重新校正"
            )
        out_w = W - tw + 1
        score = None
        for x0, x1 in self.segments:
            r = cv2.matchTemplate(gray, self.template[:, x0:x1], cv2.TM_CCOEFF_NORMED)
            # 這段的分數圖 r[y, x'] 是「這段左上角在 x'」的分數；
            # 整張模板左上角在 x 時，這段的左上角在 x + x0，所以要往右平移 x0 對齊。
            r = r[:, x0 : x0 + out_w]
            score = r if score is None else np.minimum(score, r)
        # 視窗內完全沒有變化（例如純黑）時，OpenCV 內部做了保護，但仍把可能的
        # NaN / inf 清掉，避免 minMaxLoc 回傳無意義的位置。
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    def detect(self, frame: np.ndarray) -> Detection:
        gray = to_gray(frame)
        result = self._match(gray)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        x, y = max_loc
        th, tw = self.template.shape
        patch = gray[y : y + th, x : x + tw]
        # 用中位數而不是平均：就算比對位置差了 1px，有少數像素落在背景或陰影上，
        # 中位數仍然由多數的文字像素決定，比平均值穩健。
        brightness = float(np.median(patch[self.mask]))

        score = float(max_val)
        return Detection(
            score=score,
            loc=(int(x), int(y)),
            brightness=brightness,
            matched=score >= self.match_threshold,
            fresh=brightness >= self.fresh_brightness,
        )

    def second_best_score(self, frame: np.ndarray) -> float:
        """把最佳位置附近遮掉後的第二高分，用來評估模板「獨特性」。

        若第二高分也很接近 1，代表畫面上有其他長得很像的東西，門檻要調高
        或模板要框大一點（例如整行字而不是一個詞）。
        """
        result = self._match(to_gray(frame))
        _, _, _, (x, y) = cv2.minMaxLoc(result)
        th, tw = self.template.shape
        x0, y0 = max(0, x - tw // 2), max(0, y - th // 2)
        result[y0 : y + th // 2 + 1, x0 : x + tw // 2 + 1] = -1.0
        return float(result.max())
