"""Brand-style labels drawn onto MuJoCo frames (BRAND.md colours, Press Start 2P pills).

Spec (JSON), every key optional:
  {"pills": [{"text": "BRAIN: ESP32-S3", "color": "GREEN", "pos": "tr", "from": 0, "to": 99}],
   "counter": {"label": "DISTANCE", "pos": "bl"},            # live metres walked
   "chip_us": {"pos": "tr2"},                                # live Chip microseconds (chip brain only)
   "fallen": true,                                           # red FALLEN stamp once the dog is down
   "clock": {"label": "MINUTE", "value": "2", "pos": "tl"}} # a fixed big caption
Text may use {dist} and {t}. Positions: tl tr bl br tc bc, and tl2/tr2 for a second row.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_F = Path(__file__).resolve().parents[3] / "resources" / "fonts" / "PressStart2P-Regular.ttf"
FONT = _F if _F.exists() else Path("/System/Library/Fonts/Menlo.ttc")   # the channel's pixel font, else Menlo
C = {"INK": "#ECEFF4", "DIM": "#8A93A3", "GREEN": "#7BD88F", "BLUE": "#58C4DD", "YELLOW": "#F4D35E",
     "RED": "#FC6255", "ORANGE": "#FF9F43", "PURPLE": "#B48EAD", "CARD": "#161B22", "EDGE": "#30363D"}


class Hud:
    def __init__(self, spec, w, h):
        self.spec, self.w, self.h = spec or {}, w, h
        self.k = h / 1080
        self.font = ImageFont.truetype(str(FONT), round(30 * self.k))
        self.big = ImageFont.truetype(str(FONT), round(64 * self.k))
        # numbers: bold Menlo, because the pixel font draws 8 like S (BRAND.md)
        self.num = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", round(34 * self.k), index=1)
        self.num_big = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", round(66 * self.k), index=1)
        self.margin = round(56 * self.k)

    def _anchor(self, pos, bw, bh):
        m, w, h = self.margin, self.w, self.h
        row2 = bh + round(18 * self.k)
        return {
            "tl": (m, m), "tr": (w - m - bw, m), "bl": (m, h - m - bh), "br": (w - m - bw, h - m - bh),
            "tc": ((w - bw) // 2, m), "bc": ((w - bw) // 2, h - m - bh),
            "tl2": (m, m + row2), "tr2": (w - m - bw, m + row2),
            "c": ((w - bw) // 2, (h - bh) // 2 - round(150 * self.k)),
        }[pos]

    def pill(self, d, text, color="INK", pos="tl", font=None):
        if any(ch.isdigit() for ch in text):
            font = self.num_big if font is self.big else self.num
        font = font or self.font
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
        px, py = round(26 * self.k), round(20 * self.k)
        bw, bh = x1 - x0 + 2 * px, y1 - y0 + 2 * py
        x, y = self._anchor(pos, bw, bh)
        d.rounded_rectangle([x, y, x + bw, y + bh], radius=round(14 * self.k), fill=C["CARD"] + "E6",
                            outline=C.get(color, color), width=max(2, round(4 * self.k)))
        d.text((x + px - x0, y + py - y0), text, font=font, fill=C.get(color, color))

    def draw(self, frame, t=0.0, dist=0.0, fallen=False, us=None):
        s = self.spec
        if not s:
            return frame
        img = Image.fromarray(frame).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for p in s.get("pills", []):
            if p.get("from", 0) <= t <= p.get("to", 1e9):
                self.pill(d, p["text"].format(dist=dist, t=t), p.get("color", "INK"), p.get("pos", "tl"))
        if "clock" in s:
            c = s["clock"]
            self.pill(d, f"{c['label']} {c['value']}", c.get("color", "YELLOW"), c.get("pos", "tl"), self.big)
        if "counter" in s:
            c = s["counter"]
            self.pill(d, f"{c.get('label', 'DISTANCE')} {max(dist, 0):5.1f} M", c.get("color", "INK"),
                      c.get("pos", "bl"))
        if "chip_us" in s and us is not None:
            self.pill(d, f"CHIP: {us} µS", "GREEN", s["chip_us"].get("pos", "tr2"))
        if s.get("fallen") and fallen:
            self.pill(d, "FALLEN", "RED", s.get("fallen_pos", "c"), self.big)
        return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))
