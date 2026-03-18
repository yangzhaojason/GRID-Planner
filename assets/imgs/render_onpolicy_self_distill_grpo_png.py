from pathlib import Path
from math import atan2, cos, sin, pi

from PIL import Image, ImageDraw, ImageFilter, ImageFont


OUT_PATH = Path(__file__).with_name("onpolicy_self_distill_grpo.png")

W, H = 2800, 1780
BG = "#FBFCFE"
TEXT = "#17212B"
MUTED = "#5D6B7A"
LINE = "#2B3642"

BLUE = "#DDEAF7"
GREEN = "#DDEFD8"
ORANGE = "#F9DFC8"
GRAY = "#ECEFF3"
TAG = "#FCE9CC"
RED = "#B04545"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


FONT_REG = font("/System/Library/Fonts/Supplemental/Arial.ttf", 40)
FONT_BOLD = font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
FONT_SMALL = font("/System/Library/Fonts/Supplemental/Arial.ttf", 34)
FONT_SMALL_BOLD = font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)
FONT_TINY = font("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
FONT_FORMULA = font("/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf", 48)
FONT_FORMULA_SMALL = font("/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf", 42)
FONT_FORMULA_BOLD = font("/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf", 50)


def rounded_box(img: Image.Image, box, fill, outline=LINE, radius=28, shadow=True, shadow_offset=(0, 12)):
    x1, y1, x2, y2 = box
    if shadow:
        shadow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow_layer)
        ox, oy = shadow_offset
        sd.rounded_rectangle((x1 + ox, y1 + oy, x2 + ox, y2 + oy), radius=radius, fill=(33, 43, 54, 40))
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(14))
        img.alpha_composite(shadow_layer)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=4)


def centered_text(draw: ImageDraw.ImageDraw, xy, text, font_obj, fill=TEXT, spacing=8):
    draw.multiline_text(xy, text, font=font_obj, fill=fill, anchor="mm", align="center", spacing=spacing)


def left_text(draw: ImageDraw.ImageDraw, xy, text, font_obj, fill=TEXT, spacing=8):
    draw.multiline_text(xy, text, font=font_obj, fill=fill, anchor="la", align="left", spacing=spacing)


def arrow(draw: ImageDraw.ImageDraw, start, end, fill=LINE, width=7, head_len=26, head_angle=0.55):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=fill, width=width)
    ang = atan2(y2 - y1, x2 - x1)
    a1 = ang + pi - head_angle
    a2 = ang + pi + head_angle
    p1 = (x2 + head_len * cos(a1), y2 + head_len * sin(a1))
    p2 = (x2 + head_len * cos(a2), y2 + head_len * sin(a2))
    draw.polygon([end, p1, p2], fill=fill)


def vertical_connector(draw: ImageDraw.ImageDraw, x, y1, y2, fill=LINE, width=6):
    draw.line((x, y1, x, y2), fill=fill, width=width)


def tag_box(img: Image.Image, box, fill, text, font_obj, text_fill=TEXT):
    rounded_box(img, box, fill=fill, outline=LINE, radius=22, shadow=False)
    draw = ImageDraw.Draw(img)
    centered_text(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), text, font_obj, fill=text_fill)


def accent_row(draw: ImageDraw.ImageDraw, x, y, color, label, formula, note=None):
    draw.rounded_rectangle((x, y + 8, x + 20, y + 54), radius=8, fill=color)
    draw.text((x + 42, y), label, font=FONT_SMALL_BOLD, fill=TEXT)
    draw.text((x + 270, y), formula, font=FONT_FORMULA_SMALL, fill=TEXT)
    if note:
        draw.text((x + 290, y + 52), note, font=FONT_TINY, fill=MUTED)


img = Image.new("RGBA", (W, H), BG)
draw = ImageDraw.Draw(img)

# Title
title_box = (80, 58, 980, 182)
rounded_box(img, title_box, fill="#FFFFFF", radius=26)
centered_text(draw, (530, 106), "(a) Unified On-Policy Self-Distillation", FONT_BOLD, fill=TEXT)
centered_text(draw, (530, 149), "for Flow Planning", FONT_BOLD, fill=TEXT)

tag_box(img, (2100, 72, 2660, 146), TAG, "Current Training: GRPO, Critic-Free", FONT_SMALL_BOLD)

# Scene context and planner
centered_text(draw, (1400, 88), "Scene context G", FONT_BOLD, fill=TEXT)
centered_text(draw, (1400, 132), "(ego history, neighbors, map, route)", FONT_SMALL, fill=MUTED)
arrow(draw, (1400, 168), (1400, 232))

planner_box = (1085, 238, 1715, 372)
rounded_box(img, planner_box, fill=BLUE)
centered_text(draw, (1400, 288), "Flow Planner πθ", FONT_BOLD, fill=TEXT)
centered_text(draw, (1400, 338), "(current policy)", FONT_SMALL_BOLD, fill=MUTED)

# Left rollout branch
left_text(draw, (160, 346), "On-policy rollout branch", FONT_SMALL_BOLD, fill=TEXT)
left_text(draw, (160, 394), "sample K trajectories τ1, ..., τK", FONT_SMALL, fill=MUTED)
arrow(draw, (1160, 372), (1000, 438))
arrow(draw, (1400, 372), (1400, 438))

reward_box = (1095, 444, 1705, 574)
rounded_box(img, reward_box, fill=GREEN)
centered_text(draw, (1400, 488), "RRS proxy reward on K rollouts", FONT_BOLD, fill=TEXT)
centered_text(draw, (1400, 536), "evaluate r̃(τk, G) for each sampled rollout", FONT_SMALL, fill=MUTED)

rank_box = (930, 650, 1870, 850)
rounded_box(img, rank_box, fill=GREEN)
centered_text(draw, (1400, 708), "GRPO group-relative ranking", FONT_BOLD, fill=TEXT)
centered_text(
    draw,
    (1400, 764),
    "Agrp,k = (r̃k - meanj r̃j) / (stdj r̃j + ε)",
    FONT_FORMULA_SMALL,
    fill=TEXT,
)
centered_text(
    draw,
    (1400, 816),
    "center / normalize across K\npositive-advantage weighting; select τ+",
    FONT_TINY,
    fill=MUTED,
    spacing=6,
)
arrow(draw, (1400, 574), (1400, 650))

# Student branch and projection
left_text(draw, (1868, 338), "Deterministic prediction branch", FONT_SMALL_BOLD, fill=TEXT)
left_text(draw, (1868, 386), "student output fθ(G)", FONT_SMALL, fill=MUTED)
arrow(draw, (1715, 304), (2280, 304))
vertical_connector(draw, 2280, 304, 608)
arrow(draw, (2280, 608), (2280, 650))

proj_box = (1915, 656, 2640, 840)
rounded_box(img, proj_box, fill=ORANGE)
centered_text(draw, (2278, 708), "Self-distillation projection", FONT_BOLD, fill=TEXT)
centered_text(draw, (2278, 762), "D(fθ(G), τ+)", FONT_FORMULA_BOLD, fill=TEXT)
centered_text(draw, (2278, 814), "project the student output onto the selected improved rollout", FONT_TINY, fill=MUTED)

# Gate
gate_box = (1088, 916, 1822, 994)
rounded_box(img, gate_box, fill="#F4F7FB", radius=22)
centered_text(draw, (1455, 955), "Improvement-aware gate  ω(G, τ+) from Agrp and ΔR", FONT_SMALL_BOLD, fill=TEXT)
left_text(draw, (1950, 508), "ΔR compares τ+ with the\ncurrent deterministic output", FONT_TINY, fill=MUTED)
arrow(draw, (1400, 850), (1455, 916))
arrow(draw, (1822, 955), (1915, 748))

# FM anchor
fm_box = (190, 724, 890, 868)
rounded_box(img, fm_box, fill=GRAY)
centered_text(draw, (540, 780), "FM anchor / supervised prior", FONT_BOLD, fill=TEXT)
centered_text(draw, (540, 828), "LFM(θ) on dataset trajectories for stability", FONT_SMALL, fill=MUTED)

# Objective panel
obj_box = (160, 1080, 2640, 1658)
rounded_box(img, obj_box, fill="#FFFFFF", radius=30)
draw.text((220, 1142), "Final objective", font=FONT_BOLD, fill=TEXT)
draw.text((220, 1216), "Ltotal(θ) =", font=FONT_FORMULA_BOLD, fill=TEXT)

accent_row(draw, 520, 1202, "#C9CED7", "FM term", "LFM(θ)")
accent_row(
    draw,
    520,
    1302,
    "#9BCB8F",
    "GRPO term",
    "λRL · meank [ ediff(τk, G) · stopgrad(Agrp,k) ]",
    note="critic-free rollout weighting; no separately trained value head",
)
accent_row(
    draw,
    520,
    1422,
    "#F0B983",
    "Distill term",
    "ω(G, τ+) · D(fθ(G), τ+)",
    note="gate combines group-relative improvement and output-vs-rollout gain",
)

tag_box(img, (1940, 1160, 2525, 1234), "#FDE6E3", "No critic / no λV·Lvalue", FONT_SMALL_BOLD, text_fill=RED)

draw.text(
    (220, 1556),
    "Critic-free GRPO samples a rollout group, computes group-relative scores, and uses them to weight both the "
    "diffusion surrogate and the self-distillation target.",
    font=FONT_TINY,
    fill=MUTED,
)

# Connectors to objective
arrow(draw, (540, 868), (540, 1080))
arrow(draw, (1455, 994), (1455, 1080))
arrow(draw, (2278, 840), (2278, 1080))
draw.text((2315, 938), "update planner θ", font=FONT_SMALL_BOLD, fill=MUTED)

img = img.convert("RGB")
img.save(OUT_PATH, quality=100)
print(f"Saved {OUT_PATH}")
