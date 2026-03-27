from __future__ import annotations

from math import atan2, cos, sin
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_PATH = Path(__file__).with_name("onpolicy_self_distill_grpo.png")

FINAL_W = 2400
FINAL_H = 1450
SCALE = 2
W = FINAL_W * SCALE
H = FINAL_H * SCALE


def s(v: int | float) -> int:
    return int(round(v * SCALE))


TEXT = "#1F1F1F"
TEXT_SUB = "#404040"
LINE = "#202020"
BG = "#FFFFFF"
BLUE = "#DDEAF7"
GREEN = "#DCEECD"
ORANGE = "#F9D0A5"
GRAY = "#ECECEC"
RED = "#B13A2F"
RED_BG = "#FBE5E1"


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, s(size))


FONT_SANS = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_SANS_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_SERIF = "/System/Library/Fonts/Supplemental/Georgia.ttf"
FONT_SERIF_ITALIC = "/System/Library/Fonts/Supplemental/Georgia Italic.ttf"
FONT_SERIF_BOLD_ITALIC = "/System/Library/Fonts/Supplemental/Georgia Bold Italic.ttf"

TITLE_FONT = load_font(FONT_SANS, 31)
TITLE_SUB_FONT = load_font(FONT_SANS, 29)
LABEL_FONT = load_font(FONT_SANS, 26)
TEXT_FONT = load_font(FONT_SANS, 24)
NOTE_FONT = load_font(FONT_SANS, 20)
BOX_TITLE_FONT = load_font(FONT_SANS_BOLD, 28)
BOX_TEXT_FONT = load_font(FONT_SANS, 24)
BOX_NOTE_FONT = load_font(FONT_SANS, 20)
FORMULA_FONT = load_font(FONT_SERIF_ITALIC, 28)
FORMULA_BIG_FONT = load_font(FONT_SERIF_BOLD_ITALIC, 30)
OBJ_TITLE_FONT = load_font(FONT_SANS, 30)
OBJ_FORMULA_FONT = load_font(FONT_SERIF_BOLD_ITALIC, 29)
OBJ_NOTE_FONT = load_font(FONT_SANS, 18)
BADGE_FONT = load_font(FONT_SANS_BOLD, 21)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str = TEXT,
) -> None:
    draw.text((s(xy[0]), s(xy[1])), text, font=font, fill=fill, anchor="mm")


def draw_multiline_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    fill: str = TEXT,
    spacing: int = 8,
) -> None:
    draw.multiline_text(
        (s(xy[0]), s(xy[1])),
        "\n".join(lines),
        font=font,
        fill=fill,
        anchor="mm",
        align="center",
        spacing=s(spacing),
    )


def draw_multiline_left(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    fill: str = TEXT,
    spacing: int = 8,
) -> None:
    draw.multiline_text(
        (s(xy[0]), s(xy[1])),
        "\n".join(lines),
        font=font,
        fill=fill,
        anchor="lm",
        align="left",
        spacing=s(spacing),
    )


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str = LINE,
    radius: int = 16,
    width: int = 2,
) -> None:
    x, y, w, h = box
    draw.rounded_rectangle(
        (s(x), s(y), s(x + w), s(y + h)),
        radius=s(radius),
        fill=fill,
        outline=outline,
        width=s(width),
    )


def callout_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str = LINE,
    notch: int = 26,
    width: int = 2,
) -> None:
    x, y, w, h = box
    pts = [
        (s(x), s(y)),
        (s(x + w), s(y)),
        (s(x + w), s(y + h - notch)),
        (s(x + w - notch), s(y + h)),
        (s(x), s(y + h)),
    ]
    draw.polygon(pts, fill=fill, outline=outline)
    draw.line(pts + [pts[0]], fill=outline, width=s(width))


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    fill: str = LINE,
    width: int = 3,
    head_len: int = 14,
) -> None:
    scaled = [(s(x), s(y)) for x, y in points]
    for p1, p2 in zip(scaled, scaled[1:]):
        draw.line((p1[0], p1[1], p2[0], p2[1]), fill=fill, width=s(width))

    (x1, y1), (x2, y2) = scaled[-2], scaled[-1]
    ang = atan2(y2 - y1, x2 - x1)
    head = s(head_len)
    left = (x2 - head * cos(ang - 0.45), y2 - head * sin(ang - 0.45))
    right = (x2 - head * cos(ang + 0.45), y2 - head * sin(ang + 0.45))
    draw.polygon([(x2, y2), left, right], fill=fill)


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Title and header
    callout_box(draw, (110, 70, 860, 118), fill="#F8F8F8")
    draw_multiline_centered(
        draw,
        (505, 128),
        ["(a) Unified On-Policy Self-Distillation", "for Flow Planning"],
        TITLE_FONT,
        spacing=4,
    )

    draw_multiline_centered(
        draw,
        (1410, 118),
        ["Scene context G", "(ego history, neighbors, map, route)"],
        LABEL_FONT,
        fill=TEXT,
        spacing=2,
    )
    draw_arrow(draw, [(1410, 150), (1410, 260)])

    rounded_box(draw, (2010, 84, 280, 42), fill="#FFF2D8")
    draw_centered_text(draw, (2150, 105), "GRPO / critic-free", BADGE_FONT)

    # Main planner
    planner_box = (1260, 250, 400, 110)
    rounded_box(draw, planner_box, fill=BLUE)
    draw_centered_text(draw, (1460, 302), "Flow Planner πθ", BOX_TITLE_FONT)
    draw_centered_text(draw, (1460, 342), "(current policy)", BOX_TEXT_FONT)
    draw_arrow(draw, [(1460, 150), (1460, 250)])

    # Branch labels
    draw_centered_text(draw, (675, 438), "On-policy rollout branch", LABEL_FONT)
    draw_centered_text(draw, (675, 478), "sample K trajectories τ₁, ..., τ_K", TEXT_FONT)
    draw_centered_text(draw, (1965, 438), "Deterministic prediction", LABEL_FONT)
    draw_centered_text(draw, (1965, 478), "fθ(G) / student output", TEXT_FONT)
    draw_arrow(draw, [(1315, 360), (1165, 430), (875, 430)])
    draw_arrow(draw, [(1605, 360), (1775, 430), (1935, 430)])

    # Reward proxy
    rrs_box = (1255, 500, 410, 118)
    rounded_box(draw, rrs_box, fill=GREEN)
    draw_centered_text(draw, (1460, 539), "RRS reward proxy", BOX_TITLE_FONT)
    draw_centered_text(
        draw,
        (1460, 581),
        "r(τ, G) := shape( - w_G(G) · C_RRS(τ, G) )",
        FORMULA_FONT,
    )
    draw_arrow(draw, [(1460, 360), (1460, 500)])

    # Group-relative scoring / selection
    group_box = (1230, 675, 460, 150)
    rounded_box(draw, group_box, fill=GREEN)
    draw_centered_text(draw, (1460, 717), "Improvement step", BOX_TITLE_FONT)
    draw_centered_text(
        draw,
        (1460, 760),
        "A_grp({τ_k}, G) from {r(τ_k, G)}",
        FORMULA_FONT,
    )
    draw_multiline_centered(
        draw,
        (1460, 800),
        ["select τ+", "(best-of-K / positive-adv)"],
        BOX_TEXT_FONT,
        spacing=2,
    )
    draw_arrow(draw, [(1460, 618), (1460, 675)])

    # Left annotation near scoring
    draw_multiline_left(
        draw,
        (230, 760),
        ["compute rollout reward r(τ_k, G)", "and group score A_grp(τ_k, G)"],
        NOTE_FONT,
        fill=TEXT_SUB,
        spacing=4,
    )

    # Projection branch
    proj_box = (1815, 770, 500, 132)
    rounded_box(draw, proj_box, fill=ORANGE)
    draw_centered_text(draw, (2065, 812), "Self-distillation projection", BOX_TITLE_FONT)
    draw_centered_text(draw, (2065, 854), "D( fθ(G), τ+ )", FORMULA_BIG_FONT)

    draw_arrow(draw, [(1940, 500), (1940, 770)])

    # Compare with current deterministic output to get the improvement gap Delta R.
    draw_arrow(draw, [(1815, 805), (1745, 805), (1745, 735), (1690, 735)])
    # Gated projection target from the selected rollout.
    draw_arrow(draw, [(1690, 796), (1760, 796), (1760, 842), (1815, 842)])

    draw_multiline_left(
        draw,
        (1835, 630),
        ["compare τ+ with current output", "to get improvement gap ΔR"],
        NOTE_FONT,
        fill=TEXT_SUB,
        spacing=4,
    )
    draw_centered_text(draw, (1710, 895), "gate ω(G, τ+) = ω(A_grp, ΔR)", NOTE_FONT)

    # FM anchor
    fm_box = (610, 1010, 730, 122)
    rounded_box(draw, fm_box, fill=GRAY)
    draw_centered_text(draw, (975, 1060), "FM anchor / supervised prior", BOX_TITLE_FONT)
    draw_centered_text(draw, (975, 1100), "L_FM(θ) on dataset trajectory", FORMULA_BIG_FONT)

    # Objective box
    obj_box = (360, 1182, 1800, 220)
    rounded_box(draw, obj_box, fill="#F7F7F7", radius=0)
    draw_centered_text(draw, (540, 1218), "Final objective:", TITLE_SUB_FONT)
    draw_multiline_left(
        draw,
        (545, 1282),
        [
            "L_total(θ) = L_FM(θ)",
            "+ λ_RL [ e_diff(τ+, G) * stopgrad(A_grp(τ+, G)) ]",
            "+ ω(G, τ+) * D( fθ(G), τ+ )",
        ],
        OBJ_FORMULA_FONT,
        spacing=10,
    )
    draw_centered_text(
        draw,
        (1715, 1218),
        "critic-free: no λ_V * L_value",
        OBJ_NOTE_FONT,
        fill=RED,
    )
    draw_centered_text(
        draw,
        (1260, 1380),
        "RRS only defines the rollout reward proxy here; it is not shown as a separate additive loss term.",
        OBJ_NOTE_FONT,
        fill=TEXT_SUB,
    )

    # Objective arrows
    draw_arrow(draw, [(975, 1132), (975, 1182)])
    draw_arrow(draw, [(1460, 825), (1460, 1182)])
    draw_arrow(draw, [(2065, 902), (2065, 1182)])
    draw_centered_text(draw, (1905, 1035), "update planner θ", LABEL_FONT)

    # Save high-res result with smooth downsample
    img = img.resize((FINAL_W, FINAL_H), Image.Resampling.LANCZOS)
    img.save(OUT_PATH, quality=100)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
