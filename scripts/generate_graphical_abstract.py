"""Generate an improved graphical abstract for the Quantum Active Learning paper.

Outputs:
- graphical_abstract.png (2048x1152)
- graphical_abstract.svg

Dependencies: Pillow, matplotlib, cairosvg (for SVG export, optional). Install via:

pip install pillow matplotlib cairosvg

Run:
python tools/generate_graphical_abstract.py
"""
from PIL import Image, ImageDraw, ImageFont
import math
import os
import textwrap

OUT_PNG = os.path.join(os.path.dirname(__file__), "..", "graphical_abstract.png")
OUT_SVG = os.path.join(os.path.dirname(__file__), "..", "graphical_abstract.svg")

W, H = 2048, 1152
BG = (255, 255, 255)

# Try to load a default sans-serif font; fall back to default PIL font
try:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    FONT_BOLD = ImageFont.truetype(FONT_PATH, 56)
    FONT_REG = ImageFont.truetype(FONT_PATH, 36)
    FONT_SMALL = ImageFont.truetype(FONT_PATH, 28)
    FONT_MONO = ImageFont.truetype(FONT_PATH, 24)
except Exception:
    FONT_BOLD = ImageFont.load_default()
    FONT_REG = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()
    FONT_MONO = ImageFont.load_default()

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# Colors
Q_BLUE = (12, 97, 175)
ACCENT = (233, 98, 42)
GRAY = (90, 90, 95)
BOX_COL = (240, 245, 250)

padding = 80
col_w = (W - padding*2 - 40) // 3
col_h = H - padding*2

# Title
title = "Quantum-Enhanced Active Learning"
subtitle = "Accelerating materials discovery with quantum-inspired uncertainty"

title_x = padding
title_y = 40
d.text((title_x, title_y), title, fill=Q_BLUE, font=FONT_BOLD)
# subtitle wrap to fit width
subtitle_lines = textwrap.wrap(subtitle, width=70)
for i, line in enumerate(subtitle_lines):
    d.text((title_x, title_y+70 + i*38), line, fill=GRAY, font=FONT_REG)

# Draw three panels: Materials -> Quantum States -> Selection/Experiments
panel_y = 180
panel_h = 700

# Panel centers
centers = [padding + col_w//2 + i*(col_w+20) for i in range(3)]

# Draw panels
for i, cx in enumerate(centers):
    left = cx - col_w//2
    top = panel_y
    right = cx + col_w//2
    bottom = top + panel_h
    # rounded rect aesthetic: simple rectangle with subtle border
    d.rectangle([left, top, right, bottom], fill=BOX_COL, outline=(220,220,220))

def draw_panel_text(center_x, top_y, text, max_width, font, fill=GRAY):
    # wrap text to fit the max_width and center it under top_y
    lines = textwrap.wrap(text, width=30)
    # measure total height
    if lines:
        bbox = font.getbbox(lines[0])
        line_h = bbox[3] - bbox[1]
    else:
        line_h = 0
    total_h = line_h * len(lines)
    start_y = top_y
    for i, line in enumerate(lines):
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
        d.text((center_x - w//2, start_y + i*line_h), line, fill=fill, font=font)

# Panel 1: Materials
m_cx = centers[0]
mc_y = panel_y + 80
for i in range(3):
    cx = m_cx - 120 + i*80
    cy = mc_y
    size = 48
    d.polygon([(cx, cy-size), (cx+size, cy), (cx, cy+size), (cx-size, cy)], fill=(168,216,255), outline=(120,160,200))
draw_panel_text(m_cx, mc_y + 90, "Candidate materials (DFT & datasets)", col_w-40, FONT_REG)

# Panel 2: Quantum states
q_cx = centers[1]
qy = panel_y + 40
r = 100
d.ellipse([q_cx-r, qy, q_cx+r, qy+2*r], outline=Q_BLUE, width=6)
for a in range(5):
    d.arc([q_cx-r-20, qy-20, q_cx+r+20, qy+2*r+20], start=30+a*18, end=150+a*18, fill=(60,140,200))
draw_panel_text(q_cx, qy + 2*r + 20, "Quantum representation — superposition & entanglement", col_w-40, FONT_REG)

# Panel 3: Selection / experiments
s_cx = centers[2]
sy = panel_y + 80
for i in range(3):
    sx = s_cx - 80 + i*80
    syy = sy
    d.rectangle([sx-30, syy-30, sx+30, syy+30], fill=(220,255,230), outline=(170,210,170))
    d.line([(sx-18, syy), (sx-4, syy+12), (sx+20, syy-14)], fill=ACCENT, width=8)
draw_panel_text(s_cx, sy + 90, "Quantum-enhanced selection for experiments and accelerated discovery", col_w-40, FONT_REG)

# Arrows between panels
arrow_y = panel_y + panel_h//2
for i in range(2):
    x1 = centers[i] + col_w//2 - 10
    x2 = centers[i+1] - col_w//2 + 10
    y = arrow_y
    # line
    d.line([(x1, y), (x2, y)], fill=Q_BLUE, width=8)
    # arrow head
    ah = 18
    d.polygon([(x2, y), (x2-ah, y-ah//2), (x2-ah, y+ah//2)], fill=Q_BLUE)

# Footer: small notes and ORCiD
foot_y = H - 120
d.text((padding, foot_y), "Arnav Kapoor — IISER Bhopal  •  ORCiD: 0009-0007-9818-7908", fill=GRAY, font=FONT_SMALL)
essence = "Graphical abstract: encode materials as quantum states, quantify multi-observable uncertainty, select experiments to accelerate discovery."
ess_lines = textwrap.wrap(essence, width=120)
for i, line in enumerate(ess_lines):
    d.text((padding, foot_y + 36 + i*22), line, fill=(120,120,120), font=FONT_SMALL)

# Save PNG
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
img.save(OUT_PNG, "PNG")
print(f"Saved {OUT_PNG}")

# Also create a simple SVG fallback using Pillow + cairosvg if available
try:
    import cairosvg
    # create an SVG by converting the PNG (lossless vector won't be perfect but provides scalable preview)
    cairosvg.svg_from_png(OUT_PNG, write_to=OUT_SVG)
    print(f"Saved {OUT_SVG}")
except Exception:
    # If cairosvg not available, skip svg
    pass
