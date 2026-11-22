"""Generates the slide images used by the tests (run once; output is committed)."""
import os
import sys
from PIL import Image, ImageDraw, ImageFont

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "images")
os.makedirs(out, exist_ok=True)
colors = ["#1f3a5f", "#7a2e3a", "#2f5d3a", "#5b3f8c", "#8a5a14", "#2b5f6b"]
font = ImageFont.load_default(size=96)
small = ImageFont.load_default(size=36)

for n, color in enumerate(colors, start=1):
    img = Image.new("RGB", (1024, 576), color)
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 984, 536], outline="#ffffff", width=6)
    d.text((512, 250), f"Slide {n}", font=font, fill="#ffffff", anchor="mm")
    d.text((512, 360), "Test deck for Slide Saver", font=small, fill="#dddddd", anchor="mm")
    img.save(os.path.join(out, f"slide-{n}.jpg"), "JPEG", quality=70)

# A 4:3 PNG and a WebP for format handling tests.
png = Image.new("RGB", (800, 600), "#334455")
ImageDraw.Draw(png).text((400, 300), "PNG", font=font, fill="#ffffff", anchor="mm")
png.save(os.path.join(out, "slide-png.png"), "PNG")
Image.new("RGB", (640, 360), "#445566").save(os.path.join(out, "slide-webp.webp"), "WEBP")
print("ok")
