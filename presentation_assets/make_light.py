"""Lighter copies of the renders for the deck: JPEG composited on the slide background colour,
and a downscaled hero PNG (keeps transparency). Run after make_figures.py."""
import os
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))
BG = (11, 14, 18)
for n in ["fr2_cloud", "fr1_cloud", "synth_cloud", "corrected_fr2", "legacy_fr2", "strip_rgb", "thumb_fr1",
          "thumb_fr2", "thumb_synth", "bug_old_kept", "bug_new_depth", "strip_depth"]:
    im = Image.open(n + ".png").convert("RGBA")
    if im.width > 2000:
        im = im.resize((2000, round(im.height * 2000 / im.width)), Image.LANCZOS)
    base = Image.new("RGBA", im.size, BG + (255,))
    base.alpha_composite(im)
    base.convert("RGB").save(n + ".jpg", quality=90, optimize=True)
h = Image.open("hero_fr2.png")
h.resize((2400, round(h.height * 2400 / h.width)), Image.LANCZOS).save("hero_fr2_web.png", optimize=True)
print("light copies written")
