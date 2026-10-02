"""Build demo/project_demo.mp4 from the committed result charts.

Run from the repo root:
    python scripts/make_demo_video.py
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "project_demo.mp4"
W, H = 1280, 720
FPS = 8

BG = (15, 23, 42)
CARD = (30, 41, 59)
INK = (248, 250, 252)
MUTED = (148, 163, 184)
ACCENT = (56, 189, 248)
AMBER = (251, 191, 36)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = Path("/System/Library/Fonts/Supplemental") / name
    return ImageFont.truetype(str(path), size)


def _blank() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def _header(draw: ImageDraw.ImageDraw, kicker: str, title: str) -> int:
    draw.text((64, 36), kicker.upper(), font=_font(18, True), fill=ACCENT)
    draw.text((64, 68), title, font=_font(36, True), fill=INK)
    return 130


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, x: int, y: int, max_w: int, fill, gap: int = 8) -> int:
    words = text.split()
    line = ""
    for word in words:
        trial = word if not line else f"{line} {word}"
        if draw.textlength(trial, font=font) <= max_w:
            line = trial
        else:
            draw.text((x, y), line, font=font, fill=fill)
            y += font.size + gap
            line = word
    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + gap
    return y


def _fit(path: Path, box: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(box, Image.Resampling.LANCZOS)
    return img


def _paste_centered(canvas: Image.Image, img: Image.Image, cx: int, cy: int) -> None:
    canvas.paste(img, (cx - img.width // 2, cy - img.height // 2))


def slide_title() -> Image.Image:
    img = _blank()
    d = ImageDraw.Draw(img)
    d.text((64, 180), "PROJECT DEMO", font=_font(20, True), fill=ACCENT)
    d.text((64, 220), "Algorithmic Trading", font=_font(58, True), fill=INK)
    d.text((64, 290), "Signal Predictor", font=_font(58, True), fill=INK)
    body = (
        "A leakage-safe machine-learning pipeline that predicts next-day "
        "up or down for RELIANCE.NS, then backtests the signal against buy-and-hold."
    )
    _wrap(d, body, _font(26), 64, 400, 1000, MUTED, gap=10)
    d.text((64, 560), "Python   ·   pandas   ·   scikit-learn   ·   yfinance", font=_font(22), fill=AMBER)
    return img


def slide_pipeline() -> Image.Image:
    img = _blank()
    d = ImageDraw.Draw(img)
    _header(d, "How it runs", "Four steps, one command")
    steps = [
        ("1  Data", "Ten years of daily OHLCV, cached as parquet so the run does not depend on Yahoo."),
        ("2  Features", "Ten stationary inputs: returns, SMA and EMA ratios, RSI, MACD, volatility, volume."),
        ("3  Models", "Logistic regression, random forest, and SVM. Tuned with TimeSeriesSplit."),
        ("4  Backtest", "Long/flat over the last 252 days, with a 0.1% cost when the position changes."),
    ]
    y = 150
    for title, body in steps:
        d.rounded_rectangle((64, y, 1216, y + 110), radius=12, fill=CARD)
        d.text((88, y + 16), title, font=_font(26, True), fill=INK)
        d.text((88, y + 58), body, font=_font(22), fill=MUTED)
        y += 126
    return img


def slide_chart(kicker: str, title: str, caption: str, chart: Path, box=(1080, 430)) -> Image.Image:
    img = _blank()
    d = ImageDraw.Draw(img)
    _header(d, kicker, title)
    pic = _fit(chart, box)
    frame = Image.new("RGB", (pic.width + 16, pic.height + 16), (226, 232, 240))
    frame.paste(pic, (8, 8))
    _paste_centered(img, frame, W // 2, 360)
    _wrap(d, caption, _font(22), 64, 620, 1150, MUTED, gap=6)
    return img


def slide_numbers() -> Image.Image:
    img = _blank()
    d = ImageDraw.Draw(img)
    _header(d, "Out-of-sample year", "The strategy does not beat buy-and-hold")
    rows = [
        ("", "ML strategy", "Buy & hold"),
        ("Total return", "−11.9%", "−4.6%"),
        ("Sharpe ratio", "−0.93", "−0.42"),
        ("Max drawdown", "−22.1%", "−20.6%"),
        ("Trades", "61", "—"),
    ]
    y = 170
    for i, (a, b, c) in enumerate(rows):
        fill = (51, 65, 85) if i == 0 else CARD
        d.rounded_rectangle((160, y, 1120, y + 72), radius=10, fill=fill)
        font = _font(24, True) if i == 0 else _font(26)
        color = ACCENT if i == 0 else INK
        d.text((190, y + 20), a, font=font, fill=color)
        d.text((640, y + 20), b, font=font, fill=color)
        d.text((900, y + 20), c, font=font, fill=color)
        y += 84
    d.text((160, 640), "Test accuracy 49.2%   ·   majority baseline 48.8%   ·   Reliance, 252 sessions",
           font=_font(22), fill=MUTED)
    return img


def slide_run() -> Image.Image:
    img = _blank()
    d = ImageDraw.Draw(img)
    _header(d, "Reproduce it", "Clone, install, run")
    commands = [
        "pip install -r requirements.txt",
        "python -m main",
        "python -m main --ticker ^NSEI",
    ]
    y = 180
    for cmd in commands:
        d.rounded_rectangle((64, y, 1100, y + 78), radius=12, fill=CARD)
        d.text((92, y + 22), cmd, font=_font(28), fill=ACCENT)
        y += 100
    note = (
        "Prices are already cached, so this works without a network call. "
        "The test window is never used to choose the model."
    )
    _wrap(d, note, _font(24), 64, 520, 1100, MUTED, gap=8)
    return img


def slide_close() -> Image.Image:
    img = _blank()
    d = ImageDraw.Draw(img)
    d.text((64, 230), "What to take from it", font=_font(22, True), fill=ACCENT)
    d.text((64, 280), "Accuracy near 50% is the", font=_font(46, True), fill=INK)
    d.text((64, 345), "honest result.", font=_font(46, True), fill=INK)
    note = (
        "The useful part is the discipline: chronological split, scaling inside "
        "the training folds, model choice by cross-validation, and a costed backtest."
    )
    _wrap(d, note, _font(26), 64, 450, 1050, MUTED, gap=10)
    return img


def main() -> None:
    slides = [
        (slide_title(), 8),
        (slide_pipeline(), 12),
        (slide_chart(
            "Model selection",
            "Random forest wins on cross-validation",
            "The test set is not used to pick the winner. On that set it scores 49.2%, against a 48.8% baseline.",
            ROOT / "artifacts" / "model_comparison.png",
        ), 11),
        (slide_chart(
            "Backtest",
            "Cumulative growth of 1 unit",
            "After 0.1% transaction costs the forest strategy finishes below buy-and-hold.",
            ROOT / "artifacts" / "equity_curves.png",
            box=(1000, 400),
        ), 11),
        (slide_numbers(), 12),
        (slide_chart(
            "Errors",
            "It predicts Up on most days",
            "186 of 252 test days are called Up. That is why accuracy stays near the base rate.",
            ROOT / "artifacts" / "confusion_matrix.png",
            box=(520, 430),
        ), 9),
        (slide_chart(
            "What the forest uses",
            "MACD histogram leads",
            "Next are the 1-day return and 10-day volatility. All ten features are available at the close of day t.",
            ROOT / "artifacts" / "feature_importance.png",
            box=(860, 420),
        ), 9),
        (slide_run(), 10),
        (slide_close(), 8),
    ]

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    OUT.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = Path(tmp)
        n = 0
        for slide, seconds in slides:
            for _ in range(seconds * FPS):
                slide.save(frame_dir / f"{n:04d}.png")
                n += 1
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(FPS),
            "-i", str(frame_dir / "%04d.png"),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(OUT),
        ]
        subprocess.run(cmd, check=True)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
