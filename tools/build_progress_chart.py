#!/usr/bin/env python3
"""Build the optimization-progress chart from committed benchmark JSONs.

Reads the native optimization round measurements in benchmarks/results/ and
writes two assets consumed by the README and the project site:

- site/assets/optimization-progress.svg  (static line chart)
- site/assets/optimization-progress.gif  (animated, line draws round by round)

Run: uv run python tools/build_progress_chart.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
OUT_DIR = ROOT / "site" / "assets"

ROUND_FILES = [
    ("round-1", "4bd6893", "native-round1-zero-cache.json"),
    ("round-2", "e3d1b87", "native-round2-direct-pread.json"),
    ("round-3", "fb4c0aa", "native-round3-cpu-modulation.json"),
    ("round-4", "40420aa", "native-round4-fused-attention-views.json"),
]

PALETTE = {
    "bg": "#0e1118",
    "grid": "#232939",
    "text": "#e8eaf0",
    "muted": "#9aa3b2",
    "faint": "#6b7488",
    "time": "#ff7a6e",
    "mem": "#46c285",
    "warn": "#e2a336",
}

W, H = 960, 540
ML, MR = 84, 52
PA_T, PA_B = 88, 244
PB_T, PB_B = 300, 420
TIME_MIN, TIME_MAX = 32.0, 45.0
MEM_MAX = 3.6


def load_points():
    r1 = json.loads((RESULTS / "native-round1-zero-cache.json").read_text())
    points = [
        {
            "label": "baseline",
            "commit": "b5bf491",
            "seconds": r1["baseline_seconds_median"],
            "peak_gib": r1["baseline_peak_bytes"] / 2**30,
        }
    ]
    for label, commit, name in ROUND_FILES:
        d = json.loads((RESULTS / name).read_text())
        peak = d.get("candidate_peak_bytes_max") or d.get("candidate_peak_bytes")
        points.append(
            {
                "label": label,
                "commit": commit,
                "seconds": d["candidate_seconds_median"],
                "peak_gib": peak / 2**30,
            }
        )
    return points


def x_at(i, n):
    return ML + (W - ML - MR) * (i / (n - 1))


def y_time(s):
    return PA_T + (PA_B - PA_T) * (1 - (s - TIME_MIN) / (TIME_MAX - TIME_MIN))


def y_mem(g):
    return PB_T + (PB_B - PB_T) * (1 - g / MEM_MAX)


def build_svg(points):
    n = len(points)
    p = []
    p.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="ui-monospace, Menlo, monospace" role="img" '
        f'aria-label="Optimization progress: 1024x1024 median wall time falls '
        f'from 42.30 s to 34.65 s over four rounds while the output hash stays identical">'
    )
    p.append(f'<rect width="{W}" height="{H}" rx="14" fill="{PALETTE["bg"]}"/>')
    p.append(
        f'<text x="{ML}" y="34" font-size="15" fill="{PALETTE["text"]}" font-weight="bold">'
        f'FOUR ROUNDS, ONE BYTE-IDENTICAL TEAPOT</text>'
    )
    p.append(
        f'<text x="{ML}" y="54" font-size="11.5" fill="{PALETTE["muted"]}">'
        f'1024×1024 · seed 42 · 6 steps · M5 Max 36 GB · output SHA-256 identical in every round</text>'
    )
    p.append(f'<text x="{ML}" y="{PA_T - 10}" font-size="11" fill="{PALETTE["time"]}">MEDIAN WALL TIME</text>')
    p.append(f'<text x="{ML}" y="{PB_T - 10}" font-size="11" fill="{PALETTE["mem"]}">PEAK PROCESS BYTES</text>')
    for t in (32, 36, 40, 44):
        y = y_time(t)
        p.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
        p.append(f'<text x="{ML - 10}" y="{y + 4:.1f}" font-size="11" fill="{PALETTE["faint"]}" text-anchor="end">{t} s</text>')
    for g in (0, 1, 2, 3):
        y = y_mem(g)
        p.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
        p.append(f'<text x="{ML - 10}" y="{y + 4:.1f}" font-size="11" fill="{PALETTE["faint"]}" text-anchor="end">{g} GiB</text>')

    def polyline(key, yfn, color):
        pts = " ".join(f"{x_at(i, n):.1f},{yfn(pt[key]):.1f}" for i, pt in enumerate(points))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')
        for i, pt in enumerate(points):
            p.append(f'<circle cx="{x_at(i, n):.1f}" cy="{yfn(pt[key]):.1f}" r="4.5" fill="{PALETTE["bg"]}" stroke="{color}" stroke-width="2.5"/>')

    polyline("seconds", y_time, PALETTE["time"])
    polyline("peak_gib", y_mem, PALETTE["mem"])

    for i, pt in enumerate(points):
        x = x_at(i, n)
        p.append(f'<text x="{x:.1f}" y="{y_time(pt["seconds"]) - 12:.1f}" font-size="11.5" fill="{PALETTE["time"]}" text-anchor="middle">{pt["seconds"]:.2f} s</text>')
        p.append(f'<text x="{x:.1f}" y="{y_mem(pt["peak_gib"]) + 18:.1f}" font-size="11.5" fill="{PALETTE["mem"]}" text-anchor="middle">{pt["peak_gib"]:.2f} GiB</text>')
        p.append(f'<text x="{x:.1f}" y="{PB_B + 26}" font-size="11.5" fill="{PALETTE["text"]}" text-anchor="middle">{pt["label"]}</text>')
        p.append(f'<text x="{x:.1f}" y="{PB_B + 42}" font-size="10.5" fill="{PALETTE["faint"]}" text-anchor="middle">{pt["commit"]}</text>')

    p.append(
        f'<text x="{x_at(1, n) + 16:.1f}" y="{y_time(points[1]["seconds"]) - 4:.1f}" font-size="10.5" '
        f'fill="{PALETTE["warn"]}">round 1 regressed; published anyway</text>'
    )
    speedup = points[0]["seconds"] / points[-1]["seconds"]
    mem_drop = 100 * (1 - points[-1]["peak_gib"] / points[0]["peak_gib"])
    p.append(
        f'<text x="{W - MR}" y="{H - 18}" font-size="12" fill="{PALETTE["text"]}" text-anchor="end">'
        f'42.30 s → 34.65 s · {speedup:.2f}× faster · −{mem_drop:.0f}% peak memory · same pixels</text>'
    )
    p.append("</svg>")
    return "".join(p)


def _font(size):
    from PIL import ImageFont

    for candidate in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_gif(points, path):
    from PIL import Image, ImageDraw

    scale = 2
    w, h = W * scale, H * scale

    def sc(v):
        return int(round(v * scale))

    f_title = _font(15 * scale)
    f_note = _font(11 * scale)
    f_axis = _font(11 * scale)
    f_val = _font(12 * scale)
    f_xl = _font(12 * scale)
    f_commit = _font(10 * scale)

    n = len(points)

    def new_frame():
        img = Image.new("RGB", (w, h), PALETTE["bg"])
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=sc(14), fill=PALETTE["bg"])
        d.text((sc(ML), sc(18)), "FOUR ROUNDS, ONE BYTE-IDENTICAL TEAPOT", font=f_title, fill=PALETTE["text"])
        d.text(
            (sc(ML), sc(44)),
            "1024×1024 · seed 42 · 6 steps · M5 Max 36 GB · output SHA-256 identical in every round",
            font=f_note,
            fill=PALETTE["muted"],
        )
        d.text((sc(ML), sc(PA_T - 22)), "MEDIAN WALL TIME", font=f_note, fill=PALETTE["time"])
        d.text((sc(ML), sc(PB_T - 22)), "PEAK PROCESS BYTES", font=f_note, fill=PALETTE["mem"])
        for t in (32, 36, 40, 44):
            y = sc(y_time(t))
            d.line([(sc(ML), y), (sc(W - MR), y)], fill=PALETTE["grid"], width=scale)
            d.text((sc(ML - 12), y), f"{t} s", font=f_axis, fill=PALETTE["faint"], anchor="rm")
        for g in (0, 1, 2, 3):
            y = sc(y_mem(g))
            d.line([(sc(ML), y), (sc(W - MR), y)], fill=PALETTE["grid"], width=scale)
            d.text((sc(ML - 12), y), f"{g} GiB", font=f_axis, fill=PALETTE["faint"], anchor="rm")
        for i, pt in enumerate(points):
            x = sc(x_at(i, n))
            d.text((x, sc(PB_B + 12)), pt["label"], font=f_xl, fill=PALETTE["text"], anchor="ma")
            d.text((x, sc(PB_B + 32)), pt["commit"], font=f_commit, fill=PALETTE["faint"], anchor="ma")
        return img, d

    def draw_series(d, upto, seg_t):
        for key, yfn, color in (("seconds", y_time, PALETTE["time"]), ("peak_gib", y_mem, PALETTE["mem"])):
            coords = [(sc(x_at(i, n)), sc(yfn(pt[key]))) for i, pt in enumerate(points)]
            for i in range(upto):
                d.line([coords[i], coords[i + 1]], fill=color, width=sc(2.5))
            if 0 <= upto < n - 1 and seg_t > 0:
                (x0, y0), (x1, y1) = coords[upto], coords[upto + 1]
                d.line([(x0, y0), (x0 + (x1 - x0) * seg_t, y0 + (y1 - y0) * seg_t)], fill=color, width=sc(2.5))
            last = min(upto + (1 if seg_t >= 1 else 0), n - 1)
            for i in range(last + 1):
                x, y = coords[i]
                d.ellipse([x - sc(4.5), y - sc(4.5), x + sc(4.5), y + sc(4.5)], fill=PALETTE["bg"], outline=color, width=sc(2.5))

    def draw_labels(d, upto):
        for i in range(upto + 1):
            pt = points[i]
            d.text((sc(x_at(i, n)), sc(y_time(pt["seconds"]) - 12)), f'{pt["seconds"]:.2f} s', font=f_val, fill=PALETTE["time"], anchor="mb")
            d.text((sc(x_at(i, n)), sc(y_mem(pt["peak_gib"]) + 10)), f'{pt["peak_gib"]:.2f} GiB', font=f_val, fill=PALETTE["mem"], anchor="ma")

    def draw_warn(d):
        d.text(
            (sc(x_at(1, n) + 16), sc(y_time(points[1]["seconds"]) - 4)),
            "round 1 regressed; published anyway",
            font=f_commit,
            fill=PALETTE["warn"],
            anchor="lm",
        )

    frames = []
    durations = []

    def push(img, ms):
        frames.append(img.resize((W, H), Image.LANCZOS))
        durations.append(ms)

    img, d = new_frame()
    draw_series(d, 0, 0)
    draw_labels(d, 0)
    push(img, 700)

    steps = 7
    for i in range(n - 1):
        for s in range(1, steps + 1):
            img, d = new_frame()
            draw_series(d, i, s / steps)
            draw_labels(d, i)
            if i >= 1:
                draw_warn(d)
            push(img, 45)
        img, d = new_frame()
        draw_series(d, i + 1, 0)
        draw_labels(d, i + 1)
        if i >= 0:
            draw_warn(d)
        push(img, 420)

    img, d = new_frame()
    draw_series(d, n - 1, 0)
    draw_labels(d, n - 1)
    draw_warn(d)
    speedup = points[0]["seconds"] / points[-1]["seconds"]
    mem_drop = 100 * (1 - points[-1]["peak_gib"] / points[0]["peak_gib"])
    d.text(
        (sc(W - MR), sc(H - 26)),
        f"42.30 s → 34.65 s · {speedup:.2f}× faster · −{mem_drop:.0f}% peak memory · same pixels",
        font=f_val,
        fill=PALETTE["text"],
        anchor="ra",
    )
    push(img, 2600)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def main():
    points = load_points()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_DIR / "optimization-progress.svg"
    gif_path = OUT_DIR / "optimization-progress.gif"
    svg_path.write_text(build_svg(points))
    build_gif(points, gif_path)
    print(f"wrote {svg_path}")
    print(f"wrote {gif_path}")
    for pt in points:
        print(f'  {pt["label"]:9s} {pt["commit"]}  {pt["seconds"]:.2f} s  {pt["peak_gib"]:.2f} GiB')


if __name__ == "__main__":
    main()
