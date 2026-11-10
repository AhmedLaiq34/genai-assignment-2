"""Render dependency-free SVG plots from results/attention/benchmark.json."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _svg_start(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17202a}'
        '.axis{stroke:#52606d;stroke-width:1}.grid{stroke:#d9e2ec;stroke-width:1}'
        '.label{font-size:13px}.title{font-size:18px;font-weight:600}'
        '.legend{font-size:12px}</style>',
    ]


def _axes(x0, y0, x1, y1, ymin, ymax, xlabels, title, ylabel, log=False):
    out = [f'<text x="{(x0 + x1) / 2}" y="{y0 - 30}" text-anchor="middle" class="title">{title}</text>']
    for step in range(5):
        frac = step / 4
        y = y1 - frac * (y1 - y0)
        value = ymin + frac * (ymax - ymin)
        out.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" class="grid"/>')
        text = f"1e{value:.0f}" if log else f"{value:.1f}"
        out.append(f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" class="label">{text}</text>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis"/>')
    out.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis"/>')
    for index, label in enumerate(xlabels):
        x = x0 + (index + 0.5) * (x1 - x0) / len(xlabels)
        out.append(f'<text x="{x}" y="{y1 + 24}" text-anchor="middle" class="label">{label}</text>')
    out.append(f'<text x="{x0 - 55}" y="{(y0 + y1) / 2}" text-anchor="middle" '
               f'transform="rotate(-90 {x0 - 55} {(y0 + y1) / 2})" class="label">{ylabel}</text>')
    return out, y0, y1


def plot_error(rows, path: Path):
    lengths = sorted({row["length"] for row in rows})
    tiles = sorted({row["tile"] for row in rows})
    values = [math.log10(max(row["error"]["max_abs_error"], 1e-12)) for row in rows]
    ymin, ymax = math.floor(min(values)) - 1, math.ceil(max(values)) + 1
    width, height = 900, 540
    x0, x1, y0, y1 = 105, 670, 70, 450
    out = _svg_start(width, height)
    axis, y0, y1 = _axes(x0, y0, x1, y1, ymin, ymax, [str(v) for v in lengths],
                         "Tiled attention max absolute error", "log10(max abs error)", log=True)
    out += axis
    colors = ["#2563eb", "#d97706", "#059669"]
    for index, tile in enumerate(tiles):
        points = []
        for row_index, length in enumerate(lengths):
            row = next(r for r in rows if r["length"] == length and r["tile"] == tile)
            value = math.log10(max(row["error"]["max_abs_error"], 1e-12))
            x = x0 + (row_index + 0.5) * (x1 - x0) / len(lengths)
            y = y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)
            points.append(f"{x:.2f},{y:.2f}")
        out.append(f'<polyline fill="none" stroke="{colors[index % len(colors)]}" '
                   f'stroke-width="2.5" points="{" ".join(points)}"/>')
        for point in points:
            x, y = point.split(",")
            out.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{colors[index % len(colors)]}"/>')
        ly = 90 + index * 24
        out.append(f'<line x1="700" y1="{ly}" x2="735" y2="{ly}" stroke="{colors[index % len(colors)]}" '
                   f'stroke-width="2.5"/>')
        out.append(f'<text x="742" y="{ly + 4}" class="legend">tile {tile}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


def plot_runtime_memory(rows, path: Path, tile: int | None = None):
    lengths = sorted({row["length"] for row in rows})
    available_tiles = sorted({row["tile"] for row in rows})
    if not lengths or not available_tiles:
        return
    if tile is None or tile not in available_tiles:
        tile = available_tiles[-1]
    selected = [
        next(r for r in rows if r["length"] == length and r["tile"] == tile)
        for length in lengths
    ]
    memory = [
        (r["dense"]["peak_allocated_bytes"], r["tiled"]["peak_allocated_bytes"])
        for r in selected
    ]
    has_memory = all(a is not None and b is not None for a, b in memory)
    width, height = 1000, 760 if has_memory else 460
    out = _svg_start(width, height)
    panels = [
        ("Runtime (lower is better)", "median milliseconds", 60, 330,
         [("dense", [r["dense"]["median_s"] * 1000 for r in selected]),
          (f"tiled tile {tile}", [r["tiled"]["median_s"] * 1000 for r in selected])]),
    ]
    if has_memory:
        panels.append(
            ("Peak allocated GPU memory", "megabytes", 430, 700,
             [("dense", [a / 1024**2 for a, _ in memory]),
              (f"tiled tile {tile}", [b / 1024**2 for _, b in memory])])
        )
    colors = ["#dc2626", "#2563eb"]
    for title, ylabel, y0, y1, series in panels:
        maximum = max(max(values) for _, values in series)
        ymax = maximum * 1.15 if maximum > 0 else 1.0
        x0, x1 = 105, 610
        axis, y0, y1 = _axes(x0, y0, x1, y1, 0, ymax, [str(v) for v in lengths], title, ylabel)
        out += axis
        for index, (name, values) in enumerate(series):
            points = []
            for row_index, value in enumerate(values):
                x = x0 + (row_index + 0.5) * (x1 - x0) / len(lengths)
                y = y1 - value / ymax * (y1 - y0)
                points.append(f"{x:.2f},{y:.2f}")
            out.append(f'<polyline fill="none" stroke="{colors[index]}" stroke-width="2.5" '
                       f'points="{" ".join(points)}"/>')
            ly = y0 + 20 + index * 24
            out.append(f'<line x1="640" y1="{ly}" x2="675" y2="{ly}" stroke="{colors[index]}" '
                       f'stroke-width="2.5"/>')
            out.append(f'<text x="682" y="{ly + 4}" class="legend">{name}</text>')
    if not has_memory:
        out.append('<text x="105" y="420" class="label">Peak memory unavailable on this device.</text>')
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/attention/benchmark.json"))
    parser.add_argument("--error", type=Path, default=Path("results/attention/error_by_length.svg"))
    parser.add_argument("--runtime", type=Path, default=Path("results/attention/runtime_memory.svg"))
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))["rows"]
    args.error.parent.mkdir(parents=True, exist_ok=True)
    plot_error(rows, args.error)
    plot_runtime_memory(rows, args.runtime)
    print(f"wrote {args.error} and {args.runtime}")


if __name__ == "__main__":
    main()
