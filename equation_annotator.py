#!/usr/bin/env python3
"""
Equation Annotator — Color-coded educational math equation annotations.

Creates Stuart Riffle / Reddit DFT-style visualizations where each part of a
LaTeX equation gets a distinct color with a descriptive label underneath, all
on a dark background.

Usage:
    # As a module
    from equation_annotator import annotate_equation, save_figure
    fig = annotate_equation(segments, title="My Equation")
    save_figure(fig, "my_equation", "output")

    # From CLI
    python equation_annotator.py --input segments.json --output output/ --show
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ============================================================================
# CONFIGURATION - Edit these for Spyder / interactive use
# ============================================================================
INPUT_SEGMENTS = None       # List of segment dicts, or None to use CLI/example
INPUT_FILE = None           # Path to JSON file with segments, or None
OUTPUT_DIR = "output"
OUTPUT_NAME = "annotated_equation"
BACKGROUND_COLOR = "#1a1a2e"
EQUATION_FONTSIZE = 36
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 18
TITLE_COLOR = "#cccccc"
DPI = 300
SHOW_CONNECTORS = True
USE_LATEX = False           # Set True if you have LaTeX installed
SHOW_PLOT = False
# ============================================================================


def save_figure(fig, name, outdir, formats=("png", "svg"), dpi=300):
    """Save a matplotlib figure in multiple formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Base filename (no extension).
    outdir : str or Path
        Output directory (created if needed).
    formats : tuple of str
        File formats to save.
    dpi : int
        Resolution for raster formats.

    Returns
    -------
    list of Path
        Paths to saved files.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = outdir / f"{name}.{fmt}"
        fig.savefig(
            path,
            format=fmt,
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.3,
        )
        paths.append(path)
        print(f"  Saved: {path}")
    return paths


def _measure_text(fig, text_str, fontsize, use_latex):
    """Measure a single text string on a figure, return (width_px, height_px)."""
    canvas = FigureCanvasAgg(fig)
    txt = fig.text(
        0.0, 0.0, text_str,
        fontsize=fontsize,
        usetex=use_latex,
    )
    canvas.draw()
    renderer = canvas.get_renderer()
    bbox = txt.get_window_extent(renderer)
    txt.remove()
    return bbox.width, bbox.height


def _measure_all(segments, equation_fontsize, label_fontsize, use_latex,
                 figsize, fig_dpi):
    """Measure all segment and label text on a properly-sized figure.

    Returns
    -------
    seg_sizes : list of (width_px, height_px)
    label_sizes : list of (width_px, height_px) or None
    """
    fig = plt.figure(figsize=figsize, dpi=fig_dpi)

    seg_sizes = []
    for seg in segments:
        is_sup = seg.get("superscript", False)
        fs = equation_fontsize * 0.65 if is_sup else equation_fontsize
        w, h = _measure_text(fig, seg["latex"], fs, use_latex)
        seg_sizes.append((w, h))

    label_sizes = []
    for seg in segments:
        label = seg.get("label")
        if label:
            w, h = _measure_text(fig, label, label_fontsize, False)
            label_sizes.append((w, h))
        else:
            label_sizes.append(None)

    plt.close(fig)
    return seg_sizes, label_sizes


def _auto_figsize(seg_sizes, label_sizes, segments, spacing_px,
                  equation_fontsize, title, fig_dpi):
    """Estimate figure dimensions from content measurements."""
    # Total equation width, accounting for superscripts having no extra spacing
    total_eq_width = 0
    for i, (seg, (w, h)) in enumerate(zip(segments, seg_sizes)):
        total_eq_width += w
        if i < len(segments) - 1:
            next_is_sup = segments[i + 1].get("superscript", False)
            if not next_is_sup:
                total_eq_width += spacing_px

    max_eq_height = max(h for _, h in seg_sizes)

    max_label_height = 0
    for ls in label_sizes:
        if ls is not None:
            max_label_height = max(max_label_height, ls[1])

    title_height = (equation_fontsize * 1.4) if title else 0
    connector_gap = equation_fontsize * 0.8
    vertical_px = (
        title_height + max_eq_height + connector_gap
        + max_label_height + equation_fontsize * 1.0
    )

    horizontal_px = total_eq_width + equation_fontsize * 3

    width_in = max(horizontal_px / fig_dpi, 6)
    height_in = max(vertical_px / fig_dpi, 2.5)

    return (width_in, height_in)


def _resolve_label_overlaps(label_centers_px, label_widths_px, min_gap_px=8):
    """Push apart overlapping label x-positions.

    Operates in-place on label_centers_px (list where entries can be None).
    """
    # Collect indices of labels that exist
    active = [(i, label_centers_px[i], label_widths_px[i])
              for i in range(len(label_centers_px))
              if label_centers_px[i] is not None]

    if len(active) <= 1:
        return

    # Sort by x position
    active.sort(key=lambda t: t[1])

    # Iteratively push overlapping labels apart
    for _ in range(10):  # max iterations
        changed = False
        for j in range(len(active) - 1):
            idx_a, cx_a, w_a = active[j]
            idx_b, cx_b, w_b = active[j + 1]
            min_dist = (w_a + w_b) / 2 + min_gap_px
            actual_dist = cx_b - cx_a
            if actual_dist < min_dist:
                push = (min_dist - actual_dist) / 2
                cx_a -= push
                cx_b += push
                active[j] = (idx_a, cx_a, w_a)
                active[j + 1] = (idx_b, cx_b, w_b)
                changed = True
        if not changed:
            break

    # Write back
    for idx, cx, _ in active:
        label_centers_px[idx] = cx


def annotate_equation(
    segments,
    *,
    title=None,
    background_color="#1a1a2e",
    equation_fontsize=36,
    label_fontsize=12,
    title_fontsize=18,
    title_color="#cccccc",
    figsize=None,
    dpi=300,
    show_connectors=True,
    use_latex=False,
    spacing_scale=1.0,
):
    """Create a color-coded annotated equation figure.

    Parameters
    ----------
    segments : list of dict
        Each dict must have 'latex' (str) and 'color' (str). Optionally:
        - 'label' (str): annotation text below (use '\\n' for line breaks)
        - 'superscript' (bool): render smaller and raised (for exponent parts)
    title : str, optional
        Title displayed above the equation.
    background_color : str
        Background color (hex or named).
    equation_fontsize : int
        Font size for equation segments.
    label_fontsize : int
        Font size for label text.
    title_fontsize : int
        Font size for the title.
    title_color : str
        Color for the title text.
    figsize : tuple of float, optional
        (width, height) in inches. Auto-computed if None.
    dpi : int
        Figure resolution.
    show_connectors : bool
        Draw lines from equation segments to their labels.
    use_latex : bool
        Use system LaTeX for rendering (requires LaTeX installation).
    spacing_scale : float
        Multiplier for spacing between equation segments.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if use_latex:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
    else:
        plt.rcParams["text.usetex"] = False
        plt.rcParams["mathtext.fontset"] = "cm"

    fig_dpi = 100
    spacing_px = equation_fontsize * 0.3 * spacing_scale
    sup_fontsize = equation_fontsize * 0.65

    # Initial measurement with a guess figsize for auto-sizing
    init_figsize = figsize or (14, 4)
    seg_sizes, label_sizes = _measure_all(
        segments, equation_fontsize, label_fontsize, use_latex,
        init_figsize, fig_dpi,
    )

    if figsize is None:
        figsize = _auto_figsize(
            seg_sizes, label_sizes, segments, spacing_px,
            equation_fontsize, title, fig_dpi,
        )

    # Re-measure with final figsize
    seg_sizes, label_sizes = _measure_all(
        segments, equation_fontsize, label_fontsize, use_latex,
        figsize, fig_dpi,
    )

    fig_width_px = figsize[0] * fig_dpi
    fig_height_px = figsize[1] * fig_dpi

    # --- Compute horizontal layout (pixel positions) ---
    total_eq_width = 0
    for i, (seg, (w, h)) in enumerate(zip(segments, seg_sizes)):
        total_eq_width += w
        if i < len(segments) - 1:
            next_is_sup = segments[i + 1].get("superscript", False)
            if not next_is_sup:
                total_eq_width += spacing_px

    x_start = (fig_width_px - total_eq_width) / 2
    x_cursor = x_start

    # Vertical positions (figure fractions)
    eq_y = 0.55 if title else 0.58
    label_y = 0.15

    seg_x_px = []       # left edge x in pixels
    seg_center_px = []   # center x in pixels (for connectors)

    for i, (seg, (w, h)) in enumerate(zip(segments, seg_sizes)):
        seg_x_px.append(x_cursor)
        seg_center_px.append(x_cursor + w / 2)
        x_cursor += w
        if i < len(segments) - 1:
            next_is_sup = segments[i + 1].get("superscript", False)
            if not next_is_sup:
                x_cursor += spacing_px

    # Label x-centers (initially centered under their segment)
    label_centers_px = []
    label_widths_px = []
    for i, (seg, ls) in enumerate(zip(segments, label_sizes)):
        if ls is not None:
            label_centers_px.append(seg_center_px[i])
            label_widths_px.append(ls[0])
        else:
            label_centers_px.append(None)
            label_widths_px.append(0)

    # Resolve overlaps
    _resolve_label_overlaps(label_centers_px, label_widths_px, min_gap_px=10)

    # --- Create the figure ---
    fig = plt.figure(figsize=figsize, dpi=fig_dpi)
    fig.set_facecolor(background_color)

    # Render equation segments
    for i, seg in enumerate(segments):
        is_sup = seg.get("superscript", False)
        fs = sup_fontsize if is_sup else equation_fontsize
        y = eq_y + 0.12 if is_sup else eq_y

        x_frac = seg_x_px[i] / fig_width_px
        fig.text(
            x_frac, y,
            seg["latex"],
            fontsize=fs,
            color=seg.get("color", "white"),
            ha="left", va="center",
            usetex=use_latex,
            transform=fig.transFigure,
        )

    # Render labels
    for i, seg in enumerate(segments):
        if label_centers_px[i] is None:
            continue
        label = seg.get("label", "")
        if not label:
            continue
        x_frac = label_centers_px[i] / fig_width_px
        fig.text(
            x_frac, label_y,
            label,
            fontsize=label_fontsize,
            color=seg.get("color", "white"),
            ha="center", va="top",
            usetex=False,
            transform=fig.transFigure,
            linespacing=1.4,
        )

    # Render connectors
    if show_connectors:
        for i, seg in enumerate(segments):
            if label_centers_px[i] is None:
                continue
            color = seg.get("color", "white")
            is_sup = seg.get("superscript", False)

            seg_cx_frac = seg_center_px[i] / fig_width_px
            lab_cx_frac = label_centers_px[i] / fig_width_px

            line_top = (eq_y - 0.08) if not is_sup else (eq_y + 0.02)
            line_bot = label_y + 0.04

            line = matplotlib.lines.Line2D(
                [seg_cx_frac, lab_cx_frac],
                [line_top, line_bot],
                transform=fig.transFigure,
                color=color,
                alpha=0.4,
                linewidth=1.0,
            )
            fig.add_artist(line)

    # Render title
    if title:
        fig.text(
            0.5, 0.90,
            title,
            fontsize=title_fontsize,
            color=title_color,
            ha="center", va="top",
            usetex=False,
            transform=fig.transFigure,
            fontweight="bold",
        )

    return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create color-coded annotated equation images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example JSON input file:
{
  "title": "My Equation",
  "segments": [
    {"latex": "$X_k$", "color": "#FF6B6B", "label": "output"},
    {"latex": "$=$", "color": "white"},
    {"latex": "$\\\\sum$", "color": "#4ECDC4", "label": "sum of"}
  ]
}
        """,
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to JSON file with segments (and optional title).",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--name", "-n", type=str, default=None,
        help=f"Output base filename (default: {OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--background", type=str, default=None,
        help=f"Background color (default: {BACKGROUND_COLOR}).",
    )
    parser.add_argument(
        "--fontsize", type=int, default=None,
        help=f"Equation font size (default: {EQUATION_FONTSIZE}).",
    )
    parser.add_argument(
        "--label-fontsize", type=int, default=None,
        help=f"Label font size (default: {LABEL_FONTSIZE}).",
    )
    parser.add_argument(
        "--dpi", type=int, default=None,
        help=f"Output DPI (default: {DPI}).",
    )
    parser.add_argument(
        "--no-connectors", action="store_true",
        help="Disable connector lines.",
    )
    parser.add_argument(
        "--use-latex", action="store_true",
        help="Use system LaTeX for rendering.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the figure interactively.",
    )
    args = parser.parse_args()

    # Resolve settings: CLI args > constants
    input_file = args.input or INPUT_FILE
    segments = INPUT_SEGMENTS
    title = None

    if input_file:
        with open(input_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            segments = data
        elif isinstance(data, dict):
            segments = data["segments"]
            title = data.get("title", title)
        else:
            raise ValueError("JSON must be a list of segments or a dict with 'segments' key.")

    if segments is None:
        parser.error(
            "No segments provided. Use --input to specify a JSON file, "
            "or set INPUT_SEGMENTS in the script."
        )

    outdir = args.output or OUTPUT_DIR
    name = args.name or OUTPUT_NAME
    bg = args.background or BACKGROUND_COLOR
    fontsize = args.fontsize or EQUATION_FONTSIZE
    label_fs = args.label_fontsize or LABEL_FONTSIZE
    dpi = args.dpi or DPI
    connectors = not args.no_connectors and SHOW_CONNECTORS
    use_latex = args.use_latex or USE_LATEX
    show = args.show or SHOW_PLOT

    print(f"Rendering {len(segments)} segments...")
    fig = annotate_equation(
        segments,
        title=title,
        background_color=bg,
        equation_fontsize=fontsize,
        label_fontsize=label_fs,
        dpi=dpi,
        show_connectors=connectors,
        use_latex=use_latex,
    )

    print("Saving output:")
    save_figure(fig, name, outdir, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
