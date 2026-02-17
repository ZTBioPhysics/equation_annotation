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
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
GROUP_FONTSIZE = 11
DESCRIPTION_FONTSIZE = 11
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
                  equation_fontsize, title, fig_dpi, layout=None):
    """Estimate figure dimensions from content measurements."""
    # Total equation width, accounting for superscripts having no extra spacing
    total_eq_width = 0
    for i, (seg, (w, h)) in enumerate(zip(segments, seg_sizes)):
        total_eq_width += w
        if i < len(segments) - 1:
            next_is_sup = segments[i + 1].get("superscript", False)
            if not next_is_sup:
                total_eq_width += spacing_px

    horizontal_px = total_eq_width + equation_fontsize * 3

    # Use dynamic layout for vertical sizing if available
    if layout is not None:
        vertical_px = layout["total_height_px"]
    else:
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


def _validate_groups(groups, num_segments):
    """Validate group definitions against segment list.

    Raises ValueError on invalid groups.
    """
    for gi, g in enumerate(groups):
        indices = g.get("segment_indices", [])
        if not indices:
            raise ValueError(f"Group {gi}: 'segment_indices' is empty.")
        for idx in indices:
            if idx < 0 or idx >= num_segments:
                raise ValueError(
                    f"Group {gi}: segment index {idx} out of range "
                    f"(0..{num_segments - 1})."
                )
        level = g.get("level", 1)
        if level < 1:
            raise ValueError(f"Group {gi}: level must be >= 1, got {level}.")

    # Check for overlaps within the same level
    by_level = {}
    for gi, g in enumerate(groups):
        level = g.get("level", 1)
        by_level.setdefault(level, []).append((gi, set(g["segment_indices"])))
    for level, entries in by_level.items():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                overlap = entries[i][1] & entries[j][1]
                if overlap:
                    raise ValueError(
                        f"Groups {entries[i][0]} and {entries[j][0]} overlap "
                        f"on level {level} at indices {sorted(overlap)}."
                    )


def _safe_eval_expr(expr_str, x_array, parameters=None):
    """Evaluate a numpy expression string safely.

    Parameters
    ----------
    expr_str : str
        Expression using ``x`` and numpy functions (e.g., ``"np.sin(2 * np.pi * x)"``).
    x_array : numpy.ndarray
        Values to substitute for ``x``.
    parameters : dict, optional
        Additional name/value pairs available in the expression.

    Returns
    -------
    numpy.ndarray
        Result broadcast to match ``x_array`` shape.
    """
    namespace = {"np": np, "x": x_array, "__builtins__": {}}
    if parameters:
        namespace.update(parameters)
    try:
        result = eval(expr_str, namespace)  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Failed to evaluate expression {expr_str!r}: {exc}") from exc
    # Broadcast scalar results
    result = np.asarray(result)
    if result.shape == ():
        result = np.broadcast_to(result, x_array.shape).copy()
    return result


def _validate_plot(plot_spec):
    """Validate a plot spec dict.

    Raises ValueError on invalid specs.
    """
    if not isinstance(plot_spec, dict):
        raise ValueError("plot must be a dict.")
    curves = plot_spec.get("curves")
    if not curves:
        raise ValueError("plot.curves must be a non-empty list.")
    for ci, c in enumerate(curves):
        if "expr" not in c:
            raise ValueError(f"plot.curves[{ci}]: missing 'expr'.")
        if "color" not in c:
            raise ValueError(f"plot.curves[{ci}]: missing 'color'.")
    x_range = plot_spec.get("x_range")
    if x_range is None:
        raise ValueError("plot.x_range is required.")
    if (not isinstance(x_range, (list, tuple)) or len(x_range) != 2
            or x_range[0] >= x_range[1]):
        raise ValueError("plot.x_range must be [min, max] with min < max.")
    annotations = plot_spec.get("annotations", [])
    valid_types = {"point", "vline", "hline", "region"}
    for ai, a in enumerate(annotations):
        atype = a.get("type")
        if atype not in valid_types:
            raise ValueError(
                f"plot.annotations[{ai}]: type must be one of {valid_types}, got {atype!r}."
            )


def _compute_vertical_layout(
    equation_fontsize, title, has_labels, groups, description, use_cases,
    group_fontsize, description_fontsize, fig_dpi, symbols=None, plot=None,
):
    """Compute y-positions (figure-fraction) for all vertical layers.

    Works top-down in pixel space, then converts to figure fractions.

    Returns
    -------
    layout : dict
        Keys: 'title_y', 'eq_y', 'label_y', 'group_levels' (dict of
        level -> bracket_y, label_y), 'desc_y', 'use_cases_y',
        'total_height_px'.
    """
    y_cursor = 0  # pixels from top

    # Title
    if title:
        title_top = y_cursor + equation_fontsize * 0.4
        title_height = equation_fontsize * 2.0
        y_cursor = title_top + title_height
    else:
        title_top = None
        y_cursor += equation_fontsize * 0.4

    # Equation — reserve space for tall symbols (summation with limits)
    eq_gap = equation_fontsize * 0.8
    y_cursor += eq_gap
    eq_center = y_cursor + equation_fontsize * 1.0
    y_cursor = eq_center + equation_fontsize * 1.5

    # Mark the bottom of the equation zone (below all symbols incl. limits)
    eq_bottom = y_cursor

    # Connector + per-term labels — generous gap so labels sit well below
    if has_labels:
        connector_gap = equation_fontsize * 2.2
        y_cursor += connector_gap
        label_top = y_cursor
        label_height = equation_fontsize * 1.4  # approximate multi-line label
        y_cursor = label_top + label_height
    else:
        label_top = None

    # Group levels
    max_level = 0
    if groups:
        max_level = max(g.get("level", 1) for g in groups)
    group_level_positions = {}
    for lv in range(1, max_level + 1):
        gap = equation_fontsize * 0.4
        y_cursor += gap
        bracket_y = y_cursor
        y_cursor += equation_fontsize * 0.15  # bracket height (tick)
        group_label_y = y_cursor + equation_fontsize * 0.1
        group_label_height = group_fontsize * 2.5  # allow 2 lines
        y_cursor = group_label_y + group_label_height
        group_level_positions[lv] = {
            "bracket_y": bracket_y,
            "label_y": group_label_y,
        }

    # Description
    if description:
        gap = equation_fontsize * 1.2
        y_cursor += gap
        desc_y = y_cursor
        # Count lines in description for height
        desc_lines = description.count("\n") + 1
        desc_height = description_fontsize * 1.6 * desc_lines
        y_cursor = desc_y + desc_height
    else:
        desc_y = None

    # 2-column layout decision (before height calc, affects wrapping estimate)
    info_columns = bool(symbols and use_cases)

    # Symbols (variable/parameter/constant definitions) — compute height
    symbols_height = 0
    if symbols:
        type_order = ["variable", "parameter", "constant"]
        grouped = {t: [] for t in type_order}
        for s in symbols:
            t = s.get("type", "constant")
            if t not in grouped:
                t = "constant"
            grouped[t].append(s)
        active_types = [t for t in type_order if grouped[t]]
        show_headers = len(active_types) > 1

        if info_columns:
            # Estimate wrapped line count per entry (column is ~56% of figure)
            # Use ~100 chars as approximate wrap width for height estimation
            est_wrap = 100
            num_lines = 0
            for t in active_types:
                for s in grouped[t]:
                    name = s.get("name", "")
                    desc = s.get("description", "")
                    sym = s.get("symbol", "")
                    raw = f"  {sym} ({name}) \u2014 {desc}" if name else f"  {sym} \u2014 {desc}"
                    num_lines += max(1, -(-len(raw) // est_wrap))  # ceil division
            if show_headers:
                num_lines += len(active_types)
                num_lines += max(0, len(active_types) - 1)
        else:
            num_lines = sum(len(grouped[t]) for t in active_types)
            if show_headers:
                num_lines += len(active_types)
                num_lines += max(0, len(active_types) - 1)

        symbols_height = description_fontsize * 2.0 * (num_lines + 0.5)

    # Use cases — compute height
    uc_height = 0
    if use_cases:
        # +1.5 lines for "Use Cases" header when in 2-column mode
        uc_header_lines = 1.5 if info_columns else 0
        if info_columns:
            est_wrap = 50
            uc_lines = sum(max(1, -(-len(uc) // est_wrap)) for uc in use_cases)
        else:
            uc_lines = len(use_cases)
        uc_height = description_fontsize * 2.2 * (uc_lines + uc_header_lines)
    symbols_y = None
    uc_y = None
    info_y_px = None

    if info_columns:
        gap = equation_fontsize * 0.6
        y_cursor += gap
        info_y_px = y_cursor
        shared_height = max(symbols_height, uc_height)
        y_cursor += shared_height
    else:
        if symbols:
            gap = equation_fontsize * 0.3
            y_cursor += gap
            symbols_y = y_cursor
            y_cursor += symbols_height
        if use_cases:
            gap = equation_fontsize * 0.3
            y_cursor += gap
            uc_y = y_cursor
            y_cursor += uc_height

    # Plot section
    if plot:
        y_cursor += equation_fontsize * 2.5  # gap above plot
        plot_top = y_cursor
        plot_height_px = plot.get("height_px", 250)
        y_cursor += plot_height_px
    else:
        plot_top = None

    # Bottom padding
    y_cursor += equation_fontsize * 0.6

    total_height_px = y_cursor

    # Convert to figure fractions (top-down -> matplotlib bottom-up)
    def to_frac(px):
        if px is None:
            return None
        return 1.0 - (px / total_height_px)

    layout = {
        "title_y": to_frac(title_top + equation_fontsize * 0.6) if title else None,
        "eq_y": to_frac(eq_center),
        "connector_top_y": to_frac(eq_bottom),
        "label_y": to_frac(label_top) if has_labels else None,
        "group_levels": {
            lv: {
                "bracket_y": to_frac(pos["bracket_y"]),
                "label_y": to_frac(pos["label_y"]),
            }
            for lv, pos in group_level_positions.items()
        },
        "desc_y": to_frac(desc_y) if description else None,
        "info_columns": info_columns,
        "info_y": to_frac(info_y_px) if info_y_px is not None else None,
        "symbols_y": to_frac(symbols_y) if symbols_y is not None else None,
        "use_cases_y": to_frac(uc_y) if uc_y is not None else None,
        "plot_top_y": to_frac(plot_top) if plot else None,
        "plot_bottom_y": to_frac(plot_top + plot.get("height_px", 250)) if plot else None,
        "total_height_px": total_height_px,
    }
    return layout


def _render_bracket(fig, x_left_frac, x_right_frac, y_frac, color,
                    tick_height=0.015, linewidth=1.5):
    """Draw a bracket (horizontal line with end ticks) in figure coordinates."""
    # Horizontal line
    h_line = matplotlib.lines.Line2D(
        [x_left_frac, x_right_frac],
        [y_frac, y_frac],
        transform=fig.transFigure,
        color=color, alpha=0.7, linewidth=linewidth,
    )
    fig.add_artist(h_line)
    # Left tick
    l_tick = matplotlib.lines.Line2D(
        [x_left_frac, x_left_frac],
        [y_frac + tick_height, y_frac],
        transform=fig.transFigure,
        color=color, alpha=0.7, linewidth=linewidth,
    )
    fig.add_artist(l_tick)
    # Right tick
    r_tick = matplotlib.lines.Line2D(
        [x_right_frac, x_right_frac],
        [y_frac + tick_height, y_frac],
        transform=fig.transFigure,
        color=color, alpha=0.7, linewidth=linewidth,
    )
    fig.add_artist(r_tick)


def _render_groups(fig, groups, seg_x_px, seg_sizes, fig_width_px, layout,
                   group_fontsize):
    """Render group brackets and labels."""
    for g in groups:
        indices = g["segment_indices"]
        level = g.get("level", 1)
        color = g.get("color", "#AAAAAA")
        label = g.get("label", "")

        level_pos = layout["group_levels"].get(level)
        if level_pos is None:
            continue

        # Bracket spans from left edge of first segment to right edge of last
        first_idx = min(indices)
        last_idx = max(indices)
        x_left_px = seg_x_px[first_idx]
        x_right_px = seg_x_px[last_idx] + seg_sizes[last_idx][0]

        x_left_frac = x_left_px / fig_width_px
        x_right_frac = x_right_px / fig_width_px
        bracket_y = level_pos["bracket_y"]

        _render_bracket(fig, x_left_frac, x_right_frac, bracket_y, color)

        # Label centered below bracket
        if label:
            x_center_frac = (x_left_frac + x_right_frac) / 2
            label_y = level_pos["label_y"]
            fig.text(
                x_center_frac, label_y,
                label,
                fontsize=group_fontsize,
                color=color,
                ha="center", va="top",
                usetex=False,
                transform=fig.transFigure,
                linespacing=1.3,
                fontstyle="italic",
            )


def _render_description(fig, description, layout, description_fontsize,
                        fig_width_px):
    """Render the plain-English description text block."""
    desc_y = layout["desc_y"]
    if desc_y is None:
        return
    fig.text(
        0.5, desc_y,
        description,
        fontsize=description_fontsize,
        color="#BBBBBB",
        ha="center", va="top",
        usetex=False,
        transform=fig.transFigure,
        linespacing=1.4,
        fontstyle="italic",
        wrap=True,
    )


def _render_symbols(fig, symbols, layout, description_fontsize,
                    x_pos=0.5, ha="center", wrap_width=0):
    """Render the symbol definitions section grouped by type.

    Parameters
    ----------
    x_pos : float
        Horizontal position in figure fraction (default 0.5 for centered).
    ha : str
        Horizontal alignment ('center' or 'left').
    wrap_width : int
        If > 0, wrap description lines to this many characters.
    """
    symbols_y = layout.get("symbols_y")
    if symbols_y is None:
        symbols_y = layout.get("info_y")
    if symbols_y is None:
        return

    # Group entries by type
    type_order = ["variable", "parameter", "constant"]
    type_labels = {
        "variable": "Variables",
        "parameter": "Parameters",
        "constant": "Constants",
    }
    grouped = {t: [] for t in type_order}
    for s in symbols:
        t = s.get("type", "constant")
        if t not in grouped:
            t = "constant"
        grouped[t].append(s)

    active_types = [t for t in type_order if grouped[t]]
    show_headers = len(active_types) > 1

    # Build a single multi-line string
    lines = []
    for ti, t in enumerate(active_types):
        if ti > 0:
            lines.append("")  # blank separator
        if show_headers:
            lines.append(type_labels[t])
        for s in grouped[t]:
            name = s.get("name", "")
            desc = s.get("description", "")
            sym = s.get("symbol", "")
            if name:
                raw = f"  {sym} ({name}) \u2014 {desc}"
            else:
                raw = f"  {sym} \u2014 {desc}"
            if wrap_width > 0:
                wrapped = textwrap.fill(
                    raw, width=wrap_width,
                    subsequent_indent="        ",
                )
                lines.append(wrapped)
            else:
                lines.append(raw)

    full_text = "\n".join(lines)
    fs = description_fontsize * 1.1

    fig.text(
        x_pos, symbols_y,
        full_text,
        fontsize=fs,
        color="#AAAAAA",
        ha=ha, va="top",
        multialignment="left",
        usetex=False,
        transform=fig.transFigure,
        linespacing=1.5,
    )


def _render_use_cases(fig, use_cases, layout, description_fontsize,
                      x_pos=0.5, ha="center", show_header=False,
                      wrap_width=0):
    """Render bulleted use-case list.

    Parameters
    ----------
    x_pos : float
        Horizontal position in figure fraction (default 0.5 for centered).
    ha : str
        Horizontal alignment ('center' or 'left').
    show_header : bool
        If True, render a "Use Cases" header above the bullets.
    wrap_width : int
        If > 0, wrap bullet lines to this many characters.
    """
    uc_y = layout.get("use_cases_y")
    if uc_y is None:
        uc_y = layout.get("info_y")
    if uc_y is None:
        return
    lines = []
    if show_header:
        lines.append("Use Cases")
        lines.append("")  # blank separator
    for uc in use_cases:
        raw = f"\u2022  {uc}"
        if wrap_width > 0:
            wrapped = textwrap.fill(
                raw, width=wrap_width,
                subsequent_indent="    ",
            )
            lines.append(wrapped)
        else:
            lines.append(raw)
    bullet_text = "\n".join(lines)
    fig.text(
        x_pos, uc_y,
        bullet_text,
        fontsize=description_fontsize * 1.1,
        color="#999999",
        ha=ha, va="top",
        usetex=False,
        transform=fig.transFigure,
        linespacing=1.6,
    )


def _render_plot(fig, plot_spec, layout, background_color, description_fontsize):
    """Render an annotated matplotlib plot below the equation annotation.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    plot_spec : dict
        Plot specification with curves, x_range, annotations, etc.
    layout : dict
        Vertical layout dict from ``_compute_vertical_layout``.
    background_color : str
        Figure background color (used for axes styling).
    description_fontsize : int
        Base font size for labels.
    """
    plot_top_y = layout.get("plot_top_y")
    plot_bottom_y = layout.get("plot_bottom_y")
    if plot_top_y is None or plot_bottom_y is None:
        return

    # Axes rect: [left, bottom, width, height] in figure fraction
    left = 0.12
    right = 0.88
    ax_width = right - left
    ax_bottom = plot_bottom_y
    ax_height = plot_top_y - plot_bottom_y
    ax = fig.add_axes([left, ax_bottom, ax_width, ax_height])

    # Dark theme styling
    ax_face = "#252545"
    ax.set_facecolor(ax_face)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.tick_params(colors="#999999", labelsize=description_fontsize * 0.75)
    ax.grid(True, color="#333355", linewidth=0.5, alpha=0.6)

    # Evaluate and plot curves
    x_range = plot_spec["x_range"]
    parameters = plot_spec.get("parameters", {})
    num_points = plot_spec.get("num_points", 500)
    x = np.linspace(x_range[0], x_range[1], num_points)

    for curve in plot_spec["curves"]:
        y = _safe_eval_expr(curve["expr"], x, parameters)
        label = curve.get("label")
        style = curve.get("style", "-")
        lw = curve.get("linewidth", 2)
        ax.plot(x, y, color=curve["color"], label=label, linestyle=style,
                linewidth=lw, alpha=curve.get("alpha", 0.9))

    # Y range
    y_range = plot_spec.get("y_range")
    if y_range:
        ax.set_ylim(y_range)

    # Render annotations
    for ann in plot_spec.get("annotations", []):
        atype = ann["type"]
        color = ann.get("color", "#AAAAAA")
        alpha = ann.get("alpha", 0.7)
        style = ann.get("style", "solid")
        label = ann.get("label")

        if atype == "point":
            ax.plot(ann["x"], ann["y"], "o", color=color, markersize=7,
                    zorder=5, alpha=alpha)
            if label:
                ax.annotate(
                    label,
                    xy=(ann["x"], ann["y"]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=description_fontsize * 0.7,
                    color=color, alpha=alpha,
                )
        elif atype == "vline":
            ax.axvline(ann["x"], color=color, linestyle=style, alpha=alpha,
                       linewidth=1.2, label=label)
        elif atype == "hline":
            ax.axhline(ann["y"], color=color, linestyle=style, alpha=alpha,
                       linewidth=1.2, label=label)
        elif atype == "region":
            region_range = ann.get("x_range", [x_range[0], x_range[1]])
            region_alpha = ann.get("alpha", 0.15)
            ax.axvspan(region_range[0], region_range[1], color=color,
                       alpha=region_alpha, label=label)

    # Axis labels
    x_label = plot_spec.get("x_label")
    y_label = plot_spec.get("y_label")
    if x_label:
        ax.set_xlabel(x_label, color="#BBBBBB",
                      fontsize=description_fontsize * 0.8)
    if y_label:
        ax.set_ylabel(y_label, color="#BBBBBB",
                      fontsize=description_fontsize * 0.8)

    # Plot subtitle
    plot_title = plot_spec.get("title")
    if plot_title:
        ax.set_title(plot_title, color="#CCCCCC",
                     fontsize=description_fontsize * 0.9, pad=8)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        leg = ax.legend(
            fontsize=description_fontsize * 0.7,
            facecolor=ax_face, edgecolor="#555555",
            labelcolor="#BBBBBB", loc="best",
        )
        leg.get_frame().set_alpha(0.8)


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
    groups=None,
    description=None,
    use_cases=None,
    symbols=None,
    constants=None,
    group_fontsize=None,
    description_fontsize=None,
    plot=None,
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
    groups : list of dict, optional
        Hierarchical groupings. Each dict has 'segment_indices' (list of int),
        'label' (str), 'color' (str), and 'level' (int, default 1).
    description : str, optional
        Plain-English description rendered below groups.
    use_cases : list of str, optional
        Practical use-case examples rendered as a bulleted list.
    symbols : list of dict, optional
        Symbol definitions with 'symbol', 'name', 'type', and 'description'.
        Type is one of 'variable', 'parameter', or 'constant'. Rendered
        between description and use cases, grouped by type.
    constants : list of dict, optional
        Deprecated — use ``symbols`` instead. Legacy format with 'symbol'
        and 'description' keys. Converted to symbols format automatically.
    group_fontsize : int, optional
        Font size for group labels (default: GROUP_FONTSIZE).
    description_fontsize : int, optional
        Font size for description and use cases (default: DESCRIPTION_FONTSIZE).
    plot : dict, optional
        Plot specification for an annotated matplotlib plot rendered below
        the equation annotation. Keys: curves, x_range, y_range,
        parameters, annotations, x_label, y_label, title, height_px.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if group_fontsize is None:
        group_fontsize = GROUP_FONTSIZE
    if description_fontsize is None:
        description_fontsize = DESCRIPTION_FONTSIZE
    if groups is None:
        groups = []
    if use_cases is None:
        use_cases = []
    # Backward compat: convert legacy constants to symbols format
    if symbols is None and constants is not None:
        symbols = [
            {
                "symbol": c.get("symbol", ""),
                "name": c.get("symbol", ""),
                "type": "constant",
                "description": c.get("description", ""),
            }
            for c in constants
        ]
    if symbols is None:
        symbols = []

    # Validate groups
    if groups:
        _validate_groups(groups, len(segments))

    # Validate plot
    if plot:
        _validate_plot(plot)

    if use_latex:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
    else:
        plt.rcParams["text.usetex"] = False
        plt.rcParams["mathtext.fontset"] = "cm"

    fig_dpi = 100
    spacing_px = equation_fontsize * 0.3 * spacing_scale
    sup_fontsize = equation_fontsize * 0.65

    # Check if any segments have labels
    has_labels = any(seg.get("label") for seg in segments)

    # Compute dynamic vertical layout
    layout = _compute_vertical_layout(
        equation_fontsize, title, has_labels, groups, description, use_cases,
        group_fontsize, description_fontsize, fig_dpi, symbols=symbols,
        plot=plot,
    )

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
            layout=layout,
        )

    # Re-measure with final figsize
    seg_sizes, label_sizes = _measure_all(
        segments, equation_fontsize, label_fontsize, use_latex,
        figsize, fig_dpi,
    )

    fig_width_px = figsize[0] * fig_dpi

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

    # Get y-positions from layout
    eq_y = layout["eq_y"]
    label_y = layout["label_y"]

    seg_x_px = []       # left edge x in pixels
    seg_center_px = []   # center x in pixels (for connectors)
    # Visual center of math glyphs sits slightly left of bounding-box
    # center (subscripts, limits, etc. extend the bbox rightward).
    VISUAL_CENTER = 0.4

    for i, (seg, (w, h)) in enumerate(zip(segments, seg_sizes)):
        seg_x_px.append(x_cursor)
        seg_center_px.append(x_cursor + w * VISUAL_CENTER)
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
    # Superscript raise is relative to equation position in figure fraction
    sup_raise = layout["total_height_px"] * 0.06 / layout["total_height_px"]
    # Use a fraction of figure height for superscript offset
    sup_offset = 0.06 * (figsize[1] / max(figsize[1], 2.5))
    for i, seg in enumerate(segments):
        is_sup = seg.get("superscript", False)
        fs = sup_fontsize if is_sup else equation_fontsize
        y = eq_y + sup_offset if is_sup else eq_y

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
    if has_labels and label_y is not None:
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

    # Render connectors (straight lines from below equation to labels)
    if show_connectors and has_labels and label_y is not None:
        connector_top_y = layout["connector_top_y"]
        fig_height_px = figsize[1] * fig_dpi
        gap = connector_top_y - label_y

        for i, seg in enumerate(segments):
            if label_centers_px[i] is None:
                continue
            color = seg.get("color", "white")
            is_sup = seg.get("superscript", False)

            seg_cx_frac = seg_center_px[i] / fig_width_px
            lab_cx_frac = label_centers_px[i] / fig_width_px
            line_bot = label_y + 0.01

            if is_sup:
                seg_h_frac = seg_sizes[i][1] / fig_height_px
                line_top = (eq_y + sup_offset) - (seg_h_frac / 2) - 0.015
            else:
                line_top = connector_top_y - gap * 0.20

            line_top = max(line_top, line_bot + 0.02)

            line = matplotlib.lines.Line2D(
                [seg_cx_frac, lab_cx_frac],
                [line_top, line_bot],
                transform=fig.transFigure,
                color=color, alpha=0.4, linewidth=1.0,
            )
            fig.add_artist(line)

    # Render title
    if title and layout["title_y"] is not None:
        fig.text(
            0.5, layout["title_y"],
            title,
            fontsize=title_fontsize,
            color=title_color,
            ha="center", va="top",
            usetex=False,
            transform=fig.transFigure,
            fontweight="bold",
        )

    # Render group brackets
    if groups:
        _render_groups(
            fig, groups, seg_x_px, seg_sizes, fig_width_px, layout,
            group_fontsize,
        )

    # Render description
    if description:
        _render_description(fig, description, layout, description_fontsize,
                            fig_width_px)

    # Render symbol definitions and use cases
    if layout.get("info_columns"):
        # 2-column layout: symbols 60% left, use cases 35% right
        # Column geometry — span full figure width
        sym_left = 0.03
        sym_right = 0.61
        divider_x = 0.63
        uc_left = 0.66
        uc_right = 0.97

        # Estimate wrap widths from figure size and font
        # Convert fontsize (points) to average character width (pixels)
        pts_to_px = fig_dpi / 72.0
        avg_char_factor = 0.50  # average width/height for proportional font

        sym_col_px = figsize[0] * fig_dpi * (sym_right - sym_left)
        sym_char_px = (description_fontsize * 1.1) * pts_to_px * avg_char_factor
        sym_wrap = max(40, int(sym_col_px / sym_char_px))
        uc_col_px = figsize[0] * fig_dpi * (uc_right - uc_left)
        uc_char_px = (description_fontsize * 1.1) * pts_to_px * avg_char_factor
        uc_wrap = max(30, int(uc_col_px / uc_char_px))

        if symbols:
            _render_symbols(fig, symbols, layout, description_fontsize,
                            x_pos=sym_left, ha="left", wrap_width=sym_wrap)
        if use_cases:
            _render_use_cases(fig, use_cases, layout, description_fontsize,
                              x_pos=uc_left, ha="left", show_header=True,
                              wrap_width=uc_wrap)
        # Subtle vertical divider between columns
        info_y = layout.get("info_y")
        if info_y is not None:
            plot_top = layout.get("plot_top_y")
            divider_bottom = plot_top + 0.01 if plot_top is not None else 0.02
            divider = matplotlib.lines.Line2D(
                [divider_x, divider_x],
                [info_y - 0.01, divider_bottom],
                transform=fig.transFigure,
                color="#333355", alpha=0.5, linewidth=1.0,
            )
            fig.add_artist(divider)
    else:
        # Single-column centered layout
        if symbols:
            _render_symbols(fig, symbols, layout, description_fontsize)
        if use_cases:
            _render_use_cases(fig, use_cases, layout, description_fontsize)

    # Render plot
    if plot:
        _render_plot(fig, plot, layout, background_color, description_fontsize)

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
        "--group-fontsize", type=int, default=None,
        help=f"Group label font size (default: {GROUP_FONTSIZE}).",
    )
    parser.add_argument(
        "--desc-fontsize", type=int, default=None,
        help=f"Description font size (default: {DESCRIPTION_FONTSIZE}).",
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
    groups = None
    description = None
    use_cases = None
    symbols = None
    constants = None
    plot = None

    if input_file:
        with open(input_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            segments = data
        elif isinstance(data, dict):
            segments = data["segments"]
            title = data.get("title", title)
            groups = data.get("groups", None)
            description = data.get("description", None)
            use_cases = data.get("use_cases", None)
            symbols = data.get("symbols", None)
            constants = data.get("constants", None)
            plot = data.get("plot", None)
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
    group_fs = args.group_fontsize or GROUP_FONTSIZE
    desc_fs = args.desc_fontsize or DESCRIPTION_FONTSIZE

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
        groups=groups,
        description=description,
        use_cases=use_cases,
        symbols=symbols,
        constants=constants,
        group_fontsize=group_fs,
        description_fontsize=desc_fs,
        plot=plot,
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
