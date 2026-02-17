#!/usr/bin/env python3
"""
HTML/CSS Renderer for Equation Annotator.

Renders equation annotations as self-contained HTML documents with embedded
matplotlib images. Text layout (title, description, symbols, use cases,
insight) is handled by HTML/CSS — no more pixel-math for text flow.

The equation card and plot are rendered as matplotlib figures and embedded
as base64 data URIs.

Usage:
    from html_renderer import render_from_spec_html

    # HTML output (no extra dependencies)
    render_from_spec_html(spec, "output", "my_equation", fmt="html")

    # PDF output (requires weasyprint>=60)
    render_from_spec_html(spec, "output", "my_equation", fmt="pdf")
"""

import base64
import html as html_mod
import re
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from equation_annotator import (
    annotate_equation,
    _group_symbols_by_type,
    _safe_eval_expr,
    _validate_plot,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
BACKGROUND_COLOR = "#1a1a2e"
EQ_DPI = 200
PLOT_DPI = 150


# ============================================================================
# Matplotlib → base64
# ============================================================================

def _fig_to_base64(fig, dpi=200):
    """Save a matplotlib figure to a base64-encoded PNG data URI."""
    buf = BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi,
        facecolor=fig.get_facecolor(), edgecolor="none",
        bbox_inches="tight", pad_inches=0.3,
    )
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    return f"data:image/png;base64,{encoded}"


# ============================================================================
# Render equation card (matplotlib) — equation + labels + connectors + brackets
# ============================================================================

def render_equation_card(spec):
    """Render just the equation annotation (no text sections, no plot).

    Returns a matplotlib Figure containing the equation with color-coded
    segments, labels, connectors, and group brackets.
    """
    fig = annotate_equation(
        spec["segments"],
        title=None,
        groups=spec.get("groups", []),
        description=None,
        use_cases=[],
        symbols=[],
        plot=None,
        insight=None,
        display_mode="full",
        equation_fontsize=38,
        label_fontsize=11,
        show_connectors=True,
    )
    return fig


# ============================================================================
# Render plot as a standalone figure
# ============================================================================

def render_plot_standalone(plot_spec, background_color=BACKGROUND_COLOR,
                           description_fontsize=11):
    """Render the annotated plot as a standalone matplotlib figure.

    Reuses _safe_eval_expr() for curve evaluation. Returns a Figure.
    """
    _validate_plot(plot_spec)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    fig.set_facecolor(background_color)

    # Dark theme styling
    ax_face = "#252545"
    ax.set_facecolor(ax_face)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.tick_params(colors="#999999", labelsize=description_fontsize * 0.85)
    ax.grid(True, color="#333355", linewidth=0.5, alpha=0.6)

    # Evaluate and plot curves
    x_range = plot_spec["x_range"]
    parameters = plot_spec.get("parameters", {})
    num_points = plot_spec.get("num_points", 500)
    x = np.linspace(x_range[0], x_range[1], num_points)

    for curve in plot_spec["curves"]:
        all_params = dict(parameters)
        all_params.update(curve.get("curve_parameters", {}))
        y = _safe_eval_expr(curve["expr"], x, all_params)
        label = curve.get("label")
        style = curve.get("style", "-")
        lw = curve.get("linewidth", 2)
        ax.plot(x, y, color=curve["color"], label=label, linestyle=style,
                linewidth=lw, alpha=curve.get("alpha", 0.9))

    # Y range
    y_range = plot_spec.get("y_range")
    if y_range:
        ax.set_ylim(y_range)

    # Annotations
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
                    label, xy=(ann["x"], ann["y"]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=description_fontsize * 0.8, color=color, alpha=alpha,
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
                      fontsize=description_fontsize * 0.9)
    if y_label:
        ax.set_ylabel(y_label, color="#BBBBBB",
                      fontsize=description_fontsize * 0.9)

    # Plot subtitle
    plot_title = plot_spec.get("title")
    if plot_title:
        ax.set_title(plot_title, color="#CCCCCC",
                     fontsize=description_fontsize, pad=8)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        leg = ax.legend(
            fontsize=description_fontsize * 0.8,
            facecolor=ax_face, edgecolor="#555555",
            labelcolor="#BBBBBB", loc="best",
        )
        leg.get_frame().set_alpha(0.8)

    fig.tight_layout()
    return fig


# ============================================================================
# HTML section builders
# ============================================================================

def _esc(text):
    """Escape HTML special characters."""
    return html_mod.escape(text, quote=True)


def _build_title_html(title):
    """Build the title section."""
    return f'<h1 class="title">{_esc(title)}</h1>'


def _build_description_html(description):
    """Build the description section."""
    lines = description.split("\n")
    html_lines = "<br>".join(_esc(line) for line in lines)
    return f'<p class="description">{html_lines}</p>'


def _build_symbols_html(symbols, compact=False):
    """Build the symbols section, grouped by type.

    Parameters
    ----------
    symbols : list of dict
    compact : bool
        If True, render in compact mode (name only, no descriptions).
    """
    grouped, active_types, show_headers = _group_symbols_by_type(symbols)
    type_labels = {
        "variable": "Variables",
        "parameter": "Parameters",
        "constant": "Constants",
    }

    parts = []
    parts.append('<div class="symbols">')

    if compact:
        # Compact: "Type: sym (name), sym (name), ..." per type
        for t in active_types:
            items = []
            for s in grouped[t]:
                sym = _esc(s.get("symbol", ""))
                name = _esc(s.get("name", ""))
                items.append(f'<span class="sym-compact">'
                             f'<strong>{sym}</strong> ({name})</span>')
            label = type_labels[t]
            parts.append(
                f'<p class="sym-line"><span class="sym-type-inline">'
                f'{label}:</span> {" &middot; ".join(items)}</p>'
            )
    else:
        # Full: grouped with descriptions
        for t in active_types:
            if show_headers:
                parts.append(f'<h3 class="sym-type-header">{type_labels[t]}</h3>')
            parts.append('<dl class="sym-list">')
            for s in grouped[t]:
                sym = _esc(s.get("symbol", ""))
                name = _esc(s.get("name", ""))
                desc = _esc(s.get("description", ""))
                term = f"<strong>{sym}</strong>"
                if name:
                    term += f" ({name})"
                parts.append(f"  <dt>{term}</dt>")
                if desc:
                    parts.append(f"  <dd>{desc}</dd>")
            parts.append("</dl>")

    parts.append("</div>")
    return "\n".join(parts)


def _build_use_cases_html(use_cases):
    """Build the use cases section."""
    parts = []
    parts.append('<div class="use-cases">')
    parts.append('<h3 class="uc-header">Use Cases</h3>')
    parts.append("<ul>")
    for uc in use_cases:
        parts.append(f"  <li>{_esc(uc)}</li>")
    parts.append("</ul>")
    parts.append("</div>")
    return "\n".join(parts)


def _build_info_section_html(symbols, use_cases, compact=False):
    """Build the combined symbols + use cases section.

    Uses CSS Grid for 2-column layout when both are present.
    """
    has_symbols = bool(symbols)
    has_uc = bool(use_cases)

    if not has_symbols and not has_uc:
        return ""

    sym_html = _build_symbols_html(symbols, compact=compact) if has_symbols else ""
    uc_html = _build_use_cases_html(use_cases) if has_uc else ""

    if has_symbols and has_uc:
        return (
            '<div class="info-grid two-col">\n'
            f"  {sym_html}\n"
            f"  {uc_html}\n"
            "</div>"
        )
    else:
        return (
            '<div class="info-grid one-col">\n'
            f"  {sym_html}{uc_html}\n"
            "</div>"
        )


def _build_insight_html(insight):
    """Build the insight section."""
    return f'<p class="insight">{_esc(insight)}</p>'


# ============================================================================
# CSS template
# ============================================================================

_CSS = """\
@page {
    size: letter;
    margin: 0.75in;
    background-color: #1a1a2e;
}
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}
body {
    background-color: #1a1a2e;
    color: #cccccc;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    padding: 2em;
    max-width: 900px;
    margin: 0 auto;
}
.title {
    color: #cccccc;
    font-size: 20pt;
    font-weight: bold;
    text-align: center;
    margin-bottom: 0.5em;
    letter-spacing: 0.5px;
}
.eq-card {
    text-align: center;
    margin: 1em 0 1.5em 0;
}
.eq-card img {
    max-width: 100%;
    height: auto;
}
.description {
    color: #bbbbbb;
    font-style: italic;
    text-align: center;
    margin: 0.5em 2em 1.2em 2em;
    line-height: 1.6;
}
.info-grid {
    margin: 1em 0;
    border-top: 1px solid #333355;
    padding-top: 1em;
}
.info-grid.two-col {
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: 1.5em;
}
.info-grid.one-col {
    display: block;
}
/* Symbols */
.symbols {
    color: #aaaaaa;
}
.sym-type-header {
    color: #888888;
    font-size: 9pt;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.8em;
    margin-bottom: 0.3em;
    border-bottom: 1px solid #2a2a4e;
    padding-bottom: 0.2em;
}
.sym-type-header:first-child {
    margin-top: 0;
}
.sym-list {
    margin: 0 0 0.5em 0;
}
.sym-list dt {
    color: #bbbbbb;
    margin-top: 0.4em;
}
.sym-list dd {
    color: #999999;
    margin-left: 1.5em;
    font-size: 10pt;
    line-height: 1.5;
}
/* Compact symbols */
.sym-line {
    margin: 0.3em 0;
    color: #aaaaaa;
    line-height: 1.6;
}
.sym-type-inline {
    color: #888888;
    font-weight: bold;
    font-size: 9pt;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.sym-compact strong {
    color: #bbbbbb;
}
/* Use cases */
.use-cases {
    color: #999999;
}
.uc-header {
    color: #888888;
    font-size: 9pt;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4em;
    border-bottom: 1px solid #2a2a4e;
    padding-bottom: 0.2em;
}
.use-cases ul {
    list-style: disc;
    margin-left: 1.2em;
}
.use-cases li {
    margin-bottom: 0.3em;
    line-height: 1.5;
}
/* Plot */
.plot-section {
    text-align: center;
    margin: 1.5em 0;
    border-top: 1px solid #333355;
    padding-top: 1em;
}
.plot-section img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
}
/* Insight */
.insight {
    color: #bbbbbb;
    margin: 1em 1em;
    line-height: 1.7;
    text-align: justify;
    border-left: 3px solid #333355;
    padding-left: 1em;
}
"""


# ============================================================================
# Main HTML assembly
# ============================================================================

def render_html(spec, display_mode="full", eq_dpi=EQ_DPI, plot_dpi=PLOT_DPI):
    """Render a full annotation spec as a self-contained HTML string.

    Parameters
    ----------
    spec : dict
        Annotation spec (same format as JSON spec files).
    display_mode : str
        Display mode: full, compact, plot, minimal, insight.
    eq_dpi : int
        DPI for the equation card image.
    plot_dpi : int
        DPI for the plot image.

    Returns
    -------
    str
        Complete HTML document.
    """
    # Resolve display mode from spec if not overridden
    mode = display_mode if display_mode != "full" else spec.get("display_mode", "full")

    valid_modes = {"full", "compact", "plot", "minimal", "insight"}
    if mode not in valid_modes:
        raise ValueError(f"display_mode must be one of {valid_modes}, got {mode!r}")

    # Extract spec fields
    title = spec.get("title")
    description = spec.get("description")
    symbols = spec.get("symbols", [])
    if not symbols and spec.get("constants"):
        symbols = [
            {"symbol": c.get("symbol", ""), "name": c.get("symbol", ""),
             "type": "constant", "description": c.get("description", "")}
            for c in spec["constants"]
        ]
    use_cases = spec.get("use_cases", [])
    plot_spec = spec.get("plot")
    insight = spec.get("insight")

    # Apply display mode filtering
    compact_symbols = False
    if mode == "compact":
        plot_spec = None
        insight = None
    elif mode == "plot":
        description = None
        symbols = []
        use_cases = []
        insight = None
    elif mode == "minimal":
        plot_spec = None
        insight = None
        description = None
        use_cases = []
        compact_symbols = True
    elif mode == "insight":
        use_cases = []
        compact_symbols = True

    # --- Render matplotlib images ---
    # Equation card
    eq_fig = render_equation_card(spec)
    eq_data_uri = _fig_to_base64(eq_fig, dpi=eq_dpi)
    plt.close(eq_fig)

    # Plot (if present after mode filtering)
    plot_data_uri = None
    if plot_spec:
        plot_fig = render_plot_standalone(plot_spec)
        plot_data_uri = _fig_to_base64(plot_fig, dpi=plot_dpi)
        plt.close(plot_fig)

    # --- Build HTML sections ---
    sections = []

    if title:
        sections.append(_build_title_html(title))

    # Equation card image
    sections.append(
        f'<div class="eq-card">'
        f'<img src="{eq_data_uri}" alt="Annotated equation">'
        f'</div>'
    )

    if description:
        sections.append(_build_description_html(description))

    # Info section (symbols + use cases)
    info_html = _build_info_section_html(symbols, use_cases, compact=compact_symbols)
    if info_html:
        sections.append(info_html)

    # Plot image
    if plot_data_uri:
        sections.append(
            f'<div class="plot-section">'
            f'<img src="{plot_data_uri}" alt="Equation plot">'
            f'</div>'
        )

    if insight:
        sections.append(_build_insight_html(insight))

    # --- Assemble HTML document ---
    body = "\n\n".join(sections)
    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        f"  <title>{_esc(title or 'Equation Annotation')}</title>\n"
        f"  <style>\n{_CSS}  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>"
    )
    return html


# ============================================================================
# PDF export (optional weasyprint)
# ============================================================================

def save_pdf(html_str, output_path):
    """Export an HTML string to PDF via weasyprint.

    Parameters
    ----------
    html_str : str
        Complete HTML document.
    output_path : str or Path
        Path for the output PDF.

    Raises
    ------
    ImportError
        If weasyprint is not installed.
    """
    try:
        from weasyprint import HTML
    except ImportError:
        raise ImportError(
            "weasyprint is required for PDF output. "
            "Install with: pip install weasyprint>=60"
        )
    HTML(string=html_str).write_pdf(str(output_path))
    print(f"  Saved: {output_path}")


# ============================================================================
# Convenience function (parallel to auto_annotate.render_from_spec)
# ============================================================================

def render_from_spec_html(spec, output_dir="output", output_name=None,
                          fmt="pdf", display_mode="full",
                          eq_dpi=EQ_DPI, plot_dpi=PLOT_DPI):
    """Render an annotation spec as HTML and/or PDF.

    Parameters
    ----------
    spec : dict
        Annotation spec (same format as JSON spec files).
    output_dir : str or Path
        Output directory (created if needed).
    output_name : str, optional
        Base filename (no extension). Auto-generated from title if None.
    fmt : str
        Output format: "pdf", "html", or "both".
    display_mode : str
        Display mode: full, compact, plot, minimal, insight.
    eq_dpi : int
        DPI for the equation card image.
    plot_dpi : int
        DPI for the plot image.

    Returns
    -------
    list of Path
        Paths to saved files.
    """
    title = spec.get("title", "Equation")
    if output_name is None:
        output_name = re.sub(r"[^\w\s]", "", title.lower())
        output_name = re.sub(r"\s+", "_", output_name).strip("_")

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering (HTML): {title} (mode: {display_mode})")

    html_str = render_html(spec, display_mode=display_mode,
                           eq_dpi=eq_dpi, plot_dpi=plot_dpi)

    paths = []

    if fmt in ("html", "both"):
        html_path = outdir / f"{output_name}.html"
        html_path.write_text(html_str, encoding="utf-8")
        print(f"  Saved: {html_path}")
        paths.append(html_path)

    if fmt in ("pdf", "both"):
        pdf_path = outdir / f"{output_name}.pdf"
        save_pdf(html_str, pdf_path)
        paths.append(pdf_path)

    return paths
