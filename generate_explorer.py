#!/usr/bin/env python3
"""
Interactive Equation Explorer — HTML Generator

Reads annotation JSON specs (with optional interactive block) and produces
self-contained HTML files with:
  - Color-coded KaTeX equation with labels, connectors, and group brackets
  - Slider controls for adjustable parameters
  - Live-updating Plotly plot

Usage:
    python generate_explorer.py -i examples/michaelis_menten.json --open
    python generate_explorer.py --batch-dir examples/
"""

import argparse
import json
import re
import webbrowser
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# ============================================================================
# CONFIGURATION - Edit these for Spyder / interactive use
# ============================================================================
INPUT_FILE = None               # Path to a single JSON spec file
BATCH_DIR = None                # Path to directory of JSON specs (renders all)
OUTPUT_DIR = "output"           # Where to write generated HTML files
TEMPLATE_DIR = "templates"      # Directory containing Jinja2 templates
OPEN_IN_BROWSER = False         # Auto-open generated HTML in default browser
VALIDATE_DESCRIPTIONS = True    # Use Claude API to check description/insight vs plot
# ============================================================================


def load_spec(path):
    """Load and validate a JSON annotation spec.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    dict
        The parsed spec with at least 'title' and 'segments'.

    Raises
    ------
    ValueError
        If required fields are missing.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        data = {"title": path.stem.replace("_", " ").title(), "segments": data}

    if "segments" not in data:
        raise ValueError(f"{path}: JSON must contain a 'segments' key.")

    data.setdefault("title", "Equation Explorer")
    data.setdefault("groups", [])
    data.setdefault("description", None)
    data.setdefault("use_cases", [])
    data.setdefault("constants", [])
    data.setdefault("symbols", [])
    data.setdefault("insight", None)

    return data


def _clean_latex(raw):
    """Strip surrounding $ signs from a LaTeX string for KaTeX rendering."""
    return raw.strip().lstrip("$").rstrip("$").strip()


def _prepare_segments(segments):
    """Add template-friendly fields to each segment dict."""
    prepared = []
    for seg in segments:
        s = dict(seg)
        s["latex_clean"] = seg["latex"]  # Keep original for data-attr
        s["label_html"] = (seg.get("label") or "").replace("\n", "<br>")
        prepared.append(s)
    return prepared


def _get_group_levels(groups):
    """Return sorted list of unique group levels."""
    if not groups:
        return []
    levels = sorted({g.get("level", 1) for g in groups})
    return levels


def _numpy_to_mathjs(expr):
    """Convert a numpy-style expression to math.js syntax.

    Handles: np.log → log, np.sin → sin, np.pi → pi, ** → ^, etc.
    """
    # Replace np.func(...) calls
    expr = re.sub(r'np\.', '', expr)
    # Python ** exponent → math.js ^
    expr = expr.replace('**', '^')
    return expr


def _build_annotator_plot(plot_spec, symbols):
    """Prepare an annotator-style plot block for template rendering.

    Determines which parameters get sliders (skip physical constants),
    converts numpy expressions to math.js, and builds slider configs.

    Returns dict with keys: curves, sliders, fixed_params, x_range, y_range,
    x_label, y_label, plot_title, annotations.
    """
    params = plot_spec.get("parameters", {})
    curves = plot_spec.get("curves", [])
    annotations = plot_spec.get("annotations", [])

    # Identify per-curve parameters (used to differentiate curves, not sliders)
    per_curve_names = set()
    for c in curves:
        per_curve_names.update(c.get("curve_parameters", {}).keys())

    # Build set of constant symbol names from the symbols list
    constant_names = set()
    if symbols:
        for s in symbols:
            if s.get("type") == "constant":
                constant_names.add(s["symbol"])

    # Decide which parameters get sliders vs stay fixed
    sliders = []
    fixed_params = {}
    for name, value in params.items():
        if name in per_curve_names or name in constant_names:
            fixed_params[name] = value
        else:
            # Auto-generate slider range
            if value > 0:
                lo = round(value * 0.2, 6)
                hi = round(value * 3.0, 6)
            elif value < 0:
                lo = round(value * 3.0, 6)
                hi = round(value * 0.2, 6)
            else:
                lo, hi = -10.0, 10.0
            step = round((hi - lo) / 200, 6)
            sliders.append({
                "name": name,
                "latex": name,
                "min": lo,
                "max": hi,
                "default": value,
                "step": step,
                "unit": "",
            })
            fixed_params[name] = value  # also include as default

    # Build reverse map: default_value -> slider_name so annotation positions that
    # equal a slider's default can be replaced with the parameter name string.
    # If two sliders share the same default the mapping is ambiguous — skip those.
    _val_to_slider = {}
    for s in sliders:
        v = s["default"]
        if v not in _val_to_slider:
            _val_to_slider[v] = s["name"]
        else:
            _val_to_slider[v] = None  # ambiguous: mark as un-replaceable

    def _make_dynamic(val):
        """Replace a hardcoded numeric value with a slider param name if it matches."""
        if not isinstance(val, (int, float)):
            return val
        mapped = _val_to_slider.get(val)
        # mapped is None either because val isn't a slider default, or it's ambiguous
        return mapped if mapped is not None else val

    # Post-process annotations: replace hardcoded values that match a slider's
    # default with the slider name so they update when the slider moves.
    dynamic_annotations = []
    for ann in annotations:
        ann = dict(ann)
        if "y" in ann:
            ann["y"] = _make_dynamic(ann["y"])
        if "x" in ann:
            ann["x"] = _make_dynamic(ann["x"])
        if "x_range" in ann:
            ann["x_range"] = [_make_dynamic(v) for v in ann["x_range"]]
        dynamic_annotations.append(ann)

    # Convert curve expressions
    converted_curves = []
    for c in curves:
        converted_curves.append({
            "expr": _numpy_to_mathjs(c["expr"]),
            "color": c.get("color", "#4ECDC4"),
            "label": c.get("label", ""),
            "style": c.get("style", "-"),
            "linewidth": c.get("linewidth", 2),
            "curve_parameters": c.get("curve_parameters", {}),
        })

    return {
        "curves": converted_curves,
        "sliders": sliders,
        "fixed_params": fixed_params,
        "x_range": plot_spec.get("x_range", [0, 10]),
        "y_range": plot_spec.get("y_range", None),
        "x_label": plot_spec.get("x_label", "x"),
        "y_label": plot_spec.get("y_label", "y"),
        "log_x_axis": plot_spec.get("log_x_axis", False),
        "log_y_axis": plot_spec.get("log_y_axis", False),
        "plot_title": plot_spec.get("title", ""),
        "annotations": dynamic_annotations,
        "num_points": plot_spec.get("num_points", 300),
    }


def _validate_spec_consistency(spec, plot_info):
    """Check description/insight text against plot config using pattern rules.

    Catches common mismatches (log-axis keywords vs actual axis scale,
    "appears linear" claims on non-log axes, stale parameter values in text)
    without needing an external API.

    Parameters
    ----------
    spec : dict
        The full equation spec, used to extract description and insight.
    plot_info : dict
        Normalised plot summary with keys: x_label, y_label, x_range,
        log_x_axis, log_y_axis, curve_exprs, sliders, fixed_params, annotations.

    Returns
    -------
    list[str]
        Issue descriptions (empty list if consistent).
    """
    issues = []
    description = (spec.get("description") or "")
    insight = (spec.get("insight") or "")
    text = f"{description}\n{insight}".strip()
    if not text:
        return issues
    text_lower = text.lower()

    # ── Rule 1: Log-axis claim in text vs actual axis setting ──────────
    # Match phrases that describe the AXIS as logarithmic (not the curve)
    log_axis_phrases = [
        r'on\s+a\s+logarithmic\b',            # "on a logarithmic Q axis"
        r'logarithmic\s+\w*\s*axis',           # "logarithmic x-axis"
        r'\blog\b[₁₀10]*\s*\w*\s*axis',        # "log Q axis", "log₁₀ axis"
        r'\blog\s+scale\b',                     # "log scale"
        r'plotted\s+on\s+a\s+log\b',            # "plotted on a log"
    ]
    has_log_axis_claim = any(re.search(p, text_lower) for p in log_axis_phrases)
    if has_log_axis_claim and not plot_info.get("log_x_axis"):
        issues.append(
            'Text describes a logarithmic axis but log_x_axis is not set '
            '(add "log_x_axis": true to the plot spec)'
        )

    # ── Rule 2: "appears linear" / "straight line" + log curve on linear axis
    linear_claim = re.search(
        r'appear\w*\s+linear|straight[\s-]+line|making\s+it\s+(?:appear\s+)?linear',
        text_lower,
    )
    curve_exprs = " ".join(plot_info.get("curve_exprs", []))
    has_log_in_expr = bool(re.search(r'\blog\b|\bln\b', curve_exprs))
    if linear_claim and has_log_in_expr and not plot_info.get("log_x_axis"):
        issues.append(
            "Text says the curve appears linear but the expression uses log/ln "
            "on a linear axis (it would look curved, not straight)"
        )

    # ── Rule 3: Parameter value claims vs actual defaults ─────────────
    # Match "name = value" patterns including Unicode symbols
    param_pattern = (
        r'([A-Za-z\u0394\u03b8][A-Za-z\u2080-\u2089_\u00b0°]*)'  # name
        r'\s*[=≈]\s*'                                               # = or ≈
        r'([−\-+]?\s*[0-9]*\.?[0-9]+)'                             # value
    )
    param_claims = re.findall(param_pattern, text)
    slider_params = {s["name"]: s["default"] for s in plot_info.get("sliders", [])}
    fixed_params = plot_info.get("fixed_params", {})
    all_params = {**fixed_params, **slider_params}

    def _normalise(s):
        return s.lower().replace("_", "").replace("°", "").replace("\u00b0", "")

    for name_in_text, value_str in param_claims:
        try:
            claimed = float(value_str.replace("\u2212", "-").replace(" ", ""))
        except ValueError:
            continue
        for param_name, actual in all_params.items():
            if _normalise(name_in_text) == _normalise(param_name):
                tol = max(abs(actual) * 0.01, 0.01)
                if abs(claimed - actual) > tol:
                    issues.append(
                        f"Text says {name_in_text} = {value_str.strip()} "
                        f"but actual default is {param_name} = {actual}"
                    )

    # ── Rule 4: x-range spans >2 orders of magnitude but no log axis ─
    x_range = plot_info.get("x_range", [0, 1])
    if (len(x_range) == 2
            and x_range[0] > 0
            and x_range[1] / x_range[0] > 100
            and not plot_info.get("log_x_axis")):
        issues.append(
            f"x-range spans {x_range[1]/x_range[0]:.0f}x ({x_range}) "
            f"without a log axis — consider adding \"log_x_axis\": true"
        )

    return issues


def render_html(spec, template_dir, output_path):
    """Render a spec to an HTML file using the Jinja2 template.

    Parameters
    ----------
    spec : dict
        Loaded annotation spec.
    template_dir : str or Path
        Directory containing explorer_template.html.
    output_path : str or Path
        Where to write the generated HTML.

    Returns
    -------
    Path
        The output file path.
    """
    template_dir = Path(template_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
    )
    template = env.get_template("explorer_template.html")

    has_interactive = "interactive" in spec and spec["interactive"] is not None
    # Annotator-style plot: top-level "plot" with "curves", no "interactive"
    has_annotator_plot = (
        not has_interactive
        and "plot" in spec
        and spec["plot"] is not None
        and "curves" in spec.get("plot", {})
    )
    segments = _prepare_segments(spec["segments"])
    groups = spec.get("groups", [])
    # Ensure each group has a level field
    for g in groups:
        g.setdefault("level", 1)

    group_levels = _get_group_levels(groups)

    # Extract parameter variables for slider rendering
    parameters = []
    all_variables = []
    if has_interactive:
        for v in spec["interactive"].get("variables", []):
            if v.get("role") == "parameter":
                parameters.append(v)
            all_variables.append(v)

    # Prefer `symbols` (new format) over `constants` (legacy).
    # Group symbols by type for the template.
    symbols = spec.get("symbols", [])
    constants = spec.get("constants", [])
    symbols_by_type = {}
    if symbols:
        for sym in symbols:
            sym_type = sym.get("type", "other")
            symbols_by_type.setdefault(sym_type, []).append(sym)

    insight = spec.get("insight", None)

    # Auto-generate variable descriptions from interactive block when no
    # explicit constants or symbols are provided.  This ensures every
    # interactive spec gets a variables section without duplicating info.
    auto_variables = []
    if not constants and not symbols and all_variables:
        output = spec["interactive"].get("output", {})
        # Include the output variable first
        if output:
            unit_str = ""
            # Try to infer unit from y-axis label
            y_label = spec["interactive"].get("plot", {}).get("y_axis", {}).get("label", "")
            if "(" in y_label:
                unit_str = y_label.split("(")[-1].rstrip(")")
            auto_variables.append({
                "symbol": output.get("latex", output.get("symbol", "")),
                "description": output.get("label", ""),
                "unit": unit_str,
            })
        for v in all_variables:
            unit_str = v.get("unit", "")
            desc = v.get("label", v["name"])
            if unit_str:
                desc += f" ({unit_str})"
            role_tag = "independent variable" if v.get("role") == "independent" else "parameter"
            auto_variables.append({
                "symbol": v.get("latex", v["name"]),
                "description": f"{desc} — {role_tag}",
            })

    # Process annotator-style plot if present
    ann_plot = None
    if has_annotator_plot:
        ann_plot = _build_annotator_plot(spec["plot"], symbols)

    # Validate description/insight text against the actual plot configuration
    if VALIDATE_DESCRIPTIONS and (has_annotator_plot or has_interactive):
        if has_annotator_plot and ann_plot is not None:
            plot_info = {
                "x_label": ann_plot["x_label"],
                "y_label": ann_plot["y_label"],
                "x_range": ann_plot["x_range"],
                "log_x_axis": ann_plot["log_x_axis"],
                "log_y_axis": ann_plot["log_y_axis"],
                "curve_exprs": [c["expr"] for c in ann_plot["curves"]],
                "sliders": ann_plot["sliders"],
                "fixed_params": ann_plot["fixed_params"],
                "annotations": ann_plot["annotations"],
            }
        else:
            # Interactive block — build plot_info from the interactive spec
            iv = spec["interactive"]
            indep = next((v for v in iv.get("variables", []) if v.get("role") == "independent"), {})
            plot_info = {
                "x_label": iv.get("plot", {}).get("x_axis", {}).get("label", indep.get("name", "x")),
                "y_label": iv.get("plot", {}).get("y_axis", {}).get("label", iv.get("output", {}).get("label", "y")),
                "x_range": [indep.get("min", 0), indep.get("max", 10)],
                "log_x_axis": False,
                "log_y_axis": False,
                "curve_exprs": [iv.get("expression", "")],
                "sliders": [
                    {"name": v["name"], "default": v.get("default", 0)}
                    for v in iv.get("variables", [])
                    if v.get("role") == "parameter"
                ],
                "fixed_params": {},
                "annotations": iv.get("annotations", []),
            }
        issues = _validate_spec_consistency(spec, plot_info)
        if issues:
            print(f"  WARNING — description/plot mismatch in '{spec.get('title', output_path.stem)}':")
            for issue in issues:
                print(f"    • {issue}")

    html = template.render(
        title=spec.get("title", "Equation Explorer"),
        segments=segments,
        groups=groups,
        group_levels=group_levels,
        description=spec.get("description"),
        use_cases=spec.get("use_cases", []),
        constants=constants,
        symbols=symbols,
        symbols_by_type=symbols_by_type,
        auto_variables=auto_variables,
        insight=insight,
        has_interactive=has_interactive,
        has_annotator_plot=has_annotator_plot,
        ann_plot=ann_plot,
        parameters=parameters,
        spec_json=json.dumps(spec, indent=None),
    )

    output_path.write_text(html, encoding="utf-8")
    print(f"  Generated: {output_path}")
    return output_path


def _output_name(input_path):
    """Derive an output filename from an input JSON path."""
    stem = Path(input_path).stem
    # Sanitize: keep alphanumeric + underscores
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", stem)
    return f"{safe}.html"


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML equation explorers from JSON specs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_explorer.py -i examples/michaelis_menten.json --open
  python generate_explorer.py --batch-dir examples/
  python generate_explorer.py -i spec.json -o my_output/
        """,
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to a single JSON spec file.",
    )
    parser.add_argument(
        "--batch-dir", type=str, default=None,
        help="Directory of JSON specs to render (all .json files).",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--template-dir", type=str, default=None,
        help=f"Template directory (default: {TEMPLATE_DIR}).",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open generated HTML in default browser.",
    )
    args = parser.parse_args()

    # Resolve settings: CLI args > configuration constants
    input_file = args.input or INPUT_FILE
    batch_dir = args.batch_dir or BATCH_DIR
    output_dir = Path(args.output or OUTPUT_DIR)
    template_dir = Path(args.template_dir or TEMPLATE_DIR)
    open_browser = args.open or OPEN_IN_BROWSER

    if not template_dir.is_dir():
        parser.error(f"Template directory not found: {template_dir}")

    # Collect input files
    input_files = []
    if input_file:
        input_files.append(Path(input_file))
    if batch_dir:
        batch_path = Path(batch_dir)
        if not batch_path.is_dir():
            parser.error(f"Batch directory not found: {batch_path}")
        input_files.extend(sorted(batch_path.glob("*.json")))

    if not input_files:
        parser.error(
            "No input specified. Use --input for a single file "
            "or --batch-dir for a directory of specs."
        )

    # Render each spec
    generated = []
    for input_path in input_files:
        print(f"Processing: {input_path}")
        spec = load_spec(input_path)
        out_name = _output_name(input_path)
        out_path = output_dir / out_name
        render_html(spec, template_dir, out_path)
        generated.append(out_path)

    print(f"\nDone — {len(generated)} file(s) generated in {output_dir}/")

    # Open the first (or only) file in browser
    if open_browser and generated:
        url = generated[0].resolve().as_uri()
        print(f"Opening: {url}")
        webbrowser.open(url)


if __name__ == "__main__":
    main()
