#!/usr/bin/env python3
"""
Auto-Annotation Helper for Equation Annotator.

Provides a GENERATION_PROMPT that Claude Code follows to produce annotation
specs, plus render_from_spec() and render_batch() convenience functions.

Workflow (in Claude Code):
    1. User asks to annotate an equation (e.g., "annotate Bayes' theorem")
    2. Claude Code reads GENERATION_PROMPT from this file
    3. Claude Code generates a spec dict following the prompt
    4. Claude Code writes it to a JSON file (e.g., output/bayes_theorem.json)
    5. Claude Code runs: python auto_annotate.py --spec-file output/bayes_theorem.json

Single-equation usage:
    # Spyder / interactive: edit EQUATION_SPEC below, then run
    python auto_annotate.py

    # CLI pointing to a JSON spec file:
    python auto_annotate.py --spec-file spec.json --output output/

Batch usage:
    # Render all JSON specs in a directory:
    python auto_annotate.py --spec-dir output/

    # Render from a JSON array of specs:
    python auto_annotate.py --batch-file my_batch.json

    # Check which equations have specs, render existing, report missing:
    python auto_annotate.py --equations-list equations.json
"""

import argparse
import json
import re
from pathlib import Path

from equation_annotator import annotate_equation, save_figure

# ============================================================================
# CONFIGURATION - Edit these for Spyder / interactive use
# ============================================================================
EQUATION_INPUT = "Discrete Fourier Transform"
NUM_HIERARCHY_LEVELS = 2
NUM_USE_CASES = 3
OUTPUT_DIR = "output"
OUTPUT_NAME = None          # Auto-generated from equation title if None
SHOW_PLOT = False
EQUATIONS_LIST = None       # List of equation names for batch generation, or None

# Set this to a spec dict (see EXAMPLE_SPEC for format) or None.
# Claude Code fills this in when auto-annotating.
EQUATION_SPEC = None
# ============================================================================


# ============================================================================
# GENERATION_PROMPT — Claude Code reads this and follows it to produce a spec
# ============================================================================
GENERATION_PROMPT = """\
You are generating an annotation spec for the Equation Annotator tool.

Given:
  - Equation: {equation_input}
  - Hierarchy levels: {num_levels}
  - Number of use cases: {num_use_cases}

Produce a Python dict called `spec` with exactly these keys:

  "title": str
      Formal equation name (e.g., "Bayes' Theorem").

  "segments": list of dict
      Each segment represents one visual piece of the equation. Keys:
        - "latex": str — LaTeX for this piece, wrapped in $...$
        - "color": str — hex color (see palette below)
        - "label": str or None — concise 2-4 word educational label;
          use "\\n" for line breaks (max 2 lines). None for operators/equals.
        - "superscript": bool — True ONLY for actual exponent/power parts
          that should render smaller and raised. Default False.

  "groups": list of dict
      Hierarchical bracket groups spanning multiple segments. Keys:
        - "segment_indices": list of int — contiguous range of segment indices
        - "label": str — what this group represents conceptually
        - "color": str — hex color, should match a key segment in the group
        - "level": int — 1, 2, etc. (level 1 is closest to the equation)
      Rules:
        - No overlapping groups within the same level
        - Each group should span a meaningful conceptual unit
        - Higher levels span wider ranges that encompass lower-level groups

  "description": str
      1-2 sentence plain-English explanation accessible to an educated
      non-specialist. May use "\\n" for line breaks.

  "use_cases": list of str
      Each formatted as "Domain: description" (e.g., "Audio processing:
      Identifying dominant frequencies in a sound recording"). Draw from
      diverse domains: science, engineering, finance, biology, CS, etc.

  "constants": list of dict (optional)
      Mathematical constants appearing in the equation. Each dict has:
        - "symbol": str — the constant's symbol in plain text (e.g., "e", "π", "i")
        - "description": str — brief plain-English explanation
          (e.g., "Euler's number (≈ 2.718), the base of natural logarithms")
      Only include genuine mathematical/physical constants, not variables.
      Omit this key if the equation has no notable constants.

Color palette guidelines (for dark #1a1a2e background):
  - Use 8-10 visually distinct, saturated colors. Good choices:
    "#FF6B6B" (coral red)     "#4ECDC4" (teal)
    "#FFE66D" (gold)          "#A8E6CF" (mint green)
    "#FF8C42" (orange)        "#C792EA" (lavender)
    "#89CFF0" (sky blue)      "#F9A8D4" (pink)
    "#B8E986" (lime)          "#FFA07A" (salmon)
  - Use "#AAAAAA" (gray) for operators, equals signs, and other
    structural symbols that don't need labels
  - Reuse colors for mathematically related terms (e.g., same variable
    appearing in multiple places)

Segmentation rules:
  - Split at natural mathematical boundaries (each variable, operator,
    fraction, summation, etc. is its own segment)
  - Operators (=, +, -, ·) and structural punctuation are unlabeled
    gray segments
  - Keep segments atomic: one concept per segment
  - Superscripts (exponents) are separate segments with superscript: True

Label guidelines:
  - Concise, educational, plain-English (no jargon)
  - 2-4 words per line, max 2 lines with "\\n"
  - Describe WHAT the term represents, not its symbol name
  - Example: "normalize by\\nnumber of samples" not "one over N"

EXAMPLE (DFT) — use this as a format reference:

{example_spec}

Now generate the spec for: {equation_input}
with {num_levels} hierarchy level(s) and {num_use_cases} use case(s).

Return ONLY the Python dict literal (no markdown fences, no variable
assignment). It must be valid Python that can be parsed with ast.literal_eval().
"""
# ============================================================================


# ============================================================================
# EXAMPLE_SPEC — DFT reference (matches example_dft.py)
# ============================================================================
EXAMPLE_SPEC = {
    "title": "The Discrete Fourier Transform",
    "segments": [
        {
            "latex": r"$X_k$",
            "color": "#FF6B6B",
            "label": "frequency\nbin k",
        },
        {
            "latex": r"$ = $",
            "color": "#AAAAAA",
            "label": None,
        },
        {
            "latex": r"$\frac{1}{N}$",
            "color": "#4ECDC4",
            "label": "normalize by\nnumber of samples",
        },
        {
            "latex": r"$\sum_{n=0}^{N-1}$",
            "color": "#FFE66D",
            "label": "iterate over\nall samples",
        },
        {
            "latex": r"$x_n$",
            "color": "#A8E6CF",
            "label": "each sample\nin the signal",
        },
        {
            "latex": r"$\cdot$",
            "color": "#AAAAAA",
            "label": None,
        },
        {
            "latex": r"$e$",
            "color": "#FF8C42",
            "label": "complex\nexponential",
        },
        {
            "latex": r"$-i$",
            "color": "#C792EA",
            "label": "rotate\nbackwards",
            "superscript": True,
        },
        {
            "latex": r"$\,2\pi$",
            "color": "#89CFF0",
            "label": "a full\ncircle",
            "superscript": True,
        },
        {
            "latex": r"$\,k$",
            "color": "#FF6B6B",
            "label": "for each\nfrequency",
            "superscript": True,
        },
        {
            "latex": r"$\,n/N$",
            "color": "#FFE66D",
            "label": "fraction of\ntime elapsed",
            "superscript": True,
        },
    ],
    "groups": [
        {
            "segment_indices": [6, 7, 8, 9, 10],
            "label": "correlation with sinusoid\nat frequency k",
            "color": "#FF8C42",
            "level": 1,
        },
        {
            "segment_indices": [3, 4, 5, 6, 7, 8, 9, 10],
            "label": "for each sample, compare to a sinusoid and accumulate",
            "color": "#FFE66D",
            "level": 2,
        },
    ],
    "description": (
        "The DFT decomposes a discrete signal of N samples into N frequency "
        "components.\nEach output bin X_k measures how much the signal "
        "correlates with a sinusoid at frequency k."
    ),
    "constants": [
        {"symbol": "e", "description": "Euler's number (\u2248 2.718), the base of natural logarithms"},
        {"symbol": "\u03c0", "description": "Pi (\u2248 3.14159), ratio of a circle's circumference to its diameter"},
        {"symbol": "i", "description": "Imaginary unit, where i\u00b2 = \u22121"},
    ],
    "use_cases": [
        "Audio processing: Identifying the dominant frequencies in a sound recording",
        "Structural biology: Processing electron density maps in X-ray crystallography",
        "Telecommunications: Modulating and demodulating signals in OFDM systems",
    ],
}
# ============================================================================


def get_generation_prompt(equation_input, num_levels=2, num_use_cases=3):
    """Return the filled-in generation prompt for Claude Code to follow.

    Parameters
    ----------
    equation_input : str
        Equation name or LaTeX string.
    num_levels : int
        Number of hierarchy levels for group brackets.
    num_use_cases : int
        Number of use cases to generate.

    Returns
    -------
    str
        The complete prompt text.
    """
    return GENERATION_PROMPT.format(
        equation_input=equation_input,
        num_levels=num_levels,
        num_use_cases=num_use_cases,
        example_spec=json.dumps(EXAMPLE_SPEC, indent=2),
    )


def render_from_spec(spec, output_dir="output", output_name=None, show=False,
                     equation_fontsize=38, label_fontsize=11):
    """Render an annotated equation from a spec dict.

    Parameters
    ----------
    spec : dict
        Annotation spec with keys: title, segments, groups, description,
        use_cases.
    output_dir : str
        Directory for output files.
    output_name : str, optional
        Base filename. Auto-generated from title if None.
    show : bool
        Display the figure interactively.
    equation_fontsize : int
        Font size for equation segments.
    label_fontsize : int
        Font size for labels.

    Returns
    -------
    list of Path
        Paths to saved files.
    """
    import matplotlib.pyplot as plt

    title = spec.get("title", "Equation")
    segments = spec["segments"]
    groups = spec.get("groups", [])
    description = spec.get("description")
    use_cases = spec.get("use_cases", [])
    constants = spec.get("constants", [])

    if output_name is None:
        # Convert title to snake_case filename
        output_name = re.sub(r"[^\w\s]", "", title.lower())
        output_name = re.sub(r"\s+", "_", output_name).strip("_")

    print(f"Rendering: {title}")
    print(f"  {len(segments)} segments, {len(groups)} groups")

    fig = annotate_equation(
        segments,
        title=title,
        equation_fontsize=equation_fontsize,
        label_fontsize=label_fontsize,
        show_connectors=True,
        groups=groups,
        description=description,
        use_cases=use_cases,
        constants=constants,
    )

    print("Saving output:")
    paths = save_figure(fig, output_name, output_dir)

    # Save spec JSON alongside the figures for reproducibility
    spec_path = Path(output_dir) / f"{output_name}.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    paths.append(spec_path)
    print(f"  Saved: {spec_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return paths


def render_batch(specs, output_dir="output", show=False):
    """Render multiple annotated equations from a list of spec dicts.

    Parameters
    ----------
    specs : list of dict
        Each element is a spec dict with keys: title, segments, groups, etc.
    output_dir : str
        Directory for output files.
    show : bool
        Display figures interactively.

    Returns
    -------
    tuple of (int, int)
        (success_count, failure_count)
    """
    successes = 0
    failures = 0
    failed_names = []

    for i, spec in enumerate(specs, 1):
        title = spec.get("title", f"Equation {i}")
        try:
            render_from_spec(spec, output_dir=output_dir, show=show)
            successes += 1
        except Exception as e:
            print(f"  FAILED: {title} — {e}")
            failures += 1
            failed_names.append(title)

    print()
    print(f"Batch complete: {successes} rendered, {failures} failed")
    if failed_names:
        for name in failed_names:
            print(f"  - {name}")

    return successes, failures


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-annotate equations via Claude Code.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Single equation:
  python auto_annotate.py --spec-file output/bayes_theorem.json
  python auto_annotate.py --use-example

Batch rendering:
  python auto_annotate.py --spec-dir output/
  python auto_annotate.py --batch-file my_batch.json
  python auto_annotate.py --equations-list equations.json

The --equations-list file is a JSON array of equation names (strings)
or dicts with {"name": ..., "levels": N, "use_cases": N}. Existing
specs are rendered; missing ones are reported for generation.
        """,
    )
    parser.add_argument(
        "--equation", "-e", type=str, default=None,
        help=f"Equation name (for prompt generation). Default: {EQUATION_INPUT}",
    )
    parser.add_argument(
        "--levels", "-l", type=int, default=None,
        help=f"Number of hierarchy levels. Default: {NUM_HIERARCHY_LEVELS}",
    )
    parser.add_argument(
        "--use-cases", "-u", type=int, default=None,
        help=f"Number of use cases. Default: {NUM_USE_CASES}",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help=f"Output directory. Default: {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--name", "-n", type=str, default=None,
        help="Output base filename. Default: auto from title.",
    )
    parser.add_argument(
        "--spec-file", "-s", type=str, default=None,
        help="Path to JSON file containing the equation spec.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the figure interactively.",
    )
    parser.add_argument(
        "--print-prompt", action="store_true",
        help="Print the generation prompt and exit (for debugging).",
    )
    parser.add_argument(
        "--use-example", action="store_true",
        help="Use EXAMPLE_SPEC (DFT) instead of EQUATION_SPEC.",
    )
    # Batch mode flags
    parser.add_argument(
        "--spec-dir", type=str, default=None,
        help="Directory of JSON spec files. Renders all *.json specs found.",
    )
    parser.add_argument(
        "--batch-file", type=str, default=None,
        help="JSON file containing a list of spec dicts [{spec1}, {spec2}, ...].",
    )
    parser.add_argument(
        "--equations-list", type=str, default=None,
        help="JSON file with equation names (strings or {name, levels, use_cases} dicts). "
             "Renders specs that exist, reports missing ones for generation.",
    )
    args = parser.parse_args()

    # Resolve settings
    equation_input = args.equation or EQUATION_INPUT
    num_levels = args.levels if args.levels is not None else NUM_HIERARCHY_LEVELS
    num_use_cases = args.use_cases if args.use_cases is not None else NUM_USE_CASES
    output_dir = args.output or OUTPUT_DIR
    output_name = args.name or OUTPUT_NAME
    show = args.show or SHOW_PLOT

    # Print prompt mode
    if args.print_prompt:
        print(get_generation_prompt(equation_input, num_levels, num_use_cases))
        return

    # --- Batch modes (checked first) ---

    # 1. --spec-dir: render all *.json in a directory
    if args.spec_dir:
        spec_dir = Path(args.spec_dir)
        if not spec_dir.is_dir():
            print(f"Error: {spec_dir} is not a directory.")
            return
        json_files = sorted(spec_dir.glob("*.json"))
        if not json_files:
            print(f"No *.json files found in {spec_dir}")
            return
        specs = []
        for jf in json_files:
            with open(jf) as f:
                specs.append(json.load(f))
        print(f"Found {len(specs)} spec files in {spec_dir}")
        render_batch(specs, output_dir=output_dir, show=show)
        return

    # 2. --batch-file: render list of specs from a single JSON file
    if args.batch_file:
        with open(args.batch_file) as f:
            specs = json.load(f)
        if not isinstance(specs, list):
            print("Error: --batch-file must contain a JSON array of spec dicts.")
            return
        print(f"Loaded {len(specs)} specs from {args.batch_file}")
        render_batch(specs, output_dir=output_dir, show=show)
        return

    # 3. --equations-list: check which equations have specs, report missing
    if args.equations_list:
        with open(args.equations_list) as f:
            eq_list = json.load(f)
        if not isinstance(eq_list, list):
            print("Error: --equations-list must contain a JSON array.")
            return

        out_path = Path(output_dir)
        have_specs = []
        missing = []

        for entry in eq_list:
            if isinstance(entry, str):
                name = entry
            elif isinstance(entry, dict):
                name = entry.get("name", "")
            else:
                continue

            # Convert name to expected filename (snake_case)
            fname = re.sub(r"[^\w\s]", "", name.lower())
            fname = re.sub(r"\s+", "_", fname).strip("_")
            spec_file = out_path / f"{fname}.json"

            if spec_file.exists():
                have_specs.append(spec_file)
            else:
                missing.append((name, spec_file))

        # Render existing specs
        if have_specs:
            print(f"Found {len(have_specs)} existing spec(s), rendering...")
            specs = []
            for sf in have_specs:
                with open(sf) as f:
                    specs.append(json.load(f))
            render_batch(specs, output_dir=output_dir, show=show)

        # Report missing
        if missing:
            print()
            print(f"{len(missing)} equation(s) need spec generation:")
            for name, expected in missing:
                print(f"  - {name}  (expected: {expected})")
            print()
            print("Ask Claude Code to generate specs for these, then re-run.")

        if not have_specs and not missing:
            print("No equations in the list.")
        return

    # --- Single-spec modes (existing behavior) ---
    spec = None

    if args.use_example:
        spec = EXAMPLE_SPEC
    elif args.spec_file:
        with open(args.spec_file) as f:
            spec = json.load(f)
    elif EQUATION_SPEC is not None:
        spec = EQUATION_SPEC

    if spec is None:
        print("No equation spec provided.")
        print()
        print("Options:")
        print("  1. Set EQUATION_SPEC in this file")
        print("  2. Use --spec-file to point to a JSON spec")
        print("  3. Use --use-example to render the DFT example")
        print("  4. Use --print-prompt to see the generation prompt")
        print()
        print("Batch options:")
        print("  5. Use --spec-dir to render all specs in a directory")
        print("  6. Use --batch-file to render specs from a JSON array")
        print("  7. Use --equations-list to check/render from an equation list")
        print()
        print("Or ask Claude Code to auto-annotate an equation for you.")
        return

    render_from_spec(spec, output_dir, output_name, show)
    print("Done!")


if __name__ == "__main__":
    main()
