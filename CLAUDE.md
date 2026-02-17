# Equation Annotator

Color-coded educational math equation annotations (Stuart Riffle / Reddit DFT style).

## Project Structure

```
Equation_Annotator/
├── equation_annotator.py   # Main module (matplotlib rendering logic + CLI)
├── html_renderer.py        # Dual-path HTML renderer (KaTeX/Jinja2 for HTML, matplotlib for PDF)
├── auto_annotate.py        # Auto-annotation via Claude Code
├── templates/
│   └── equation_template.html  # Jinja2 template for KaTeX HTML output
├── example_dft.py          # DFT demo script
├── requirements.txt        # matplotlib>=3.7, numpy>=1.20, jinja2>=3.1, optional sympy/weasyprint
├── CLAUDE.md
├── .gitignore
└── README.md
```

## Architecture

- **Renderer:** matplotlib (no external LaTeX required; optional `use_latex=True`)
- **Input:** Python list of segment dicts with `latex`, `color`, optional `label` and `superscript`
- **Hierarchical groups:** optional `groups` list with `segment_indices`, `label`, `color`, `level` — renders brackets spanning multiple segments
- **Description & use cases:** optional `description` string and `use_cases` list rendered below groups
- **Symbol definitions:** `symbols` list of `{symbol, name, type, description}` dicts — grouped by type (variable/parameter/constant), rendered between description and use cases. Backward-compatible with legacy `constants` format.
- **Output:** PNG (300 DPI) + SVG via `save_figure()` helper; HTML and PDF via `html_renderer.py`
- **Two-pass rendering (matplotlib):** measure text extents first, then compute layout and render
- **Dual-path HTML renderer:** `html_renderer.py` with two rendering paths:
  - **KaTeX/Jinja2 path** (default for `--html`): Uses `templates/equation_template.html` Jinja2 template. Equations rendered client-side by KaTeX (CDN). SVG connector lines, group brackets, and label-spreading handled by embedded JS. Plot still rendered server-side as matplotlib base64 PNG. Requires `jinja2>=3.1`.
  - **matplotlib/PDF path** (used for `--pdf`): Original approach — rasterized matplotlib equation card and plot as base64 PNGs, inline CSS, no JS. Compatible with weasyprint for PDF export.
  - Router: `render_html(output_format="html"|"pdf")` dispatches to the appropriate path. `render_from_spec_html(fmt="html"|"pdf"|"both")` renders one or both formats.
- **Annotated plot:** optional `plot` dict with `curves`, `x_range`, `annotations` etc. — renders a dark-themed matplotlib plot below the annotation showing the equation's behavior. Expressions evaluated via `_safe_eval_expr()` (restricted numpy namespace). Supports point, vline, hline, and region annotations.
- **Insight text:** optional `insight` string rendered below the plot — a paragraph explaining the equation's mathematical behavior and why the plot looks the way it does
- **SymPy plot verification:** optional `--verify-plot` flag runs `verify_plot()` before rendering — auto-analysis (singularity/domain checks via `_analyze_sympy_expr()`) on every curve, plus cross-validation against a canonical `sympy_form` (via `_cross_validate_curve()`). SymPy is optional; gracefully skipped when not installed.
- **Dynamic vertical layout:** `_compute_vertical_layout()` stacks layers top-down (title → equation → per-term labels → group brackets → description → symbols → use cases → plot → insight), converts to figure fractions
- **Display modes:** `display_mode` parameter controls which sections appear:
  - `full` (default) — all sections
  - `compact` — no plot or insight; keeps description, symbols, use cases
  - `plot` — no text sections (description, symbols, use cases, insight); keeps plot
  - `insight` — plot + insight + brief symbols (no description or use cases)
  - `minimal` — no plot, description, insight, or use cases; symbols show name only (no long descriptions)

## Key Design Decisions

- `superscript: True` flag on segments renders them smaller (65% fontsize) and raised
- Label overlap resolution: iterative push-apart algorithm
- Connector lines use visual center offset (40% of bbox width) to better align with glyph centers
- Group brackets: horizontal line with end ticks, italic labels centered below
- Dual interface: editable constants at top of file (Spyder-friendly) + argparse CLI
- matplotlib mathtext by default (no LaTeX install needed); `\displaystyle` not supported in mathtext mode
- Symbols section uses `ha="center"` + `multialignment="left"` to stay centered (avoids `bbox_inches='tight'` asymmetry)
- Display mode filtering applied early in `annotate_equation()` (after backward-compat conversion, before validation) — nulls out sections so the rest of the function works unchanged
- SymPy is optional: `try: import sympy` guard at module level; `verify_plot()` returns `SKIPPED` when not installed
- Plot spec extensions (`sympy_form`, per-curve `curve_parameters`) are fully backward-compatible — existing specs work unchanged
- HTML renderer is a separate file (`html_renderer.py`) — imports from `equation_annotator.py` but makes no changes to it
- KaTeX/Jinja2 template in `templates/equation_template.html` — adapted from interactive_equation_explorer's template; stripped of Plotly/mathjs/slider interactivity; adds display-mode support via `{% if show_X %}` blocks, static plot image, compact symbols mode
- weasyprint is optional (like sympy) — `--html` works without it; `--pdf` requires it
- jinja2 is a required dependency (used by the KaTeX HTML path)

## Conda Environment

```bash
conda activate equation_annotator
```

## Auto-Annotation Workflow (via Claude Code)

`auto_annotate.py` enables fully automated annotation — provide just an equation name and Claude Code generates the full spec.

### How it works
1. User asks Claude Code: "annotate Bayes' theorem with 2 hierarchy levels"
2. Claude Code reads `GENERATION_PROMPT` from `auto_annotate.py`
3. Claude Code generates a spec dict and writes it to a JSON file (e.g., `output/bayes_theorem.json`)
4. Claude Code runs `python auto_annotate.py --spec-file output/bayes_theorem.json` → PNG + SVG + JSON output
- The spec JSON is auto-saved alongside the figures for reproducibility
- No editing or resetting of `auto_annotate.py` needed
- To re-render later: `python auto_annotate.py --spec-file output/bayes_theorem.json`

### Key components in `auto_annotate.py`
- **`GENERATION_PROMPT`** — detailed instructions for producing annotation specs (colors, segmentation, labels, groups, description, use cases, plot). Tuning this prompt improves all future annotations.
- **`EXAMPLE_SPEC`** — DFT reference showing the exact format expected
- **`render_from_spec(spec)`** — convenience function: spec dict → annotated figure
- **CLI flags:** `--equation`, `--levels`, `--use-cases`, `--output`, `--name`, `--show`, `--spec-file`, `--use-example`, `--print-prompt`, `--spec-dir`, `--batch-file`, `--equations-list`, `--display-mode` / `-m`, `--html`, `--pdf`

### Batch rendering
Three modes for rendering multiple equations at once:

```bash
# Render all JSON specs in a directory
python auto_annotate.py --spec-dir output/

# Render from a JSON array of specs
python auto_annotate.py --batch-file my_batch.json

# Check which equations have specs, render existing, report missing
python auto_annotate.py --equations-list equations.json
```

**`--equations-list` format** — JSON array of strings or dicts:
```json
[
  "Bayes' Theorem",
  {"name": "Euler-Lagrange Equation", "levels": 2, "use_cases": 3},
  "Schrödinger Equation"
]
```
Existing specs (matched by snake_case filename in output dir) are rendered. Missing ones are reported so Claude Code can generate them, then re-run.

### No external API needed
Claude Code IS the LLM — it reads the prompt in-context and generates the spec directly. No API keys or SDK required.

## Current Status

- Core rendering works for both simple (Euler's identity) and complex (DFT) equations
- **Hierarchical labeling implemented** — multi-level group brackets with labels
- **Description + use cases** — rendered below groups as italic text and bulleted list
- **Auto-annotation via Claude Code** — `auto_annotate.py` with `GENERATION_PROMPT` and `render_from_spec()`
- **Batch rendering** — `--spec-dir`, `--batch-file`, `--equations-list` for multi-equation workflows
- **Symbol definitions section** — every variable, parameter, and constant with name, type, and thorough educational description; grouped by type with headers
- **Annotated plot section** — optional `plot` key in specs renders a dark-themed matplotlib plot below the annotation; supports curves, annotations (point/vline/hline/region), and parameters
- **Insight text** — optional `insight` field in specs; paragraph explaining mathematical behavior, rendered below the plot
- **SymPy plot verification** — `--verify-plot` flag on both CLIs runs singularity/domain analysis and optional cross-validation against `sympy_form`; SymPy is optional (gracefully skipped)
- **Display modes** — `--display-mode` flag on both CLIs (`full`, `compact`, `plot`, `insight`, `minimal`); also readable from spec JSON via `display_mode` key
- **Dual-path HTML renderer** — `html_renderer.py` with `--html` and `--pdf` CLI flags; `--html` uses KaTeX/Jinja2 (browser-rendered equations, SVG connectors, system fonts); `--pdf` uses matplotlib path (rasterized, weasyprint-compatible); both support all display modes and dark theme
- **Group bracket labels** — smart positioning: natural-width labels centered under bracket midpoints, edge-clamped, with iterative overlap spreading for same-level labels; `white-space: nowrap` for single-line display
- CLI supports JSON input files (including `groups`, `description`, `use_cases`, `symbols`, `plot`, `insight`, `sympy_form`, `curve_parameters` fields; legacy `constants` still accepted)
- PNG + SVG output at 300 DPI (matplotlib); HTML + PDF output (html_renderer)
- Dynamic vertical layout adapts figure height to content
- Pushed to GitHub (`main` branch)

## Pending / Next Steps

1. Fine-tune connector line alignment for equations with very tall symbols
2. Generate more example equations (Bayes' theorem, Euler-Lagrange, etc.) via auto-annotation workflow
3. Consider alternative connector styles (dotted, curved) as options
4. Add `insight` text to existing specs (Nernst, Michaelis-Menten, etc.)
5. Consider making HTML the default output format
