# Equation Annotator (with Interactive Explorer)

Color-coded educational math equation annotations + interactive HTML explorers.
Two tools, one JSON spec format: `auto_annotate.py` generates static PNG/SVG/HTML; `generate_explorer.py` generates interactive HTML with Plotly sliders.

## Project Structure

```
Equation_Annotator/
  equation_annotator.py     # Core matplotlib rendering (PNG/SVG)
  html_renderer.py          # Static HTML/PDF rendering (KaTeX + Jinja2 / weasyprint)
  auto_annotate.py          # CLI tool: generates specs, renders via --html, --pdf, --explorer
  generate_explorer.py      # Interactive HTML generator (Plotly sliders, live plots)
  example_dft.py            # DFT demo script
  templates/
    equation_template.html  # Jinja2 template for static KaTeX HTML output
    explorer_template.html  # Jinja2 template for interactive explorer output
  examples/                 # Explorer-native JSON specs (with interactive or plot blocks)
    michaelis_menten.json
    logistic_growth.json
    euler_identity.json
  output/                   # Generated files (gitignored)
  requirements.txt          # matplotlib, numpy, jinja2; optional sympy, weasyprint
  README.md
  CLAUDE.md
  .gitignore
```

## Usage

### Static annotation (PNG + SVG)
```bash
python auto_annotate.py --spec-file output/the_hill_equation.json
python auto_annotate.py --spec-dir output/
```

### Static HTML / PDF
```bash
python auto_annotate.py --spec-file output/the_hill_equation.json --html
python auto_annotate.py --spec-file output/the_hill_equation.json --pdf
python auto_annotate.py --spec-dir output/ --html
```

### Interactive explorer
```bash
python generate_explorer.py -i examples/michaelis_menten.json --open
python generate_explorer.py --batch-dir examples/
python generate_explorer.py -i output/the_hill_equation.json --open   # annotator specs work directly
python auto_annotate.py --spec-file output/the_hill_equation.json --explorer
python auto_annotate.py --spec-dir output/ --explorer
# Combine outputs (all three formats):
python auto_annotate.py --spec-file output/the_hill_equation.json --html --explorer
```

### Teaching workflow
```bash
python generate_explorer.py --batch-dir /path/to/BCH4300/course-design/equation-specs/
```

## Current Status

### Static annotator (`equation_annotator.py` + `html_renderer.py`)
- Core rendering: matplotlib (PNG + SVG), KaTeX/Jinja2 HTML, weasyprint PDF
- `auto_annotate.py` with `GENERATION_PROMPT` + `EXAMPLE_SPEC` for Claude Code annotation
- Batch modes: `--spec-dir`, `--batch-file`, `--equations-list`
- Display modes: `full`, `compact`, `plot`, `insight`, `minimal`
- Symbol definitions grouped by type (variable/parameter/constant)
- Annotated plots with dark-theme matplotlib, curve annotations
- Insight text section
- SymPy plot verification (`--verify-plot`)
- Group bracket labels with overlap spreading
- All output in `output/` (gitignored)

### Interactive explorer (`generate_explorer.py`)
- Generates self-contained HTML: KaTeX equation + Plotly sliders + live plot
- Two input modes:
  - **`interactive` block** (explorer-native): single math.js expression with named variables/sliders
  - **`plot` block** (annotator-native): multi-curve numpy expressions, auto-converted to math.js, auto-generated sliders
- Template: `templates/explorer_template.html` (Jinja2)
- Spec validation: `_validate_spec_consistency()` checks description/insight vs plot config
- Log axis support: `log_x_axis`, `log_y_axis` in plot spec
- Dynamic plot annotations that update as sliders move
- `render_explorer_from_spec(spec, output_dir, output_name)` — API for auto_annotate.py
- `--explorer` flag in `auto_annotate.py` — not mutually exclusive with `--html`/`--pdf`
- Template path uses `Path(__file__).resolve().parent` — works from any working directory

## Key Dependencies
- Python: matplotlib, numpy, jinja2
- CDN (explorer): KaTeX, Plotly.js, math.js
- Optional: sympy (plot verification), weasyprint (PDF export)
- Conda env: `equation_annotator`

## JSON Spec Format

Both tools share the same spec format. Key fields:
- `title`, `segments` (required)
- `groups`, `description`, `use_cases`, `symbols`, `insight` (optional)
- `plot` dict with `curves`, `x_range`, `parameters`, `annotations` — used by both static and explorer
- `interactive` dict — explorer-native only; single math.js expression with variables/sliders
- `display_mode` — per-spec display mode override for static rendering

## Architecture Notes

- `generate_explorer.py` is self-contained (no imports from other project files)
- `html_renderer.py` imports from `equation_annotator.py` but does not modify it
- `auto_annotate.py` imports from both `html_renderer` and `generate_explorer` on demand
- `TEMPLATE_DIR` in `generate_explorer.py` resolves to `Path(__file__).parent / "templates"` —
  works regardless of working directory

## Pending / Next Steps

1. Add more example specs with `interactive` blocks
2. Fine-tune connector line alignment for tall symbols
3. Generate more annotated equations (Bayes, Euler-Lagrange, etc.)
4. Consider merging `interactive` and `plot` block handling in `generate_explorer.py` into a unified path
5. Support `display_mode` filtering in explorer output
