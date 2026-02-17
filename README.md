# Equation Annotator

Color-coded educational math equation annotations — inspired by [Stuart Riffle's DFT explanation](https://web.archive.org/web/20130927122515/http://www.altdevblogaday.com/2011/05/17/understanding-the-fourier-transform/) and similar visualizations.

Each part of a LaTeX equation gets a distinct color with a descriptive label, optional group brackets showing hierarchical structure, a symbol legend, and an annotated plot — all rendered on a dark background.

Also includes an **Interactive Equation Explorer** (`generate_explorer.py`) that generates self-contained HTML pages with live Plotly plots and parameter sliders.

## How It Works

This tool is designed to be used with **Claude Code** (the Anthropic CLI). The primary workflow is:

1. You ask Claude Code to annotate an equation (e.g., *"annotate the Hill equation"*)
2. Claude reads the `GENERATION_PROMPT` embedded in `auto_annotate.py`
3. Claude generates a JSON spec describing how to color, label, and annotate the equation
4. Claude writes the spec to `output/<equation_name>.json`
5. Claude runs `python auto_annotate.py --spec-file output/<equation_name>.json`

You can also generate specs manually or via the Python API, but **Claude Code is the intended primary interface**. No Claude API key or separate account is required — it runs through Claude Code's built-in tool access.

### Why a `GENERATION_PROMPT`?

`auto_annotate.py` contains a detailed structured prompt (`GENERATION_PROMPT`) that tells Claude exactly how to produce a valid spec: which keys are required, how to choose colors, how to segment the equation, how to write labels, when to include plots, etc. Claude reads this prompt from the file before generating, so the output reliably matches what the renderer expects.

You can inspect it at any time:
```bash
python auto_annotate.py --print-prompt --equation "Michaelis-Menten"
```

---

## Installation

```bash
# Create conda environment (recommended)
conda create -n equation_annotator python=3.11 -y
conda activate equation_annotator

# Install dependencies
pip install -r requirements.txt
```

**Required:** `matplotlib`, `numpy`, `jinja2`
**Optional:** `sympy` (plot expression verification), `weasyprint` (PDF export)

No system LaTeX installation needed — uses matplotlib's built-in mathtext renderer.

---

## Quick Start

### With Claude Code (recommended)

Just ask:
> *"Annotate Bayes' theorem"*
> *"Annotate the logistic growth equation and generate an interactive explorer"*
> *"Annotate these 5 equations: [list] and output as HTML"*

Claude will generate the spec and render everything automatically.

### CLI (manual)

```bash
# Render a spec file → PNG + SVG
python auto_annotate.py --spec-file output/bayes_theorem.json

# Also produce static HTML
python auto_annotate.py --spec-file output/bayes_theorem.json --html

# Also produce an interactive explorer
python auto_annotate.py --spec-file output/bayes_theorem.json --explorer

# All three formats at once
python auto_annotate.py --spec-file output/bayes_theorem.json --html --explorer

# Render all specs in a directory
python auto_annotate.py --spec-dir output/

# Use the built-in DFT example
python auto_annotate.py --use-example

# Print the generation prompt (for debugging / understanding the format)
python auto_annotate.py --print-prompt --equation "Fourier Transform"
```

### Interactive Explorer (standalone)

```bash
# From a spec file with an interactive block
python generate_explorer.py -i examples/michaelis_menten.json --open

# From an annotator output spec (plot block auto-converted)
python generate_explorer.py -i output/hill_equation.json --open

# Batch render a directory
python generate_explorer.py --batch-dir examples/
```

---

## Output Formats

All output goes to `output/` by default (gitignored).

| Flag | Output |
|------|--------|
| *(default)* | `output/<name>.png` + `output/<name>.svg` |
| `--html` | `output/<name>.html` (KaTeX + CSS layout) |
| `--pdf` | `output/<name>.pdf` (requires weasyprint) |
| `--explorer` | `output/<name>_explorer.html` (Plotly sliders, self-contained) |

Flags are combinable: `--html --pdf --explorer` all work together.

---

## Display Modes

Control how much content appears in static (PNG/SVG/HTML) output:

| Mode | Contents |
|------|----------|
| `full` *(default)* | Equation + group brackets + description + use cases + symbol legend + plot + insight |
| `compact` | Equation + description + use cases + symbol legend (no plot) |
| `plot` | Equation + plot only |
| `insight` | Equation + plot + insight text |
| `minimal` | Equation + basic symbol list |

```bash
python auto_annotate.py --spec-file output/dft.json --display-mode compact
```

Per-spec overrides are also supported via `"display_mode"` in the JSON spec.

---

## Batch Modes

```bash
# Render all *.json files in a directory
python auto_annotate.py --spec-dir output/

# Render from a JSON array of spec dicts
python auto_annotate.py --batch-file my_batch.json

# Check which equations in a list already have specs; render existing, report missing
python auto_annotate.py --equations-list equations.json
```

The `--equations-list` file is a JSON array of equation names (strings) or dicts with `{"name": ..., "levels": N, "use_cases": N}`. Missing specs are reported with a prompt to ask Claude Code to generate them.

**Teaching workflow example:**
```bash
python auto_annotate.py --equations-list course-design/biochem_equations.json --html --explorer
```

---

## Label Styles

Claude can generate labels in three styles (default: `mixed`):

| Style | Description |
|-------|-------------|
| `descriptive` | Precise and technical — describes the mathematical role |
| `creative` | Intuitive and evocative — uses metaphors and analogies |
| `mixed` | ~60% creative for key concepts, descriptive for standard terms |

```bash
python auto_annotate.py --print-prompt --equation "Nernst Equation" --label-style creative
```

---

## JSON Spec Format

Specs are plain JSON files. Both tools (`auto_annotate.py` and `generate_explorer.py`) use the same format.

### Minimal spec

```json
{
  "title": "Euler's Identity",
  "segments": [
    {"latex": "$e$",    "color": "#FF8C42", "label": "Euler's\nnumber"},
    {"latex": "$i\\pi$","color": "#C792EA", "superscript": true, "label": "imaginary\ntimes pi"},
    {"latex": "$+$",    "color": "#AAAAAA"},
    {"latex": "$1$",    "color": "#4ECDC4", "label": "unity"},
    {"latex": "$ = $",  "color": "#AAAAAA"},
    {"latex": "$0$",    "color": "#FF6B6B", "label": "zero"}
  ]
}
```

### Full spec (all optional fields)

```json
{
  "title": "Michaelis-Menten Kinetics",
  "segments": [...],
  "groups": [
    {
      "segment_indices": [2, 3, 4],
      "label": "saturation term",
      "color": "#4ECDC4",
      "level": 1
    }
  ],
  "description": "Describes enzyme reaction rate as a function of substrate concentration.",
  "use_cases": [
    "Pharmacology: Modeling drug metabolism by liver enzymes",
    "Food science: Optimizing fermentation reaction conditions"
  ],
  "symbols": [
    {
      "symbol": "V_max",
      "name": "maximum reaction rate",
      "type": "parameter",
      "description": "The reaction rate approached asymptotically at saturating substrate concentrations."
    }
  ],
  "plot": {
    "curves": [
      {
        "expr": "Vmax * x / (Km + x)",
        "color": "#4ECDC4",
        "label": "v vs [S]",
        "style": "-"
      }
    ],
    "x_range": [0, 100],
    "x_label": "Substrate [S]",
    "y_label": "Rate v",
    "parameters": {"Vmax": 10, "Km": 20},
    "annotations": [
      {"type": "hline", "y": 5, "label": "Vmax/2", "color": "#FFE66D", "style": "dashed"},
      {"type": "vline", "x": 20, "label": "Km", "color": "#FF8C42", "style": "dotted"}
    ]
  },
  "insight": "The hyperbolic curve reflects substrate binding saturation...",
  "display_mode": "full"
}
```

### Segment fields

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `latex` | str | Yes | LaTeX math string wrapped in `$...$` |
| `color` | str | Yes | Hex color or named color |
| `label` | str | No | Annotation text; use `\n` for line breaks |
| `superscript` | bool | No | Render smaller and raised (for exponent parts) |

### Group bracket fields

Groups draw bracket annotations spanning multiple segments, with hierarchical nesting:

| Key | Type | Description |
|-----|------|-------------|
| `segment_indices` | list[int] | Contiguous range of segment indices |
| `label` | str | Conceptual label for this group |
| `color` | str | Hex color (typically matching a key segment) |
| `level` | int | 1 = closest to equation; higher = wider spans |

### Interactive block (explorer-native)

Add an `interactive` block for custom slider-driven plots in the explorer:

```json
{
  "title": "Michaelis-Menten",
  "segments": [...],
  "interactive": {
    "expression": "Vmax * x / (Km + x)",
    "variables": [
      {"name": "x",    "role": "independent", "min": 0, "max": 100},
      {"name": "Vmax", "role": "parameter", "default": 10, "min": 1, "max": 50},
      {"name": "Km",   "role": "parameter", "default": 20, "min": 1, "max": 100}
    ],
    "output": {"symbol": "v", "label": "Reaction rate"},
    "plot": {
      "x_axis": {"label": "Substrate [S]"},
      "y_axis": {"label": "Rate v"}
    }
  }
}
```

Specs with a `plot` block but no `interactive` block are automatically converted when rendered as an explorer — numpy expressions become math.js, parameters become sliders.

---

## Python API

```python
from auto_annotate import render_from_spec, render_batch

spec = {
    "title": "My Equation",
    "segments": [
        {"latex": "$E$", "color": "#FF6B6B", "label": "energy"},
        {"latex": "$=$", "color": "#AAAAAA"},
        {"latex": "$mc^2$", "color": "#4ECDC4", "label": "mass ×\nspeed of light²"},
    ]
}

render_from_spec(spec, output_dir="output")
render_from_spec(spec, output_dir="output", display_mode="compact")

# Batch
render_batch([spec1, spec2, spec3], output_dir="output")
```

For interactive explorers:
```python
from generate_explorer import render_explorer_from_spec

render_explorer_from_spec(spec, output_dir="output", output_name="my_equation")
```

---

## Project Structure

```
Equation_Annotator/
├── auto_annotate.py          # Main CLI + GENERATION_PROMPT + Python API
├── equation_annotator.py     # Core matplotlib rendering (PNG/SVG)
├── html_renderer.py          # Static HTML/PDF rendering (KaTeX + Jinja2 / weasyprint)
├── generate_explorer.py      # Interactive HTML generator (Plotly sliders)
├── example_dft.py            # Standalone DFT demo (no Claude required)
├── templates/
│   ├── equation_template.html   # Jinja2 template for static HTML output
│   └── explorer_template.html   # Jinja2 template for interactive explorer
├── examples/                 # Explorer-native JSON specs
│   ├── michaelis_menten.json
│   ├── logistic_growth.json
│   └── euler_identity.json
├── output/                   # Generated files (gitignored)
├── requirements.txt
└── CLAUDE.md                 # Project instructions for Claude Code
```

---

## License

MIT
