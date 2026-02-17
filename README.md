# Equation Annotator

Create color-coded educational math equation annotations — inspired by [Stuart Riffle's DFT explanation](https://web.archive.org/web/20130927122515/http://www.altdevblogaday.com/2011/05/17/understanding-the-fourier-transform/) and similar Reddit/Twitter visualizations.

Each part of a LaTeX equation gets a distinct color with a descriptive label underneath, all rendered on a dark background.

Includes an **Interactive Equation Explorer** (`generate_explorer.py`) that generates self-contained HTML pages with live Plotly plots and parameter sliders — see [Interactive Explorer](#interactive-explorer) below.

## Installation

```bash
# Create conda environment (recommended)
conda create -n equation_annotator python=3.11 -y
conda activate equation_annotator

# Install dependencies
pip install -r requirements.txt
```

Only **matplotlib** is required. No LaTeX installation needed (uses matplotlib's built-in mathtext renderer).

## Quick Start

### Python API

```python
from equation_annotator import annotate_equation, save_figure

segments = [
    {"latex": r"$e$",      "color": "#FF8C42", "label": "Euler's\nnumber"},
    {"latex": r"$i\pi$",   "color": "#C792EA", "superscript": True, "label": "imaginary\ntimes pi"},
    {"latex": r"$+$",      "color": "#AAAAAA"},
    {"latex": r"$1$",      "color": "#4ECDC4", "label": "unity"},
    {"latex": r"$ = $",    "color": "#AAAAAA"},
    {"latex": r"$0$",      "color": "#FF6B6B", "label": "zero"},
]

fig = annotate_equation(segments, title="Euler's Identity")
save_figure(fig, "euler", "output")
```

### CLI

```bash
# From a JSON file
python equation_annotator.py --input my_equation.json --output output/ --show

# With options
python equation_annotator.py --input my_equation.json --fontsize 42 --dpi 300
```

### Run the DFT Example

```bash
python example_dft.py
# Output: output/dft_annotated.png, output/dft_annotated.svg
```

## Segment Format

Each segment is a dictionary with:

| Key          | Type   | Required | Description                                    |
|-------------|--------|----------|------------------------------------------------|
| `latex`     | str    | Yes      | LaTeX math string (wrapped in `$...$`)         |
| `color`     | str    | Yes      | Color (hex `"#FF6B6B"` or named `"white"`)     |
| `label`     | str    | No       | Annotation text below (use `\n` for newlines)  |
| `superscript` | bool | No       | Render smaller and raised (for exponent parts) |

### JSON Input Format

```json
{
  "title": "My Equation",
  "segments": [
    {"latex": "$X_k$", "color": "#FF6B6B", "label": "output"},
    {"latex": "$=$", "color": "white"},
    {"latex": "$\\sum$", "color": "#4ECDC4", "label": "sum of all"}
  ]
}
```

## Configuration

The script supports two usage modes:

1. **Spyder / interactive:** Edit the `CONFIGURATION` constants at the top of `equation_annotator.py`
2. **CLI:** Use command-line arguments (these override the constants)

### `annotate_equation()` Parameters

| Parameter          | Default       | Description                          |
|-------------------|---------------|--------------------------------------|
| `title`           | `None`        | Title above the equation             |
| `background_color`| `"#1a1a2e"`   | Background color                     |
| `equation_fontsize`| `36`          | Font size for equation segments      |
| `label_fontsize`  | `12`          | Font size for labels                 |
| `title_fontsize`  | `18`          | Font size for the title              |
| `show_connectors` | `True`        | Draw lines from segments to labels   |
| `use_latex`       | `False`       | Use system LaTeX (requires install)  |
| `spacing_scale`   | `1.0`         | Spacing multiplier between segments  |
| `figsize`         | Auto          | `(width, height)` in inches          |

## Tips

- Use matplotlib mathtext (default) for portability. Only set `use_latex=True` if you have a full LaTeX installation and need advanced commands like `\displaystyle`.
- For superscript segments, write the content as plain math (`$k$`) rather than using `^{}` notation — the tool handles the vertical positioning.
- Labels support `\n` for line breaks. Keep labels to 2-3 short lines for best appearance.
- The tool auto-sizes the figure, but you can override with `figsize=(width, height)`.

## Interactive Explorer

`generate_explorer.py` reads the same JSON spec format and produces self-contained interactive HTML files with:
- Color-coded KaTeX equation with labels, connectors, and group brackets
- Slider controls for adjustable parameters
- Live-updating Plotly plot

### Quick Start

```bash
# From a spec file with an interactive block
python generate_explorer.py -i examples/michaelis_menten.json --open

# Batch render a directory of specs
python generate_explorer.py --batch-dir examples/

# Works directly on annotator output specs
python generate_explorer.py -i output/the_hill_equation.json --open

# Via auto_annotate.py (--explorer is not mutually exclusive with --html)
python auto_annotate.py --spec-file output/the_hill_equation.json --explorer
python auto_annotate.py --spec-dir output/ --explorer
```

### Interactive Block Format

Add an `interactive` block to any spec JSON to define custom sliders and a math.js expression:

```json
{
  "title": "Michaelis-Menten Kinetics",
  "segments": [...],
  "interactive": {
    "expression": "Vmax * x / (Km + x)",
    "variables": [
      {"name": "x",    "role": "independent", "min": 0, "max": 100},
      {"name": "Vmax", "role": "parameter",   "default": 10,  "min": 1, "max": 50},
      {"name": "Km",   "role": "parameter",   "default": 20,  "min": 1, "max": 100}
    ],
    "output": {"symbol": "v", "label": "Reaction rate"},
    "plot": {
      "x_axis": {"label": "Substrate [S]"},
      "y_axis": {"label": "Rate v"}
    }
  }
}
```

Specs without an `interactive` block but with a `plot` block (annotator-native format) are automatically converted: numpy expressions become math.js, parameters become sliders.

## License

MIT
