# Equation Annotator

Color-coded educational math equation annotations (Stuart Riffle / Reddit DFT style).

## Project Structure

```
Equation_Annotator/
├── equation_annotator.py   # Main module (rendering logic + CLI)
├── example_dft.py          # DFT demo script
├── requirements.txt        # matplotlib>=3.7
├── CLAUDE.md
├── .gitignore
└── README.md
```

## Architecture

- **Renderer:** matplotlib (no external LaTeX required; optional `use_latex=True`)
- **Input:** Python list of segment dicts with `latex`, `color`, optional `label` and `superscript`
- **Output:** PNG (300 DPI) + SVG via `save_figure()` helper
- **Two-pass rendering:** measure text extents first, then compute layout and render

## Key Design Decisions

- `superscript: True` flag on segments renders them smaller (65% fontsize) and raised
- Label overlap resolution: iterative push-apart algorithm
- Dual interface: editable constants at top of file (Spyder-friendly) + argparse CLI
- matplotlib mathtext by default (no LaTeX install needed); `\displaystyle` not supported in mathtext mode

## Conda Environment

```bash
conda activate equation_annotator
```

## Current Status

- Core rendering works for both simple (Euler's identity) and complex (DFT) equations
- CLI supports JSON input files
- PNG + SVG output at 300 DPI
- Layout tightened but may need further adjustment after hierarchical labeling is added
- Git repo initialized, no commits yet

## Pending / Next Steps

1. **Hierarchical / multilayer labeling** — support groupings of terms, not just individual terms. E.g., a bracket spanning multiple segments with a group-level label like "the exponent" or "correlation with frequency k". This is the primary next feature.
2. **Equation description** — a plain-English description of the equation in simple yet accurate terms, rendered as a text block (above or below the annotation)
3. **Practical use cases** — 2 examples per equation from: physics, biology/biochemistry, comp sci, engineering, or finance
4. Revisit spacing/tightness after multilayer labeling is working
5. Add more example equations (Bayes' theorem, Euler-Lagrange, etc.)
