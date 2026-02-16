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
- **Hierarchical groups:** optional `groups` list with `segment_indices`, `label`, `color`, `level` — renders brackets spanning multiple segments
- **Description & use cases:** optional `description` string and `use_cases` list rendered below groups
- **Output:** PNG (300 DPI) + SVG via `save_figure()` helper
- **Two-pass rendering:** measure text extents first, then compute layout and render
- **Dynamic vertical layout:** `_compute_vertical_layout()` stacks layers top-down (title → equation → per-term labels → group brackets → description → use cases), converts to figure fractions

## Key Design Decisions

- `superscript: True` flag on segments renders them smaller (65% fontsize) and raised
- Label overlap resolution: iterative push-apart algorithm
- Connector lines use visual center offset (40% of bbox width) to better align with glyph centers
- Group brackets: horizontal line with end ticks, italic labels centered below
- Dual interface: editable constants at top of file (Spyder-friendly) + argparse CLI
- matplotlib mathtext by default (no LaTeX install needed); `\displaystyle` not supported in mathtext mode

## Conda Environment

```bash
conda activate equation_annotator
```

## Current Status

- Core rendering works for both simple (Euler's identity) and complex (DFT) equations
- **Hierarchical labeling implemented** — multi-level group brackets with labels
- **Description + use cases** — rendered below groups as italic text and bulleted list
- CLI supports JSON input files (including `groups`, `description`, `use_cases` fields)
- PNG + SVG output at 300 DPI
- Dynamic vertical layout adapts figure height to content
- Git repo initialized, no commits yet

## Pending / Next Steps

1. Fine-tune connector line alignment for equations with very tall symbols
2. Add more example equations (Bayes' theorem, Euler-Lagrange, etc.)
3. Consider alternative connector styles (dotted, curved) as options
