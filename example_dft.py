#!/usr/bin/env python3
"""
Example: Discrete Fourier Transform (DFT) — Stuart Riffle style.

Renders the classic color-coded DFT equation annotation to validate the
Equation Annotator tool.
"""

from equation_annotator import annotate_equation, save_figure

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "output"
OUTPUT_NAME = "dft_annotated"
SHOW_PLOT = False
# ============================================================================

# DFT equation segments with colors and labels
# X_k = (1/N) * sum_{n=0}^{N-1} x_n * e^{-i 2 pi k n / N}
dft_segments = [
    {
        "latex": r"$X_k$",
        "color": "#FF6B6B",
        "label": "frequency\nbin k",
    },
    {
        "latex": r"$ = $",
        "color": "#AAAAAA",
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
    },
    {
        "latex": r"$e$",
        "color": "#FF8C42",
        "label": "complex\nexponential",
    },
    {
        "latex": r"$-i$",
        "color": "#C792EA",
        "superscript": True,
        "label": "rotate\nbackwards",
    },
    {
        "latex": r"$\,2\pi$",
        "color": "#89CFF0",
        "superscript": True,
        "label": "a full\ncircle",
    },
    {
        "latex": r"$\,k$",
        "color": "#FF6B6B",
        "superscript": True,
        "label": "for each\nfrequency",
    },
    {
        "latex": r"$\,n/N$",
        "color": "#FFE66D",
        "superscript": True,
        "label": "fraction of\ntime elapsed",
    },
]


# Hierarchical groups: brackets spanning multiple segments
dft_groups = [
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
]

dft_description = (
    "The DFT decomposes a discrete signal of N samples into N frequency components.\n"
    "Each output bin X_k measures how much the signal correlates with a sinusoid at frequency k."
)

dft_use_cases = [
    "Audio processing: Identifying the dominant frequencies in a sound recording",
    "Structural biology: Processing electron density maps in X-ray crystallography",
]


def main():
    print("Rendering DFT equation annotation...")
    fig = annotate_equation(
        dft_segments,
        title="The Discrete Fourier Transform",
        equation_fontsize=38,
        label_fontsize=11,
        show_connectors=True,
        groups=dft_groups,
        description=dft_description,
        use_cases=dft_use_cases,
    )

    print("Saving output:")
    save_figure(fig, OUTPUT_NAME, OUTPUT_DIR)

    if SHOW_PLOT:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(fig)

    print("Done! Check output/ directory.")


if __name__ == "__main__":
    main()
