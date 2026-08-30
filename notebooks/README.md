# Reproducibility notebook

[`DBWFB2_STL10_REPRODUCIBILITY_HW_MATCHED.ipynb`](DBWFB2_STL10_REPRODUCIBILITY_HW_MATCHED.ipynb) is the final executed notebook. It includes STL-10 loading and frozen splits, all four preprocessing arms, five-seed training, validation-only selection, Float and true-INT8 evaluation, deployment package generation, Board500 header generation, and four-part full-test header generation.

The DBWFB2 structural deployment path uses trailing-edge replicate extension, causal RTL phase, even-index decimation, integer coefficients `[27, -17, -78, 273, 614, 273, -78, -17, 27]`, and an arithmetic shift by 10 after each 1-D pass. The selected deployment is seed 42, gain 1, `C1_SHIFT=9`, and `C2_SHIFT=8`.

Cached outputs are preserved as execution evidence. Compact exported tables under [`../results/`](../results/) are easier to audit programmatically.
