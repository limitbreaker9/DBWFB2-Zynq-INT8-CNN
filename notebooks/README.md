# Reproducibility notebook

[`DBWFB2_STL10_REPRODUCIBILITY.ipynb`](DBWFB2_STL10_REPRODUCIBILITY.ipynb) is the final executed notebook. It includes STL-10 loading and frozen splits, all four preprocessing arms, five-seed training, validation-only selection, Float and true-INT8 evaluation, deployment package generation, Board500 header generation, four-part full-test header generation, and the frozen software--FPGA numerical-consistency analysis.

The frozen five-seed comparison uses trailing-edge replicate extension with forward complete-window sampling. The separate hardware-equivalent DBWFB2 replay reproduces centered even-phase alignment, the effective -4 input-position offset, retained transaction state, trailing replicate extension, deployed integer arithmetic and packing. The selected DBWFB2 deployment is seed 42, gain 1, `C1_SHIFT=9`, and `C2_SHIFT=8`. AvgPool is seed 42, gain 1, and shifts 9/7.

Saved outputs are preserved as execution evidence. Exported tables are under [`../results/`](../results/). The consistency section reads stored diagnostic results without retraining or changing the selected configuration.
