# Hardware artifacts

- `rtl/`: active multiplier-free DBWFB2 and exact 2x2 AvgPool preprocessing RTL.
- `hls/src/`: canonical CNN accelerator source, interface header, and C-simulation testbench.
- `hls/dbwfb2/`: final DBWFB2 CNN configuration, weights, full-test vectors, expected predictions, and synthesis report.
- `hls/avgpool/`: same-CNN synthesis package used in the controlled routed hardware comparison.
- `deployment/`: self-contained weights, shifts, manifests, Board500 inputs, and four-part DBWFB2 full-test inputs.
- `vivado/`: active block designs, constraints, exported platforms, screenshots, and placed/routed reports.

The preprocessing-controller hierarchy and complete implemented system are different reporting scopes. The DBWFB2 controller uses 354 LUT, 400 FF, no BRAM, and no DSP; the current complete DBWFB2 design uses 10,622 LUT, 12,136 FF, 67 BRAM tiles, and 4 DSP blocks. The DSP blocks belong to the CNN/system hierarchy rather than the preprocessing controller.

Vivado power values are tool estimates, not physical board measurements. HLS cycle estimates and RTL initiation intervals are distinct from ARM-observed end-to-end timing.
