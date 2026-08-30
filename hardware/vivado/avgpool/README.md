# AvgPool Vivado evidence

The current Vivado 2025.2 reports target `xc7z020clg484-1` at 100 MHz and record:

- 10,326 LUT, 11,816 FF, 67 BRAM tiles, and 4 DSP for the complete placed design;
- WNS +0.970 ns and TNS 0.000 ns after routing;
- 1.737 W dynamic, 0.154 W device-static, and 1.891 W total on-chip power estimates;
- 0.001 W hierarchical dynamic-power attribution for `avgpool_2x2_bram_ctrl_0`.

The controller hierarchy uses 68 LUT, 82 FF, no BRAM, and no DSP. Text reports are the numerical source of record; screenshots preserve hierarchy views. The supplied XSA is an available platform export dated before the current 30 August 2026 implementation reports, so it is not used as the source for current utilization, timing, or power values.
