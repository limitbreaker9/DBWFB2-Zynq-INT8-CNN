# DBWFB2 Vivado evidence

The current Vivado 2025.2 reports target `xc7z020clg484-1` at 100 MHz and record:

- 10,622 LUT, 12,136 FF, 67 BRAM tiles, and 4 DSP for the complete placed design;
- WNS +1.591 ns and TNS 0.000 ns after routing;
- 1.741 W dynamic, 0.154 W device-static, and 1.894 W total on-chip power estimates;
- 0.008 W hierarchical dynamic-power attribution for `dbwfb2_97_bram_ctrl_0`.

The controller hierarchy uses 354 LUT, 400 FF, no BRAM, and no DSP. Text reports are the numerical source of record; screenshots in `documentation/` preserve hierarchy views and visual evidence. `vivado_block_diagram.png` is the readable architecture diagram embedded by the root README, while `PL.bd` is the actual Vivado block-design source.
