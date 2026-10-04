# AvgPool Vivado evidence

The current Vivado 2025.2 reports target `xc7z020clg484-1` at 100 MHz and record:

- 10,327 LUT, 11,817 FF, 67 BRAM tiles, and 4 DSP for the complete placed design in the October 3 run;
- WNS +0.776 ns and TNS 0.000 ns after routing;
- 1.736 W dynamic, 0.154 W device-static, and 1.889 W total on-chip power estimates.

The reported build uses selected AvgPool seed-42 tensors and shifts 9/7. The XSA contains the bitstream programmed before the reported Board500 capture.

The [October-3 implementation hierarchy](documentation/utilization_hierarchy_20261003.png), captured on October 4, shows 68 LUTs, 82 FFs, no BRAM tiles and no DSPs for the AvgPool controller, and 3,622 LUTs, 4,114 FFs, 35 BRAM tiles and 4 DSPs for the CNN hierarchy. Subtracting these from the whole system gives 6,637 LUTs, 7,621 FFs, 32 BRAM tiles and no DSPs for the residual platform/interconnect. These values are listed in the [resource comparison CSV](../../../results/hardware_resource_comparison.csv).

A separate AvgPool preprocessing-hierarchy dynamic-power attribution remains unavailable. The other images in `documentation/` show the August-30 development implementation. Earlier development artifacts are preserved under `diagnostics/provenance/avgpool_historical_20260830/`. Use the text reports and [manifest](../../../diagnostics/provenance/avgpool_hardware_provenance.json) for the reported numerical results.

The routed power report has Simulation Activity File `---`, overall Medium confidence, High clock/input-I/O activity confidence, and Medium internal activity confidence; less than 25% of internal nodes are user-specified. No exact default toggle rate is claimed.
