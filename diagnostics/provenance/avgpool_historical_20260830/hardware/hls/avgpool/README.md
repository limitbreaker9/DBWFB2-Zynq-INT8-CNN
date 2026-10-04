# AvgPool HLS package

The public AvgPool deployment configuration and quantized weights use seed 42, gain 1, `C1_SHIFT=9`, and `C2_SHIFT=7`.

The archived `csynth.rpt` came from the current same-flow routed comparison project, where the CNN HLS hierarchy was held structurally identical to the DBWFB2 design. It supports the controlled hardware-structure comparison, not a fresh AvgPool board-accuracy claim. No final AvgPool UART capture was found in the supplied evidence set, so this repository does not publish a new AvgPool FPGA-accuracy result.
