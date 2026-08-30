# Selected INT8 configurations

The per-arm `selected_package.json` files record the selected seed, gain, shifts, dtypes, storage, and preprocessing convention. Quantized weight arrays are archived as C headers under `hardware/deployment/` and `hardware/hls/`; the final executed notebook regenerates the corresponding software package arrays.

- DBWFB2: seed 42, gain 1, `C1_SHIFT=9`, `C2_SHIFT=8`.
- AvgPool: seed 42, gain 1, `C1_SHIFT=9`, `C2_SHIFT=7`.
- Haar: seed 42, gain 1, `C1_SHIFT=9`, `C2_SHIFT=7`.

Aligned Haar LL and non-overlapping 2x2 AvgPool are identical in the main software input experiment. No separately routed Haar FPGA implementation is claimed.
