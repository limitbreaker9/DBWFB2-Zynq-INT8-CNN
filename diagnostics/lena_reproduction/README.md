# Lena subband-energy reproduction

This offline replay follows the 256x256 Lena firmware transaction order.
It does not run any CNN, training, calibration, or physical FPGA experiment.
The original firmware, input header, six RTL modules, and two BRAM configuration
files are included in `source/`, `rtl/`, and `configuration/`. Their bytes match
the source hashes recorded with the captured results.

The original six RTL modules are simulated using Icarus Verilog. The BRAM
read latency is one clock, as explicitly configured in both historical XCIs.
The original controller's nine-cycle write gate and persistent pipeline state
are preserved. The firmware's execution order is 256 row transactions, then
128 even columns with a low-stream and high-stream transaction per column.
Vertical output reads retain even row addresses. The HPF uses taps x0-x6;
it is not delayed to artificially align its center with the nine-tap LPF.

LPF: [27, -17, -78, 273, 614, 273, -78, -17, 27], arithmetic shift by 10.
HPF: [3, -2, -19, 36, -19, -2, 3], arithmetic shift by 6.
Both paths use the original signed shift-and-add RTL and low-16-bit output
truncation. No rounding or saturation is introduced. Trailing last-sample
replication and retained inter-transaction state are reproduced as implemented.

To reproduce from a clone, install Python 3 and Icarus Verilog (`iverilog` and
`vvp` on PATH). From the repository root, run:

```text
python diagnostics/lena_reproduction/reproduce_lena.py --output-dir ../lena-output
```

The output directory must not already exist. For the idle-gap check, use
`--idle 256 --output-dir ../lena-output-idle256`. Explicit simulator paths may
be supplied with `--iverilog` and `--vvp`. Each input buffer is written over 256
clock cycles before starting a transaction; pipeline state is not reset.

Double-precision energies:

| Subband | Raw energy | Normalized percent | Stored reference | Absolute difference (percentage points) |
|---|---:|---:|---:|---:|
| LL | 282096500 | 99.42977672556906 | 99.429778 | 0.000001274430943 |
| LH | 339463 | 0.1196495890469816 | 0.119650 | 0.000000410953018 |
| HL | 935871 | 0.3298638748581958 | 0.329864 | 0.000000125141804 |
| HH | 342471 | 0.12070981052576817 | 0.120710 | 0.000000189474232 |

Total energy: 283714305. The two idle-gap simulations produce identical
subband samples and energies. Reproduction is near-exact, not identical at
every sixth decimal. All four results match the original firmware's five-
decimal display precision and the manuscript's two-decimal display precision.
The saved JSON records source hashes, input-pixel hash, BRAM settings, commands,
energies, target differences, and simulator output. `subbands.txt` contains
16384 rows in LL/LH/HL/HH order. `lena_pixels.hex` is derived from the original
header, which is included in `source/`. The saved records identify the original
run script and testbench. The implementation manifest separately identifies the
portable script and comment-only testbench update published here.
