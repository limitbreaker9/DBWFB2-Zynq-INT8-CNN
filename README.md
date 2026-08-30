# A Zero-DSP Wavelet Preprocessing Integration for Resource-Constrained INT8 CNNs on Edge FPGAs

This repository provides the software, HLS, RTL, firmware, deployment inputs, implementation reports, and raw final logs for a multiplier-free DBWFB2-9/7 LL preprocessing front end integrated with a compact true-INT8 CNN on a Zynq-7000 ZedBoard (`xc7z020clg484-1`). A same-flow 2x2 AvgPool implementation provides the hardware baseline.

The project is an integration and hardware-characterization study. It does not claim that DBWFB2 universally improves accuracy, total resource use, or power.

## System architecture

![Zynq-7000 DBWFB2 and INT8 CNN architecture](hardware/vivado/dbwfb2/documentation/vivado_block_diagram.png)

Raw 128x128 images are written by the Arm processor to input BRAM. The multiplier-free programmable-logic controller performs one separable DBWFB2 pass at a time; the processor performs the row/column transpose and reorder between passes. The 64x64 LL output is quantized and packed, then read from DDR by the HLS CNN through its `m_axi` interface. AXI GPIO carries start/done control.

## Dataset and selection protocol

The experiment uses the airplane, car, dog, and ship classes from STL-10:

- training pool: 2,000 images, 500 per class;
- training split: 1,600 images, 400 per class;
- validation split: 400 images, 100 per class;
- independent test: 3,200 images, 800 per class;
- Board500: a fixed stratified 500-image subset of the independent test set, 125 per class;
- training seeds: 42, 100, 123, 777, and 2024.

Validation alone controls early stopping, checkpoint selection, input gain, quantization shifts, and deployment-seed selection. The frozen indices are in [`results/split_manifest.json`](results/split_manifest.json) and [`results/board500_indices.csv`](results/board500_indices.csv).

## Hardware-matched DBWFB2 preprocessing

The deployed structural path uses:

- causal 9-tap analysis;
- trailing-edge replicate extension;
- even-index decimation;
- integer coefficients `[27, -17, -78, 273, 614, 273, -78, -17, 27]`;
- arithmetic right shift by 10 after each 1-D pass;
- input gain 1;
- CNN shifts `C1_SHIFT=9` and `C2_SHIFT=8`.

The final executed notebook is [`notebooks/DBWFB2_STL10_REPRODUCIBILITY_HW_MATCHED.ipynb`](notebooks/DBWFB2_STL10_REPRODUCIBILITY_HW_MATCHED.ipynb). It contains the four-arm five-seed experiment, true-INT8 inference, selected deployment package generation, Board500 generation, and four-part full-test header generation.

## Software results

Independent-test results over five frozen seeds are reported as mean +/- sample standard deviation.

| Arm | Float accuracy (%) | True-INT8 accuracy (%) | True-INT8 macro-F1 (%) |
|---|---:|---:|---:|
| Full | 78.119 +/- 0.305 | 77.788 +/- 0.795 | 77.696 +/- 0.704 |
| AvgPool | 80.300 +/- 0.225 | 79.156 +/- 0.767 | 79.134 +/- 0.757 |
| Haar | 80.300 +/- 0.225 | 79.156 +/- 0.767 | 79.134 +/- 0.757 |
| DBWFB2 | 79.1625 +/- 0.8302 | 78.3125 +/- 0.6914 | 78.3365 +/- 0.6633 |

The selected DBWFB2 seed-42 true-INT8 result is **2,488/3,200 (77.75%)**. Exact seed-level values are in [`results/software_five_seed_results.csv`](results/software_five_seed_results.csv); aggregate and paired-statistical tables are beside it.

Aligned Haar LL and non-overlapping 2x2 AvgPool are identical in the main software input experiment. Haar is retained as a named software control; no separately routed Haar hardware design is claimed.

## HLS verification

The archived C-simulation log contains all 3,200 predictions:

- Python true-INT8: 2,488/3,200 (77.75%);
- HLS C-simulation: 2,488/3,200 (77.75%);
- Python-to-HLS final-prediction agreement: 3,200/3,200 (100%);
- C-simulation errors: 0.

The final HLS `csynth.rpt` records 6,329,660 cycles, corresponding to 63.2966 ms nominal latency at 100 MHz, with HLS estimates of 70 BRAM resources, 4 DSP, 5,037 FF, and 6,131 LUT. This HLS BRAM estimate is kept distinct from the whole-design Block RAM tile count reported by Vivado implementation. Final-prediction identity does not by itself establish intermediate-tensor bit exactness.

## ZedBoard DBWFB2 results

### Independent 3,200-image FPGA test

The complete test was run as four 800-image parts. The archived `PRED` and summary records cover every global test index exactly once.

| Part | Global indices | FPGA correct | Primary software-reference matches | Hardware errors |
|---:|---:|---:|---:|---:|
| 0 | 0-799 | 639/800 | 759/800 | 0 |
| 1 | 800-1599 | 617/800 | 749/800 | 0 |
| 2 | 1600-2399 | 628/800 | 748/800 | 0 |
| 3 | 2400-3199 | 613/800 | 750/800 | 0 |
| **Total** | **0-3199** | **2,497/3,200 (78.03%)** | **3,006/3,200 (93.94%)** | **0** |

Per-class recall is 677/800 (84.625%) for airplane, 559/800 (69.875%) for car, 745/800 (93.125%) for dog, and 516/800 (64.500%) for ship. The recomputed confusion matrix and class metrics are in [`results/`](results/).

### Board500

The final rebuilt Board500 application completed all 500 images with zero hardware errors:

- airplane: 108/125 (86.4%);
- car: 91/125 (72.8%);
- dog: 120/125 (96.0%);
- ship: 85/125 (68.0%);
- FPGA accuracy: **404/500 (80.8%)**;
- primary software-reference agreement: **473/500 (94.6%)**.

## Same-flow hardware comparison

The tables below use the current Vivado 2025.2 placed/routed reports at a 100 MHz target. Controller-hierarchy and whole-system values are kept separate.

| Scope | Metric | DBWFB2 | AvgPool |
|---|---|---:|---:|
| Preprocessing controller | LUT | 354 | 68 |
| Preprocessing controller | FF | 400 | 82 |
| Preprocessing controller | Block RAM tiles (Vivado) | 0 | 0 |
| Preprocessing controller | DSP | 0 | 0 |
| CNN hierarchy | LUT / FF / Block RAM tiles / DSP (Vivado) | 3,620 / 4,113 / 35 / 4 | 3,620 / 4,113 / 35 / 4 |
| Whole implemented system | LUT | 10,622 | 10,326 |
| Whole implemented system | FF | 12,136 | 11,816 |
| Whole implemented system | Block RAM tiles (Vivado) | 67 | 67 |
| Whole implemented system | DSP | 4 | 4 |

| Whole-system timing | DBWFB2 | AvgPool |
|---|---:|---:|
| WNS at 100 MHz | +1.591 ns | +0.970 ns |
| TNS | 0.000 ns | 0.000 ns |

| Power scope | DBWFB2 | AvgPool |
|---|---:|---:|
| Preprocessing-controller hierarchical dynamic estimate | 0.008 W | 0.001 W |
| Complete-design dynamic estimate | 1.741 W | 1.737 W |
| Device-static estimate | 0.154 W | 0.154 W |
| Complete-design total on-chip estimate | 1.894 W | 1.891 W |

Power values are Vivado post-route tool estimates, not physical board measurements. The controller values are module-attributed hierarchy estimates and must not be compared directly with complete-system totals.

## ARM-observed DBWFB2 latency

The four-part mean end-to-end time is 78.748 ms. Mean component times are 7.239 ms row write, 0.485 ms row PL wall time, 3.699 ms row read, 3.580 ms column reorder/write, 0.242 ms column PL wall time, 1.849 ms column read, 0.243 ms quantization/packing, 0.012 ms cache flush, and 61.332 ms CNN start-to-done. Exact per-part records are in [`results/dbwfb2_latency.csv`](results/dbwfb2_latency.csv).

These ARM-observed wall times are distinct from RTL initiation interval and HLS synthesis-cycle estimates.

## Repository structure

```text
notebooks/                 executed training, quantization, and packaging workflow
software/int8_packages/    selected configuration manifests
firmware/                  DBWFB2 full-test/Board500 and AvgPool Board500 sources
hardware/rtl/              active preprocessing RTL
hardware/hls/              CNN source, testbench, weights, vectors, and reports
hardware/deployment/       raw board inputs, weights, shifts, and manifests
hardware/vivado/           block designs, XDCs, XSAs, reports, and images
results/                   frozen software tables, derived metrics, and final logs
```

## Reproduction entry points

1. Run the executed notebook from a compatible Python environment described by [`requirements.txt`](requirements.txt) to reproduce training, quantization, and deployment-package generation.
2. Use [`hardware/hls/src/`](hardware/hls/src/) with the matching per-arm configuration and weights to reproduce CNN HLS synthesis and C-simulation.
3. Package the RTL and HLS IP, then import the matching `PL.bd` and XDC under [`hardware/vivado/`](hardware/vivado/). Generated Vivado/Vitis caches and intermediate products are intentionally excluded; no unavailable one-command recreation Tcl is claimed.
4. Build the appropriate source under [`firmware/`](firmware/) with its archived deployment header. The full DBWFB2 test selects one of four 800-image parts at build time.
5. Compare the archived HLS/UART records under [`results/logs/`](results/logs/) with the machine-readable summaries documented in [`results/README.md`](results/README.md).

## License

Released under the [MIT License](LICENSE).
