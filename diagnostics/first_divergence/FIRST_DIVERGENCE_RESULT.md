# First-divergence result

## Evidence

- UART capture: `captures/first_diagnosis_uart_complete.log`
- Capture size: 1,199,121 bytes
- Images: 10 complete traces, with 10 occurrences of every required begin/end
  marker
- Execution order: 8, 107, 844, 1670, 2408, 0, 1, 813, 1603, 2400
- The diagnostic predictions reproduce the corresponding released UART-log
  predictions for all ten images.

## Result

The first difference from the frozen Python-primary reference is the `ROW`
checkpoint. The raw input read back from BRAM is exact for every image:
163,840/163,840 values match.

The difference is not an unexplained coefficient, accumulation, signed-shift,
or truncation error. For every image, captured `ROW[:, c]` is exactly equal to
the integer FIR reference at full-rate position `2*c - 4` wherever that index
is in range. This gives 7,936/7,936 exact phase-mapped values per image and
79,360/79,360 across the ten-image set. The frozen Python-primary reference
instead decimates the forward complete-window sequence at positions
`0, 2, ..., 126`.

The integrated controller behavior observed by the trace is:

1. centered even-phase filtering, with output centers `0, 2, ..., 126`;
2. four leading samples supplied by state retained from the preceding 1-D
   transaction;
3. four trailing samples supplied by replication of the current transaction's
   final sample;
4. signed integer accumulation with coefficients
   `[27,-17,-78,273,614,273,-78,-17,27]`;
5. arithmetic right shift by 10 followed by signed-16-bit truncation.

For rows 1 through 127 of each image, using the previous row's final input
sample at the leading edge reproduces both leading outputs exactly:
254/254 per image. Modeling the complete transaction sequence, including
state carried through all row passes, all column passes, and into the next
diagnostic image, reproduces every captured intermediate:

| Checkpoint | Exact values | Total values | Max absolute error |
|---|---:|---:|---:|
| RAW BRAM readback | 163,840 | 163,840 | 0 |
| ROW / `Matrix_L` | 81,920 | 81,920 | 0 |
| LL / `Subband_LL` | 40,960 | 40,960 | 0 |
| Packed CNN input bytes | 40,960 | 40,960 | 0 |

The second pass is independently isolated by using the captured `ROW` tensor
as its input. The captured `LL` tensor again has phase offset -4 and matches
3,968/3,968 valid mapped values per image. The full state-carry model matches
all 4,096 LL values per image.

The PS conversion is exact: for every captured LL value, the diagnostic
firmware's CNN input equals `clip(LL - 127, -128, 127)`. Across the ten images,
81 of 40,960 pre-clipping values are below -128 and none are above 127. This
clipping is downstream of the first difference and is not its cause.

## Diagnostic-image predictions

| Global index | True | Frozen Python primary | FPGA | Original selection |
|---:|---:|---:|---:|---|
| 8 | 1 | 3 | 1 | mismatch |
| 107 | 0 | 0 | 3 | mismatch |
| 844 | 1 | 1 | 3 | mismatch |
| 1670 | 2 | 0 | 2 | mismatch |
| 2408 | 3 | 3 | 2 | mismatch |
| 0 | 2 | 0 | 0 | match |
| 1 | 0 | 0 | 0 | match |
| 813 | 1 | 1 | 1 | match |
| 1603 | 2 | 2 | 2 | match |
| 2400 | 3 | 2 | 2 | match |

## Conservative interpretation

For this deterministic five-mismatch/five-match sample, the first divergence
is localized to the horizontal preprocessing stream alignment and its retained
transaction state. All later preprocessing differences follow from that ROW
input difference. The experiment does not require CNN-internal
instrumentation: the captured PS-packed CNN input itself already differs from
the frozen Python-primary CNN input on these cases, while the PS packing rule
is reproduced exactly from the captured LL.

This ten-image experiment does not, by itself, prove that every one of the 194
full-test prediction disagreements has the same cause. It does establish the
first numerical divergence for all five deterministically selected disagreement
cases and all five controls.

## Full-3,200 reproduction

The verified controller model was next run in the exact execution order of the
four released 800-image application runs. Controller state was initialized to
zero at the beginning of each separately downloaded part (indices 0, 800, 1600,
and 2400) and then carried between all 1-D transactions and images within that
part.

The CNN was not reimplemented approximately. The diagnostic host runner calls
the released `hardware/hls/src/cnn_accel.cpp` with the released DBWFB2
`weights.h`, `C1_SHIFT=9`, and `C2_SHIFT=8`. Before analyzing new inputs, the
following gates passed:

- Variant-A packed input: 3,276,800/3,276,800 32-bit words exactly equal to
  `test_vectors_3200_cnn_only.h`;
- Variant-A predictions: 3,200/3,200 equal to
  `expected_int8_3200.h` and the archived HLS result.

The full verified FPGA-state model (Variant C) then produced predictions
identical to all four authoritative FPGA UART logs:

- exact matches: **3,200/3,200**;
- disagreements: **0**;
- agreement: **100.0000%**;
- Variant-C accuracy: **2,497/3,200 = 78.03125%**;
- saved FPGA accuracy: **2,497/3,200 = 78.03125%**.

There are therefore no residual indices requiring additional FPGA tracing.

## Three-variant decomposition

The variants keep the raw images, integer coefficients, shifts, truncation,
gain, clipping, packing, CNN weights, and CNN arithmetic fixed.

- **A — frozen Python primary:** forward complete-window sequence with
  trailing replicate, decimated at forward positions `0,2,...,126`.
- **B — alignment only:** verified centered/even phase, but every row and
  column is independent. Constant extension uses that transaction's own first
  sample at the leading edge and its own final sample at the trailing edge.
- **C — full FPGA state:** verified centered/even phase, preceding-transaction
  state at the leading edge, and current-transaction final-sample replication
  at the trailing edge.

| Preprocessing model | Accuracy | Agreement vs FPGA | Agreement vs Python primary |
|---|---:|---:|---:|
| A — frozen Python primary | 2488/3200 (77.75000%) | 3006/3200 (93.93750%) | 3200/3200 (100.00000%) |
| B — alignment only | 2493/3200 (77.90625%) | 3113/3200 (97.28125%) | 2980/3200 (93.12500%) |
| C — full FPGA state | 2497/3200 (78.03125%) | 3200/3200 (100.00000%) | 3006/3200 (93.93750%) |

| Pair | Matching predictions | Different predictions | Agreement |
|---|---:|---:|---:|
| A vs B | 2980 | 220 | 93.12500% |
| B vs C | 3113 | 87 | 97.28125% |
| A vs C | 3006 | 194 | 93.93750% |

The 220 alignment-associated changes and 87 retained-state-associated changes
are controlled pairwise comparisons; they are not additive because some class
changes overlap or reverse.

## Retained transaction state

The numerical state originates in the continuously clocked convolution
datapath:

- `delay_chain.v`: signed 16-bit registers `x0` through `x8`;
- `dbwfb2_97_bram_ctrl.v`: `xin_reg` and `last_pixel`;
- `lpf_shiftadd.v`: registered partial products `mult614_r`, `mult273_r`,
  `mult78_r`, `mult17_r`, and `mult27_r`;
- `dbwfb2_97_conv.v`: `y_lpf_raw_reg`;
- `output_norm.v`: registered `y_lpf`.

These datapath registers clear only when `rst_n` is asserted. The controller's
`IDLE` state clears its counters and BRAM strobes, but it does not reset the
convolution datapath between `start` transactions. During `FLUSH`, the final
input sample is repeated, leaving the delay chain populated by that value for
the next transaction.

The same core is invoked once for every row and then once for every column, so
state crosses:

```text
row r tail -> row r+1 leading outputs
last row tail -> column 0 leading outputs
column c tail -> column c+1 leading outputs
last column tail -> next image leading outputs (within one application run)
```

A length-9 centered filter has radius four. At even output centers 0 and 2,
the window requires samples before the current transaction; center 4 is the
first downsampled output whose complete window lies inside the current line.
Consequently, exactly the first **two** downsampled output positions can depend
on preceding-transaction state despite the filter having nine taps.

The RTL comments explicitly describe trailing replication. They do not state
that cross-transaction leading state is an intentional boundary convention.
It is therefore reported as verified implemented behavior, not inferred design
intent and not automatically labeled a bug.

## Retained-state tensor effect: B versus C

| Tensor | Numerically changed | Total | Changed fraction | Mean abs. difference (all) | Mean abs. difference (changed only) | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| ROW / Matrix_L | 493,919 | 26,214,400 | 1.884151% | 0.125285 | 6.649428 | 52 |
| LL | 505,593 | 13,107,200 | 3.857368% | 0.336889 | 8.733651 | 89 |

For ROW, only columns 0 and 1 are structurally exposed to retained state:
819,200/26,214,400 positions (3.125%). Actual numerical changes occur at
493,919 positions across 3,196 images; all columns 2 through 63 are exact
between B and C.

For LL, the row-stage difference propagates vertically through columns 0 and
1, while column-transaction state affects rows 0 and 1. The complete affected
coordinate rule is therefore `row in {0,1} OR column in {0,1}`. This border
union contains 806,400/13,107,200 positions (6.152344%). Actual changes are:

- leading rows excluding the 2x2 corner: 250,198;
- leading columns excluding the corner: 245,206;
- 2x2 corner: 10,189;
- interior (`row>=2` and `column>=2`): **0**.

## Scientific interpretation and implementation scope

- **Stream alignment/phase:** A uses a forward complete-window phase; B and C
  use the centered phase observed on FPGA. A versus B changes 220 predictions.
- **Retained transaction state:** B resets line independence through clean
  per-line constant extension; C carries the prior transaction tail. B versus
  C changes 87 predictions and up to 89 integer LL units.
- **Trailing boundary replication:** reproduced exactly in the ten-image trace;
  it is not the first divergence.
- **Fixed-point arithmetic:** integer coefficients, signed accumulation,
  arithmetic `>>10`, signed-16 truncation, clipping, and packing are reproduced
  exactly.
- **CNN arithmetic:** the frozen HLS CNN is validated 3,200/3,200 on Variant A,
  and Variant C reproduces FPGA predictions 3,200/3,200. No CNN-internal trace
  is indicated.

Retained state makes row/column results depend on the preceding transaction
and, within an 800-image run, potentially on the preceding image. This is a
behavior of the integrated controller, not an intrinsic property of DBWFB2.
The reported implementation and hardware-equivalent replay use Variant C.
Its exact prediction agreement does not make the Variant-A software comparison
hardware-equivalent.

Independent first-sample extension, as in Variant B, would require a different
datapath initialization and leading-sample schedule. Clearing registers alone
would instead create zero-leading extension. Variant B differs from the
reported FPGA prediction vector at 87/3,200 indices. Its results must not be
substituted for the reported physical implementation. Agreement with Variant A
would additionally require its forward complete-window phase.

## Machine-readable outputs

- `generated/first_diagnosis_complete_comparison/comparison_summary.json`:
  frozen Python-primary versus captured checkpoints;
- `generated/first_diagnosis_complete_comparison/comparison_stage_summary.csv`;
- `generated/first_diagnosis_complete_comparison/mismatch_locations.csv`;
- `generated/first_diagnosis_complete_comparison/observed_controller_alignment.json`;
- `generated/first_diagnosis_complete_comparison/observed_controller_alignment.csv`;
- `generated/first_diagnosis_complete_comparison/observed_controller_expected.npz`.
- `generated/full3200_variant_analysis/full3200_variant_analysis.json`;
- `generated/full3200_variant_analysis/full3200_variant_predictions.csv`.
