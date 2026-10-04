# DBWFB2 numerical-consistency diagnostics

This document records an offline diagnostic only. It does not modify the
released firmware, RTL, HLS implementation or deployment package.
The frozen configuration is seed 42, input gain 1,
`C1_SHIFT=9`, `C2_SHIFT=8`, and DBWFB2 integer coefficients
`[27,-17,-78,273,614,273,-78,-17,27]`.

## A. Verification scope

| Verification | Evidence | Status |
|---|---|---|
| Compare intermediate tensors | Physical FPGA RAW, ROW/Matrix_L, LL, and packed CNN input for ten images; Python versus instrumented HLS at 13 CNN checkpoints for the same images | Complete |
| Include Python, C-simulation, and hardware | Physical trace covers preprocessing through CNN input. Python and HLS are compared element-by-element from CNN input through logits for both frozen-primary and hardware-equivalent inputs | Complete as a first-divergence chain; physical CNN-internal tensors were not exposed |
| Identify the first point of divergence | RAW is exact; the first Python-primary/FPGA difference is ROW/Matrix_L | Complete |
| Separate boundary handling and shifts | Phase/alignment, retained transaction state, trailing replication, DBWFB2 `>>10`, Conv1 `>>9`, Conv2 `>>8`, and GAP `>>8` are reported separately | Complete |
| Separate rounding/truncation, clipping, saturation, and accumulation | Explicit integer types/ranges, arithmetic shifts, signed-16 output truncation, actual pre-clipping counts, endpoint counts, ReLU low clipping, and high saturation are reported separately | Complete |
| Compare per-class metrics on the same subset | Python-primary and FPGA precision/recall/F1 plus both confusion matrices are computed from the same final Board500 PRED records | Complete |

The direct physical trace stops at packed CNN input. It does not claim that
Conv1/Conv2/GAP tensors were physically read out of FPGA fabric. This is not
needed to locate the first divergence, because it is already physically
observed at ROW. CNN arithmetic is independently checked between Python and
HLS on the exact same packed inputs, and the hardware-equivalent complete
software path reproduces every saved FPGA prediction.

## B. Existing first-divergence result

The deterministic diagnostic order is:

| Global index | True | Python primary | FPGA | Selection |
|---:|---:|---:|---:|---|
| 8 | 1 | 3 | 1 | mismatch |
| 107 | 0 | 0 | 3 | mismatch |
| 844 | 1 | 1 | 3 | mismatch |
| 1670 | 2 | 0 | 2 | mismatch |
| 2408 | 3 | 3 | 2 | mismatch |
| 0 | 2 | 0 | 0 | match control |
| 1 | 0 | 0 | 0 | match control |
| 813 | 1 | 1 | 1 | match control |
| 1603 | 2 | 2 | 2 | match control |
| 2400 | 3 | 2 | 2 | match control |

The complete UART trace has SHA-256
`05b2d6a770433c3f770935d8eb3010afc1522b517fd0908e3155715df6a00ce4`.
RAW readback is exact for all 163,840 values. Relative to the frozen
Python-primary forward complete-window convention, the first difference is
ROW/Matrix_L for all ten images.

The implemented FPGA stream convention observed in the trace comprises:

1. centered even-phase filtering;
2. an effective `-4` input-position alignment relative to the frozen-primary
   forward-window convention;
3. leading state retained from the preceding 1-D transaction, which can affect
   the first two downsampled outputs;
4. replication of the current transaction's final sample at the trailing edge;
5. signed 32-bit integer accumulation, arithmetic `>>>10`, and signed-16-bit
   output truncation.

Once that implemented convention is modeled, all physical preprocessing
checkpoints are exact:

| Checkpoint | Exact | Total | Maximum absolute error |
|---|---:|---:|---:|
| RAW BRAM readback | 163,840 | 163,840 | 0 |
| ROW / Matrix_L | 81,920 | 81,920 | 0 |
| LL | 40,960 | 40,960 | 0 |
| Packed CNN input | 40,960 | 40,960 | 0 |

The PS packing relation is also exact for all ten images:
`CNN_INPUT = clip(LL - 127, -128, 127)`. The first difference is therefore
upstream of clipping and CNN arithmetic.

The persistent datapath state is visible in RTL: signed 16-bit `x0` through
`x8` in `delay_chain.v`, registered partial products in `lpf_shiftadd.v`,
`y_lpf_raw_reg` in `dbwfb2_97_conv.v`, and registered `y_lpf` in
`output_norm.v`. These registers clear on `rst_n`, not on each controller
`start`. The controller repeats the last sample during `FLUSH`, so the same
core carries state across row-to-row, final-row-to-first-column,
column-to-column, and (within an application part) image-to-image boundaries.
The RTL documents trailing replication but does not document the cross-
transaction leading state as intended behavior; it is therefore called the
implemented FPGA stream convention, not a bug or a design intent.

## C. Full-3,200 hardware-equivalent result

The verified stream model was replayed in the actual four-part execution
order, with state reset at global indices 0, 800, 1600, and 2400. The
instrumented runner calls the unmodified released `cnn_accel()` with the
released weights.

| Result | Count | Percentage |
|---|---:|---:|
| Hardware-equivalent software correct | 2,497/3,200 | 78.03125% |
| Physical FPGA correct | 2,497/3,200 | 78.03125% |
| Hardware-equivalent software vs FPGA | 3,200/3,200 | 100.00000% |
| Residual disagreements | 0/3,200 | 0% |
| Frozen Python primary vs FPGA | 3,006/3,200 | 93.93750% |
| Frozen-primary/FPGA disagreements explained | 194/194 | 100% |

The Board500 raw header was separately checked against its mapped full-test
vectors: 500/500 images and 8,192,000/8,192,000 raw bytes are identical. In
Board500 execution order, the same hardware-equivalent software replay also
matches the saved FPGA predictions 500/500.

The controlled preprocessing decomposition remains:

| Model | Accuracy | Agreement vs FPGA | Agreement vs Python primary |
|---|---:|---:|---:|
| A: frozen Python primary | 2,488/3,200 (77.75000%) | 3,006/3,200 (93.93750%) | 3,200/3,200 |
| B: centered alignment, independent per-line extension | 2,493/3,200 (77.90625%) | 3,113/3,200 (97.28125%) | 2,980/3,200 (93.12500%) |
| C: full implemented FPGA stream convention | 2,497/3,200 (78.03125%) | 3,200/3,200 (100.00000%) | 3,006/3,200 (93.93750%) |

A versus B changes 220 predictions; B versus C changes 87; A versus C
changes 194. These pairwise changes are not additive.

## D. Python versus HLS CNN intermediate results

The diagnostic HLS source is an operation-for-operation copy instrumented to
write checkpoints. It is linked with the unmodified released
`hardware/hls/src/cnn_accel.cpp` and verifies every instrumented prediction
against the released function. The independent Python implementation parses
the released `weights.h`.

Both of these input contexts were tested on the same ten indices:

- the released frozen Python-primary packed CNN inputs; and
- the packed CNN inputs generated by the verified hardware-equivalent stream.

Every checkpoint below has zero mismatches and maximum absolute error zero in
both contexts. The element count shown is per ten-image context; therefore the
complete comparison contains twice the shown count.

| Checkpoint | Shape per image | Type | Elements/context | Mismatches/context | Max abs. error |
|---|---|---|---:|---:|---:|
| CNN_INPUT | 64x64 | INT8 | 40,960 | 0 | 0 |
| CONV1_ACC | 16x64x64 | INT32 | 655,360 | 0 | 0 |
| CONV1_SHIFTED | 16x64x64 | INT32 | 655,360 | 0 | 0 |
| CONV1_ACT | 16x64x64 | UINT8 | 655,360 | 0 | 0 |
| POOL1 | 16x32x32 | UINT8 | 163,840 | 0 | 0 |
| CONV2_ACC | 32x32x32 | INT32 | 327,680 | 0 | 0 |
| CONV2_SHIFTED | 32x32x32 | INT32 | 327,680 | 0 | 0 |
| CONV2_ACT | 32x32x32 | UINT8 | 327,680 | 0 | 0 |
| POOL2 | 32x16x16 | UINT8 | 81,920 | 0 | 0 |
| GAP_SUM | 32 | UINT32 | 320 | 0 | 0 |
| GAP_SHIFTED | 32 | UINT32 | 320 | 0 | 0 |
| LOGITS | 4 | INT32 | 40 | 0 | 0 |
| PREDICTION | scalar | UINT8 | 10 | 0 | 0 |

Thus all 26 checkpoint/context aggregates and every per-image comparison are
exact. Both diagnostic HLS runs report zero prediction differences from the
unmodified released `cnn_accel()`.

## E. GAP verification

`POOL2` is UINT8 with shape 32x16x16. For every image and channel, both
implementations compute:

```text
GAP_SUM[channel] = sum of all 256 POOL2 values, accumulated in uint32_t
GAP_SHIFTED[channel] = GAP_SUM[channel] >> 8
```

All 640 channel cases (10 images x 32 channels x 2 input contexts) satisfy:

- recomputed Python `POOL2` sum = Python `GAP_SUM` = HLS `GAP_SUM`;
- Python `GAP_SHIFTED` = HLS `GAP_SHIFTED` = `GAP_SUM >> 8`.

There are zero mismatches. Because both the POOL2 values and the HLS sum are
unsigned and nonnegative, negative sums cannot occur. The implemented `>>8`
is an unsigned logical shift and equals exact integer floor division by 256;
no signed-negative shift ambiguity applies at GAP. Across the full 3,200
hardware-equivalent inputs, GAP sums range from 0 to 4,158 and shifted outputs
from 0 to 16.

## F. Operation separation

| Operation | Compared paths | Result | First-divergence relevance |
|---|---|---|---|
| Boundary/stream convention | Frozen primary, physical trace, verified FPGA-state model | RAW exact; first difference at ROW; centered phase, retained leading state, and trailing replication reproduce ROW/LL exactly | First divergence |
| DBWFB2 accumulation | RTL INT32 expression and verified model | Full-test observed range: row -9,619..269,392; column -9,537..273,891; no INT32 overflow | No residual once stream convention matches |
| DBWFB2 `>>10` and truncation | Verified model vs captured ROW/LL | Arithmetic right shift, then low signed 16 bits; observed shifted range -10..267, so no signed-16 overflow; captured outputs exact | Not the first divergence |
| CNN-input clipping | Captured LL/input and full software replay | Ten-image packing exact. Full test: 40,771/13,107,200 values actually clipped (0.311058%): 21,694 low and 19,077 high | Downstream of ROW divergence |
| Conv1 `>>9` | Python vs HLS | Both input contexts exact. Full-test INT32 accumulator -40,317..40,939; shifted -79..79 | Not a divergence |
| Conv1 clipping/saturation | Instrumented HLS, same frozen arithmetic | 114,639,568/209,715,200 shifted values are below zero and are ReLU-clipped (54.664406%); zero values are counted separately; values above 127: 0 | Not a divergence |
| Conv2 `>>8` | Python vs HLS | Both input contexts exact. Full-test INT32 accumulator -17,713..22,583; shifted -70..88 | Not a divergence |
| Conv2 clipping/saturation | Instrumented HLS, same frozen arithmetic | 53,347,784/104,857,600 shifted values are below zero and are ReLU-clipped (50.876411%); values above 127: 0 | Not a divergence |
| GAP `>>8` | Python vs HLS, all 640 diagnostic channels | Exact UINT32 sum and unsigned shift in every case | Not a divergence |
| Dense/logits | Python vs HLS | 80/80 diagnostic logits across two contexts exact; full-test logit range -9,989..3,482; predictions match released function | Not a divergence |

For CNN input, "actually clipped" means the pre-clipped signed value lies
outside `[-128,127]`. This differs from merely being equal to an endpoint.
After clipping, 46,674 values equal -128 and 36,681 equal +127
(83,355/13,107,200 = 0.635948%). The actual clipping count is the smaller
40,771 value above.

For Conv1/Conv2, "low clipped" means shifted value `<0`; an exact zero is not
classified as a clipping event. "High saturated" means shifted value `>127`.
Output-zero counts include both negative-clipped values and values that were
already exactly zero. No high saturation occurs in either layer on the full
hardware-equivalent test input.

Signed Conv1/Conv2 shifts operate on INT32 and are arithmetic. Negative values
are sign-extended; the exact Python/HLS comparisons include negative shifted
values and show no rounding or truncation discrepancy. No offset or
round-to-nearest term is added: discarded low bits implement the deployed
arithmetic-shift behavior.

## G. Board500 same-subset metrics

The following values are parsed from the same 500 `PRED` rows. Each class has
125 true samples.

| Class | Python precision | FPGA precision | Python recall | FPGA recall | Python F1 | FPGA F1 |
|---|---:|---:|---:|---:|---:|---:|
| airplane | 69.2308% | 68.7898% | 86.4000% | 86.4000% | 76.8683% | 76.5957% |
| car | 92.0792% | 90.0990% | 74.4000% | 72.8000% | 82.3009% | 80.5310% |
| dog | 85.0000% | 85.7143% | 95.2000% | 96.0000% | 89.8113% | 90.5660% |
| ship | 85.4369% | 83.3333% | 70.4000% | 68.0000% | 77.1930% | 74.8899% |

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Python primary | 408/500 (81.6000%) | 81.5434% |
| Hardware-equivalent software | 404/500 (80.8000%) | 80.6457% |
| FPGA | 404/500 (80.8000%) | 80.6457% |

Python-primary confusion matrix (rows true, columns predicted; airplane, car,
dog, ship):

```text
108   3   6   8
 17  93   8   7
  5   1 119   0
 26   4   7  88
```

FPGA and hardware-equivalent confusion matrix:

```text
108   4   7   6
 16  91   8  10
  3   1 120   1
 30   5   5  85
```

The FPGA's airplane prediction column contains 157 predictions, of which 49
are false positives: 16 cars, 3 dogs, and 30 ships. Those errors reduce
airplane **precision**, not airplane recall. The 16 cars and 30 ships also
reduce the recall of their own true classes, giving car recall 72.8% and ship
recall 68.0%.

## H. Full-3,200 class metrics

| Model | Class | Precision | Recall | F1 | Correct/support |
|---|---|---:|---:|---:|---:|
| Python primary | airplane | 64.9706% | 83.0000% | 72.8869% | 664/800 |
| Python primary | car | 87.1272% | 69.3750% | 77.2443% | 555/800 |
| Python primary | dog | 87.1765% | 92.6250% | 89.8182% | 741/800 |
| Python primary | ship | 76.4110% | 66.0000% | 70.8250% | 528/800 |
| Hardware-equivalent software | airplane | 65.0962% | 84.6250% | 73.5870% | 677/800 |
| Hardware-equivalent software | car | 87.6176% | 69.8750% | 77.7469% | 559/800 |
| Hardware-equivalent software | dog | 86.7288% | 93.1250% | 89.8131% | 745/800 |
| Hardware-equivalent software | ship | 77.8281% | 64.5000% | 70.5400% | 516/800 |
| FPGA | airplane | 65.0962% | 84.6250% | 73.5870% | 677/800 |
| FPGA | car | 87.6176% | 69.8750% | 77.7469% | 559/800 |
| FPGA | dog | 86.7288% | 93.1250% | 89.8131% | 745/800 |
| FPGA | ship | 77.8281% | 64.5000% | 70.5400% | 516/800 |

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Python primary | 2,488/3,200 (77.7500%) | 77.6936% |
| Hardware-equivalent software | 2,497/3,200 (78.03125%) | 77.9217% |
| FPGA | 2,497/3,200 (78.03125%) | 77.9217% |

Current Python-primary confusion matrix:

```text
664  34  34  68
116 555  44  85
 38  11 741  10
204  37  31 528
```

Current hardware-equivalent/FPGA confusion matrix:

```text
677  33  35  55
112 559  48  81
 34  10 745  11
217  36  31 516
```

## I. Remaining unresolved issues

There are no unexplained prediction mismatches in the saved current chain:
hardware-equivalent software and FPGA agree 3,200/3,200, and the Board500
replay agrees 500/500.

Two scope limits remain and are stated explicitly rather than inferred:

1. physical FPGA Conv1/Conv2/GAP tensors were not captured; their arithmetic
   is validated by Python/HLS intermediate equality, not direct fabric
   readback;
2. the DBWFB2 pre-shift accumulator was not separately exposed by firmware,
   but the RTL-exact INT32 accumulation and shift reproduce every captured
   ROW/LL value under the verified stream convention.

The public package metadata describes an RTL structural preprocessing
reference but does not encode the newly observed retained cross-transaction
state in detail. The diagnostic evidence, not that abbreviated descriptor, is
the source for the first-divergence conclusion.

## J. Interpretation

The evidence identifies the first physical divergence in the reported
implementation,
separates phase/state/boundary behavior from shifts and clipping, verifies
Python/HLS CNN arithmetic through GAP and logits, reproduces every current
FPGA prediction, and supplies same-subset precision/recall/F1.

The selected software preprocessor is not bit-exact with FPGA. The separate
hardware-equivalent software replay reproduces the saved FPGA predictions
3,200/3,200. If the design is later
changed to enforce independent row/column boundaries, all FPGA results and
implementation reports affected by that RTL change must be rerun; no such
change was made in this diagnostic.

## Machine-readable evidence

- `diagnostic_indices.csv`
- `cnn_intermediate_comparison.json`
- `cnn_intermediate_comparison.csv`
- `gap_exactness.csv`
- `operation_audit.json`
- `board500_python_fpga_class_metrics.csv`
- `board500_python_confusion_matrix.csv`
- `board500_fpga_confusion_matrix.csv`
- `full3200_class_metrics.csv`
- `provenance.json`
