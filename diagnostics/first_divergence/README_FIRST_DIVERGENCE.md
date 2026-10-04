# DBWFB2 preprocessing first-divergence diagnostic

This experiment localizes the first numerical difference between the released
Python-primary preprocessing reference and the released FPGA raw-image path. It
uses the current public repository chain only. It does not alter training,
calibration, gain, CNN shifts, RTL, publication files, or the released firmware.

## Frozen chain

- Arm: DBWFB2
- Seed: 42
- Input gain: 1
- `C1_SHIFT`: 9
- `C2_SHIFT`: 8
- Python true-INT8 accuracy: 2488/3200 (77.75%)
- HLS CNN-only C-simulation accuracy: 2488/3200 (77.75%)
- Python/HLS final-prediction agreement: 3200/3200 (100%)
- FPGA accuracy: 2497/3200 (78.03125%)
- Python/FPGA final-prediction agreement: 3006/3200 (93.9375%)
- Python/FPGA final-prediction mismatches: 194
- FPGA hardware errors: 0

`current_chain_source_of_truth.json` records the source paths, hashes, and
independent per-image recomputation. The 2488 count is accuracy against true
labels; it is not the Python/HLS agreement count.

## Generate or regenerate the diagnostic set

From the repository root:

```powershell
python diagnostics\first_divergence\generate_first_divergence_vectors.py
```

The default selection is deterministic and contains five current-chain
Python/FPGA mismatches and five matches. Explicit global indices can instead be
supplied with:

```powershell
python diagnostics\first_divergence\generate_first_divergence_vectors.py --indices 8 107 844 1670 2408 0 1 813 1603 2400
```

The generator audits all 3,200 UART `PRED` records, all header label/prediction
arrays, the HLS expected-label array, and all 3,200 HLS per-image records before
writing output. It fails if the released chain or index coverage differs.

Generated artifacts are written under `generated/`:

- `first_divergence_vectors.h`: ten raw images in the released header layout;
- `expected_first_divergence.npz`: RAW, ROW, LL, pre-clipping, CNN input, and
  packed-word expectations;
- `selected_diagnostic_indices.csv`: true/Python/FPGA prediction metadata;
- `expected_first_divergence_summary.csv`: ranges and saturation counts;
- `header_identity_verification.csv`: value-for-value source-header checks.

## Software references

`fpga_intent_preprocess.py` keeps two deliberately separate references. The
first preserves the frozen Python-primary convention used by the public
software prediction array:

1. raw `128x128` unsigned 8-bit input;
2. signed 9-tap integer coefficients
   `[27,-17,-78,273,614,273,-78,-17,27]`;
3. complete 9-sample output windows with trailing last-sample replication;
4. signed 32-bit-compatible accumulation;
5. arithmetic right shift by 10 and signed-16-bit truncation after each pass;
6. PS even-index reads `0,2,...,126` after the horizontal and vertical passes;
7. column-major PS reuse of `Matrix_L` for the vertical pass;
8. gain 1, centering by 127, clipping to `[-128,127]`;
9. four signed INT8 bytes packed into each little-endian `uint32_t` word.

The second reference, `preprocess_observed_controller_sequence`, encodes the
controller behavior established by the complete FPGA trace: centered
even-phase outputs, state retained at the leading edge between consecutive 1-D
transactions, and trailing replication. Keeping the references separate
prevents the frozen Python prediction source from being silently reinterpreted.

## Board run

1. Regenerate the files and confirm every header identity row says `PASS`.
2. Use the ready-to-copy pair in `firmware_package/`: `main_diagnostic.c` and
   `first_divergence_vectors.h`. The latter is hash-identical to the generated
   header. `FIRST_DIVERGENCE_FIRMWARE_CHANGES.md` documents every difference
   from the released full-test firmware.
3. Copy both files into the active Vitis application source directory. Compile
   `main_diagnostic.c` as the only file defining `main` (or copy it as `main.c`)
   and keep `first_divergence_vectors.h` beside it.
4. Rebuild and download only the Vitis application ELF. Keep the current XSA,
   bitstream, DBWFB2 RTL, HLS IP, model, gain, and shifts frozen.
5. Capture the complete UART session at 115200 baud, 8-N-1, no flow control.
6. Save it without editing as, for example,
   `dbwfb2_first_divergence_uart.txt` outside the existing released log set.

Diagnostic UART printing invalidates timing measurements. The experiment is
only for numerical localization.

## Compare the trace

```powershell
python diagnostics\first_divergence\compare_first_divergence.py C:\path\to\dbwfb2_first_divergence_uart.txt
```

The comparator validates all ten image identities and reports, per checkpoint:

- exact matches and mismatches;
- match percentage;
- first mismatch coordinate and values;
- maximum and mean absolute error;
- `+1` and `-1` difference counts;
- implementation-derived right/bottom/corner boundary counts;
- expected and FPGA-side saturation diagnostics.

It classifies the first failing checkpoint conservatively:

- RAW: header/write/readback path;
- ROW: row filtering, trailing boundary, or integer arithmetic path;
- LL: PS reorder or column filtering/boundary path;
- CNN input: centering, clipping, or byte-packing path;
- prediction only: a separate CNN hardware experiment is required.

The classifier localizes a stage. It does not label the cause as boundary,
rounding, clipping, or accumulation unless the observed mismatch pattern
supports that conclusion.

## Return artifact

Provide the complete unedited `dbwfb2_first_divergence_uart.txt`. Do not provide
only screenshots or selected lines; the parser requires every checkpoint row
and begin/end marker.

## Completed trace

The complete trace is preserved at
`captures/first_diagnosis_uart_complete.log`. Its SHA-256 is
`05b2d6a770433c3f770935d8eb3010afc1522b517fd0908e3155715df6a00ce4`.
The result and its conservative interpretation are recorded in
`FIRST_DIVERGENCE_RESULT.md`. In summary, RAW is exact and ROW is the first
difference from the frozen Python-primary convention. A phase/state model
derived from the implementation reproduces all 81,920 captured ROW values,
all 40,960 LL values, and all 40,960 packed CNN-input bytes exactly.
