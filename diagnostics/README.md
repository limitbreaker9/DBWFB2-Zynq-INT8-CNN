# Diagnostic evidence and provenance

`provenance/avgpool_hardware_provenance.json` identifies the reported AvgPool seed-42, gain-1, shifts-9/7 implementation. `lena_reproduction/` contains the Lena input, subbands, RTL testbench and normalized energy results.

`first_divergence/` compares the selected software convention with physical preprocessing checkpoints and the hardware-equivalent stream model. `cnn_numerical_consistency/` contains intermediate CNN, GAP, clipping and boundary-only checks. The label `Python-primary` in these records denotes the selected seed-42 software reference, not the hardware-equivalent replay.

Earlier development artifacts are retained under `provenance/avgpool_historical_20260830/` and in the explicitly labeled development comparison. They are not the source of the reported implementation tables. Captured UART logs and numeric arrays are preserved unchanged.
