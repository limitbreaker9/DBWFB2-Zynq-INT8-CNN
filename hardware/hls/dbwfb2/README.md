# DBWFB2 HLS package

This directory contains the final DBWFB2 CNN configuration (`C1_SHIFT=9`, `C2_SHIFT=8`), quantized weights, 3,200 packed CNN-only inputs, Python true-INT8 expected predictions, and the Vivado/Vitis HLS 2025.2 synthesis report.

The archived C-simulation log is at `results/logs/dbwfb2/hls/dbwfb2_hls_csim_3200.txt`. It records 2,488/3,200 correct predictions and 3,200/3,200 agreement with the Python true-INT8 final predictions.
