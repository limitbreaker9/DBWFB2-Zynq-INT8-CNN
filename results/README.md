# Results and captured evidence

## Software

- `software_five_seed_results.csv`: seed-level Float and true-INT8 metrics.
- `software_summary.csv`: five-seed mean and sample standard deviation.
- `software_paired_statistics_and_tost.csv`: paired tests and TOST output from the executed notebook.
- `validation_results_all_arms.csv`: validation results.
- `gain_ablation_validation_only.csv`: validation-only gain selection.
- `dbwfb2_preprocessing_localization.csv`: controlled preprocessing-path localization.
- `split_manifest.json` and `board500_indices.csv`: frozen split evidence.

## Hardware and HLS

- `logs/dbwfb2/hls/dbwfb2_hls_csim_3200.txt`: final HLS C-simulation capture.
- `logs/dbwfb2/uart/`: final Board500 and four-part full-test UART captures.
- `dbwfb2_hls_verification.json`: parsed HLS result and synthesis estimate.
- `dbwfb2_board500_metrics.json`: parsed Board500 metrics.
- `dbwfb2_fpga_full3200_metrics.json`: parsed four-part aggregate.
- `dbwfb2_fpga_confusion_matrix.csv` and `dbwfb2_fpga_class_metrics.csv`: recomputed full-test class results.
- `dbwfb2_latency.csv`: captured per-run and four-part mean timing fields.
- `hardware_resource_comparison.csv`, `hardware_timing_comparison.csv`, and `hardware_power_comparison.csv`: current implementation evidence with explicit scopes.
- `logs/avgpool/uart/avgpool_board500_20261003.txt`: AvgPool Board500 capture associated with the reported bitstream.
- `avgpool_board500_metrics.json` and `avgpool_latency.csv`: exact counts, class-correct values, and raw integer-microsecond timing for that capture.

The derived DBWFB2 HLS, FPGA, Board500, confusion-matrix, class-metric, timing, and subset-index files use true labels, the selected seed-42 software reference, and network predictions. Captured logs preserve the corresponding experimental prediction and summary records. N/A for AvgPool applies only to the preprocessing-hierarchy dynamic-power attribution in hardware_power_comparison.csv; resource hierarchy values are available in hardware_resource_comparison.csv and the archived utilization summary. Development-stage hierarchy values are not substituted for this scope.
