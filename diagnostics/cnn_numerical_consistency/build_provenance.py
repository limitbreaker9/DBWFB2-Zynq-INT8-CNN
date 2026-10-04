"""Write hashes and roles for the DBWFB2 numerical-consistency diagnostics."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]


SOURCES = {
    "software/int8_packages/dbwfb2/selected_package.json": "frozen selected package metadata",
    "hardware/deployment/dbwfb2/hardware_manifest.json": "frozen deployment metadata",
    "hardware/hls/dbwfb2/test_vectors_3200_cnn_only.h": "frozen packed Python-primary CNN inputs",
    "hardware/hls/dbwfb2/expected_int8_3200.h": "frozen Python-primary predictions used by HLS C-simulation",
    "hardware/hls/dbwfb2/weights.h": "frozen deployed weights and biases",
    "hardware/hls/src/cnn_accel.cpp": "released CNN arithmetic implementation",
    "hardware/hls/src/cnn_accel.h": "released CNN dimensions and shifts",
    "results/logs/dbwfb2/hls/dbwfb2_hls_csim_3200.txt": "authoritative 3,200-image HLS C-simulation log",
    "hardware/deployment/dbwfb2/test_vectors_raw_500_dbwfb2.h": "authoritative Board500 raw vectors and Python predictions",
    "results/board500_indices.csv": "Board500 order to four-class-test global-index map",
    "results/logs/dbwfb2/uart/dbwfb2_board500.txt": "authoritative Board500 FPGA predictions",
    "diagnostics/first_divergence/captures/first_diagnosis_uart_complete.log": "complete ten-image RAW/ROW/LL/CNN-input physical trace",
    "diagnostics/first_divergence/generated/first_diagnosis_complete_comparison/observed_controller_alignment.json": "verified ten-image stream-alignment analysis",
    "diagnostics/first_divergence/generated/first_diagnosis_complete_comparison/observed_controller_expected.npz": "verified hardware-equivalent expected tensors for ten images",
    "diagnostics/first_divergence/generated/full3200_variant_analysis/full3200_variant_analysis.json": "full-3,200 three-variant analysis",
    "diagnostics/first_divergence/generated/full3200_variant_analysis/full3200_variant_predictions.csv": "full-3,200 primary/hardware-equivalent/FPGA predictions",
    "diagnostics/cnn_numerical_consistency/cnn_intermediate_trace.cpp": "diagnostic-only operation-for-operation HLS CNN trace",
    "diagnostics/cnn_numerical_consistency/cnn_trace_python.py": "independent Python true-INT8 CNN trace",
    "diagnostics/cnn_numerical_consistency/build_cnn_trace_comparison.py": "ten-image Python/HLS checkpoint comparator",
    "diagnostics/cnn_numerical_consistency/build_operation_metrics.py": "full-test operation and class-metric builder",
    "diagnostics/cnn_numerical_consistency/verify_board500_raw_identity.py": "Board500/full-test raw-vector identity check",
}

for part in range(4):
    SOURCES[
        f"hardware/deployment/dbwfb2/raw3200_parts/test_vectors_raw_3200_part{part}_dbwfb2.h"
    ] = f"authoritative raw vectors and Python predictions, global part {part}"
    start, end = part * 800, part * 800 + 799
    SOURCES[
        f"results/logs/dbwfb2/uart/dbwfb2_full3200_part{part}_{start:04d}_{end:04d}.txt"
    ] = f"authoritative FPGA UART predictions, global indices {start}-{end}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    records = []
    for relative, purpose in SOURCES.items():
        path = REPOSITORY / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "repository_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "purpose": purpose,
            }
        )
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "DBWFB2 intermediate-tensor and numerical-operation diagnostics",
        "frozen_configuration": {
            "seed": 42,
            "input_gain": 1,
            "C1_SHIFT": 9,
            "C2_SHIFT": 8,
            "dbwfb2_integer_coefficients": [27, -17, -78, 273, 614, 273, -78, -17, 27],
        },
        "artifacts": records,
    }
    (HERE / "provenance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} provenance records")


if __name__ == "__main__":
    main()
