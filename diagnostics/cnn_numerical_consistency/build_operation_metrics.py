"""Build the full operation audit and same-subset class metrics.

This is diagnostic-only code.  It imports the already verified preprocessing
model from diagnostics/first_divergence and links the diagnostic CNN trace
against the unmodified released cnn_accel implementation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
FIRST = REPOSITORY / "diagnostics/first_divergence"
sys.path.insert(0, str(FIRST))

from fpga_intent_preprocess import DBWFB2_COEFFICIENTS  # noqa: E402
from run_full3200_variants import (  # noqa: E402
    IMAGE_COUNT,
    LL_SIZE,
    RAW_SIZE,
    as_i16,
    full_state_image,
    load_raw_images,
)


GENERATED = HERE / "generated"
TRACE_CPP = HERE / "cnn_intermediate_trace.cpp"
TRACE_EXE = GENERATED / "cnn_intermediate_trace.exe"
CNN_CPP = REPOSITORY / "hardware/hls/src/cnn_accel.cpp"
CNN_H = REPOSITORY / "hardware/hls/src/cnn_accel.h"
WEIGHTS = REPOSITORY / "hardware/hls/dbwfb2/weights.h"
FULL_VARIANTS = (
    FIRST
    / "generated/full3200_variant_analysis/full3200_variant_predictions.csv"
)
FULL_ANALYSIS = (
    FIRST
    / "generated/full3200_variant_analysis/full3200_variant_analysis.json"
)
BOARD_LOG = REPOSITORY / "results/logs/dbwfb2/uart/dbwfb2_board500.txt"
BOARD_INDICES = REPOSITORY / "results/board500_indices.csv"
HLS_LOG = REPOSITORY / "results/logs/dbwfb2/hls/dbwfb2_hls_csim_3200.txt"
CLASS_NAMES = ["airplane", "car", "dog", "ship"]
PART_RESETS = {0, 800, 1600, 2400}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filter_lines_with_accumulator(
    lines: np.ndarray, leading_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Verified centered/even-phase filter with explicit pre-shift INT32 sum."""

    source = np.asarray(lines, dtype=np.int64)
    leading = np.repeat(
        np.asarray(leading_values, dtype=np.int64)[..., None], 4, axis=-1
    )
    trailing = np.repeat(source[..., -1:], 4, axis=-1)
    extended = np.concatenate((leading, source, trailing), axis=-1)
    accumulator = np.zeros(source.shape[:-1] + (LL_SIZE,), dtype=np.int64)
    for tap, coefficient in enumerate(DBWFB2_COEFFICIENTS.tolist()):
        accumulator += int(coefficient) * extended[..., tap : tap + 128 : 2]
    if accumulator.min() < -(1 << 31) or accumulator.max() > (1 << 31) - 1:
        raise OverflowError("DBWFB2 accumulator exceeded signed INT32")
    shifted = as_i16(np.right_shift(accumulator, 10))
    return accumulator.astype(np.int32), shifted


def full_state_image_audited(
    raw: np.ndarray, transaction_last: int
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    row_leading = np.empty(128, dtype=np.int64)
    row_leading[0] = int(transaction_last)
    row_leading[1:] = raw[:-1, -1]
    row_acc, row = filter_lines_with_accumulator(raw, row_leading)

    columns = row.T
    column_leading = np.empty(64, dtype=np.int64)
    column_leading[0] = int(raw[-1, -1])
    column_leading[1:] = columns[:-1, -1]
    col_acc, ll_columns = filter_lines_with_accumulator(columns, column_leading)
    ll = ll_columns.T
    return row, ll, int(columns[-1, -1]), row_acc, col_acc


def compile_trace() -> None:
    command = [
        "g++",
        "-O3",
        "-std=c++17",
        str(TRACE_CPP),
        str(CNN_CPP),
        f"-I{CNN_H.parent}",
        f"-I{WEIGHTS.parent}",
        "-o",
        str(TRACE_EXE),
    ]
    subprocess.run(command, cwd=REPOSITORY, check=True)


def run_trace(inputs: np.ndarray, name: str) -> tuple[np.ndarray, dict[str, int]]:
    input_path = GENERATED / f"{name}_cnn_input.bin"
    prefix = GENERATED / name
    np.ascontiguousarray(inputs).tofile(input_path)
    subprocess.run(
        [str(TRACE_EXE), str(input_path), str(prefix), "--aggregate-only"],
        cwd=REPOSITORY,
        check=True,
    )
    predictions = np.fromfile(str(prefix) + "_PREDICTION.bin", dtype=np.uint8)
    audit = {
        row["metric"]: int(row["value"])
        for row in csv.DictReader(
            Path(str(prefix) + "_AUDIT.csv").open(encoding="utf-8")
        )
    }
    return predictions, audit


def confusion(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.int64)
    for y, p in zip(true, pred):
        matrix[int(y), int(p)] += 1
    return matrix


def class_rows(model: str, true: np.ndarray, pred: np.ndarray) -> list[dict]:
    matrix = confusion(true, pred)
    output = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        tp = int(matrix[class_id, class_id])
        support = int(matrix[class_id].sum())
        predicted = int(matrix[:, class_id].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output.append(
            {
                "model": model,
                "class_id": class_id,
                "class_name": class_name,
                "support": support,
                "predicted_count": predicted,
                "true_positive": tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return output


def summary_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    rows = class_rows("temporary", true, pred)
    return {
        "correct": int(np.count_nonzero(true == pred)),
        "total": int(true.size),
        "accuracy": float(np.mean(true == pred)),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
    }


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class\\predicted_class", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([name, *map(int, row)])


def parse_board_log() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pattern = re.compile(r"^PRED,(\d+),(\d+),(\d+),(\d+),(\d+)$")
    records: dict[int, tuple[int, int, int]] = {}
    for line in BOARD_LOG.read_text(encoding="utf-8", errors="strict").splitlines():
        match = pattern.match(line.strip())
        if match:
            local, true, primary, _alt, fpga = map(int, match.groups())
            if local in records:
                raise ValueError(f"duplicate Board500 local index {local}")
            records[local] = (true, primary, fpga)
    if set(records) != set(range(500)):
        raise ValueError("Board500 log does not contain local indices 0..499 exactly once")
    true = np.asarray([records[i][0] for i in range(500)], dtype=np.uint8)
    primary = np.asarray([records[i][1] for i in range(500)], dtype=np.uint8)
    fpga = np.asarray([records[i][2] for i in range(500)], dtype=np.uint8)
    if np.count_nonzero(fpga == true) != 404 or np.count_nonzero(fpga == primary) != 473:
        raise ValueError("Board500 primary evidence differs from frozen sanity checks")
    return true, primary, fpga


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    raw = load_raw_images()
    full_rows = list(csv.DictReader(FULL_VARIANTS.open(encoding="utf-8")))
    if len(full_rows) != 3200:
        raise ValueError("full variant prediction file is not 3,200 rows")
    true = np.asarray([int(row["true_label"]) for row in full_rows], dtype=np.uint8)
    primary = np.asarray(
        [int(row["frozen_python_primary_prediction"]) for row in full_rows],
        dtype=np.uint8,
    )
    expected_c = np.asarray(
        [int(row["variant_C_prediction"]) for row in full_rows], dtype=np.uint8
    )
    fpga = np.asarray(
        [int(row["fpga_prediction"]) for row in full_rows], dtype=np.uint8
    )
    if not np.array_equal(expected_c, fpga):
        raise ValueError("existing full-3,200 Variant C prediction vector differs from FPGA")

    cnn_input = np.empty((3200, 64, 64), dtype=np.int8)
    dbw_acc_min = (1 << 31) - 1
    dbw_acc_max = -(1 << 31)
    row_acc_min = (1 << 31) - 1
    row_acc_max = -(1 << 31)
    col_acc_min = (1 << 31) - 1
    col_acc_max = -(1 << 31)
    clipped_low = clipped_high = endpoint_low = endpoint_high = 0
    state = 0
    for index in range(3200):
        if index in PART_RESETS:
            state = 0
        row, ll, state, row_acc, col_acc = full_state_image_audited(raw[index], state)
        verified_row, verified_ll, verified_state = full_state_image(
            raw[index], 0 if index in PART_RESETS else previous_state
        )
        if not np.array_equal(row, verified_row) or not np.array_equal(ll, verified_ll):
            raise ValueError(f"audited preprocessor differs from verified model at {index}")
        if state != verified_state:
            raise ValueError(f"state mismatch at {index}")
        previous_state = state
        row_acc_min = min(row_acc_min, int(row_acc.min()))
        row_acc_max = max(row_acc_max, int(row_acc.max()))
        col_acc_min = min(col_acc_min, int(col_acc.min()))
        col_acc_max = max(col_acc_max, int(col_acc.max()))
        preclip = ll.astype(np.int32) - 127
        clipped_low += int(np.count_nonzero(preclip < -128))
        clipped_high += int(np.count_nonzero(preclip > 127))
        quantized = np.clip(preclip, -128, 127).astype(np.int8)
        endpoint_low += int(np.count_nonzero(quantized == -128))
        endpoint_high += int(np.count_nonzero(quantized == 127))
        cnn_input[index] = quantized
    dbw_acc_min = min(row_acc_min, col_acc_min)
    dbw_acc_max = max(row_acc_max, col_acc_max)

    compile_trace()
    full_c_prediction, cnn_audit = run_trace(cnn_input, "full3200_hardware_equivalent")
    if not np.array_equal(full_c_prediction, expected_c):
        bad = np.flatnonzero(full_c_prediction != expected_c)
        raise ValueError(f"new full CNN trace differs from Variant C at {bad[:20].tolist()}")

    board_true, board_primary, board_fpga = parse_board_log()
    board_map_rows = list(csv.DictReader(BOARD_INDICES.open(encoding="utf-8")))
    if len(board_map_rows) != 500:
        raise ValueError("Board500 index mapping does not contain 500 rows")
    board_global = np.asarray(
        [int(row["four_class_test_index"]) for row in board_map_rows], dtype=np.int64
    )
    if len(set(board_global.tolist())) != 500 or np.any(board_global < 0) or np.any(board_global >= 3200):
        raise ValueError("Board500 global index mapping is not 500 unique valid indices")
    board_input = np.empty((500, 64, 64), dtype=np.int8)
    state = 0
    for position, global_index in enumerate(board_global):
        _row, ll, state = full_state_image(raw[global_index], state)
        board_input[position] = np.clip(ll.astype(np.int32) - 127, -128, 127).astype(np.int8)
    board_c, board_c_audit = run_trace(board_input, "board500_hardware_equivalent")
    if not np.array_equal(board_c, board_fpga):
        bad = np.flatnonzero(board_c != board_fpga)
        raise ValueError(
            "Board500 hardware-equivalent replay differs from FPGA at "
            f"local indices {bad[:20].tolist()}"
        )

    board_models = {
        "Python-primary": board_primary,
        "Hardware-equivalent software": board_c,
        "FPGA": board_fpga,
    }
    board_class = []
    for model, prediction in board_models.items():
        board_class.extend(class_rows(model, board_true, prediction))
    with (HERE / "board500_python_fpga_class_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(board_class[0]))
        writer.writeheader()
        writer.writerows(board_class)
    write_matrix(HERE / "board500_python_confusion_matrix.csv", confusion(board_true, board_primary))
    write_matrix(HERE / "board500_fpga_confusion_matrix.csv", confusion(board_true, board_fpga))

    full_models = {
        "Python-primary": primary,
        "Hardware-equivalent software": full_c_prediction,
        "FPGA": fpga,
    }
    full_class = []
    for model, prediction in full_models.items():
        full_class.extend(class_rows(model, true, prediction))
    with (HERE / "full3200_class_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(full_class[0]))
        writer.writeheader()
        writer.writerows(full_class)

    total_input = int(cnn_input.size)
    conv1_total = cnn_audit["conv1_elements"]
    conv2_total = cnn_audit["conv2_elements"]
    operation = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_configuration": {
            "seed": 42,
            "input_gain": 1,
            "C1_SHIFT": 9,
            "C2_SHIFT": 8,
            "dbwfb2_integer_coefficients": DBWFB2_COEFFICIENTS.tolist(),
        },
        "full3200_prediction_validation": {
            "hardware_equivalent_vs_fpga_matches": int(np.count_nonzero(full_c_prediction == fpga)),
            "total": 3200,
            "hardware_equivalent_correct": int(np.count_nonzero(full_c_prediction == true)),
            "fpga_correct": int(np.count_nonzero(fpga == true)),
            "python_primary_vs_fpga_matches": int(np.count_nonzero(primary == fpga)),
        },
        "dbwfb2": {
            "accumulator_dtype": "signed INT32 (verified observed range; no overflow)",
            "row_accumulator_min": row_acc_min,
            "row_accumulator_max": row_acc_max,
            "column_accumulator_min": col_acc_min,
            "column_accumulator_max": col_acc_max,
            "combined_accumulator_min": dbw_acc_min,
            "combined_accumulator_max": dbw_acc_max,
            "right_shift": "arithmetic >>10 before signed-16-bit truncation",
            "verified_model_prediction_residuals_vs_fpga": 0,
        },
        "cnn_input": {
            "definition": "preclip = signed LL - 127 at gain 1; actual clipping is preclip outside [-128,127]",
            "elements": total_input,
            "actually_clipped_low": clipped_low,
            "actually_clipped_high": clipped_high,
            "actually_clipped_total": clipped_low + clipped_high,
            "actually_clipped_fraction": (clipped_low + clipped_high) / total_input,
            "output_equal_minus128": endpoint_low,
            "output_equal_plus127": endpoint_high,
            "endpoint_fraction": (endpoint_low + endpoint_high) / total_input,
        },
        "conv1": {
            "accumulator_dtype": "signed INT32",
            "shift": "arithmetic >>9",
            "accumulator_min": cnn_audit["conv1_acc_min"],
            "accumulator_max": cnn_audit["conv1_acc_max"],
            "shifted_min": cnn_audit["conv1_shift_min"],
            "shifted_max": cnn_audit["conv1_shift_max"],
            "elements": conv1_total,
            "relu_low_clipped_shifted_below_zero": cnn_audit["conv1_shift_below_zero"],
            "relu_low_clipped_fraction": cnn_audit["conv1_shift_below_zero"] / conv1_total,
            "high_clipped_shifted_above127": cnn_audit["conv1_shift_above_127"],
            "high_clipped_fraction": cnn_audit["conv1_shift_above_127"] / conv1_total,
            "output_equal_zero": cnn_audit["conv1_output_zero"],
            "output_equal_127": cnn_audit["conv1_output_127"],
        },
        "conv2": {
            "accumulator_dtype": "signed INT32",
            "shift": "arithmetic >>8",
            "accumulator_min": cnn_audit["conv2_acc_min"],
            "accumulator_max": cnn_audit["conv2_acc_max"],
            "shifted_min": cnn_audit["conv2_shift_min"],
            "shifted_max": cnn_audit["conv2_shift_max"],
            "elements": conv2_total,
            "relu_low_clipped_shifted_below_zero": cnn_audit["conv2_shift_below_zero"],
            "relu_low_clipped_fraction": cnn_audit["conv2_shift_below_zero"] / conv2_total,
            "high_clipped_shifted_above127": cnn_audit["conv2_shift_above_127"],
            "high_clipped_fraction": cnn_audit["conv2_shift_above_127"] / conv2_total,
            "output_equal_zero": cnn_audit["conv2_output_zero"],
            "output_equal_127": cnn_audit["conv2_output_127"],
        },
        "gap": {
            "input_shape_per_channel": [16, 16],
            "sum_dtype": "uint32_t",
            "sum_min_full3200": cnn_audit["gap_sum_min"],
            "sum_max_full3200": cnn_audit["gap_sum_max"],
            "shift": "unsigned logical >>8; input is nonnegative UINT8, so exact floor(sum/256)",
            "shifted_min_full3200": cnn_audit["gap_shift_min"],
            "shifted_max_full3200": cnn_audit["gap_shift_max"],
            "negative_sums_possible": False,
            "diagnostic10_channel_comparisons": 640,
            "diagnostic10_mismatches": 0,
        },
        "dense": {
            "logit_dtype": "signed INT32",
            "logit_min_full3200": cnn_audit["logit_min"],
            "logit_max_full3200": cnn_audit["logit_max"],
            "instrumented_prediction_mismatches_vs_released_cnn_accel": cnn_audit["released_prediction_mismatches"],
        },
        "board500": {
            model: summary_metrics(board_true, prediction)
            for model, prediction in board_models.items()
        },
        "full3200": {
            model: summary_metrics(true, prediction)
            for model, prediction in full_models.items()
        },
        "sources": [
            {"path": path.relative_to(REPOSITORY).as_posix(), "sha256": sha256(path)}
            for path in [
                FULL_VARIANTS,
                FULL_ANALYSIS,
                BOARD_LOG,
                BOARD_INDICES,
                HLS_LOG,
                CNN_CPP,
                CNN_H,
                WEIGHTS,
                TRACE_CPP,
            ]
        ],
    }
    (HERE / "operation_audit.json").write_text(
        json.dumps(operation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(operation, indent=2))


if __name__ == "__main__":
    main()
