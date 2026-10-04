"""Run the frozen CNN on three full-3,200 preprocessing variants.

The script compiles the released HLS CNN C++ together with the small binary
runner in this diagnostic directory.  It first requires Variant A to reproduce
both the released packed HLS input header and all 3,200 frozen Python/HLS
predictions.  Only then are Variants B and C evaluated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fpga_intent_preprocess import DBWFB2_COEFFICIENTS
from generate_first_divergence_vectors import (
    RAW_HEADER_RELATIVE,
    REPOSITORY,
    audit_headers,
    audit_uart_logs,
    extract_selected_raw,
    parse_c_uint8_array,
    sha256,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "generated" / "full3200_variant_analysis"
HLS_PACKED = REPOSITORY / "hardware/hls/dbwfb2/test_vectors_3200_cnn_only.h"
HLS_EXPECTED = REPOSITORY / "hardware/hls/dbwfb2/expected_int8_3200.h"
CNN_CPP = REPOSITORY / "hardware/hls/src/cnn_accel.cpp"
CNN_HEADER = REPOSITORY / "hardware/hls/src/cnn_accel.h"
CNN_WEIGHTS = REPOSITORY / "hardware/hls/dbwfb2/weights.h"
RUNNER_SOURCE = HERE / "cnn_binary_runner.cpp"
RUNNER_EXE = OUTPUT / "cnn_binary_runner.exe"

IMAGE_COUNT = 3200
RAW_SIZE = 128
LL_SIZE = 64
SHIFT = 10


def as_i16(values: np.ndarray) -> np.ndarray:
    unsigned = np.asarray(values, dtype=np.int64) & 0xFFFF
    return np.where(unsigned >= 0x8000, unsigned - 0x10000, unsigned).astype(
        np.int16
    )


def filtered_decimated_lines(
    lines: np.ndarray, leading_values: np.ndarray, centered: bool
) -> np.ndarray:
    """Filter arrays whose final axis contains one 128-sample transaction."""

    source = np.asarray(lines, dtype=np.int64)
    if source.shape[-1] != RAW_SIZE:
        raise ValueError(f"expected line length 128, got {source.shape}")
    if centered:
        leading = np.repeat(
            np.asarray(leading_values, dtype=np.int64)[..., None], 4, axis=-1
        )
        trailing = np.repeat(source[..., -1:], 4, axis=-1)
        extended = np.concatenate((leading, source, trailing), axis=-1)
        accumulator = np.zeros(source.shape[:-1] + (LL_SIZE,), dtype=np.int64)
        for tap, coefficient in enumerate(DBWFB2_COEFFICIENTS.tolist()):
            accumulator += int(coefficient) * extended[..., tap : tap + 128 : 2]
    else:
        trailing = np.repeat(source[..., -1:], 8, axis=-1)
        extended = np.concatenate((source, trailing), axis=-1)
        accumulator = np.zeros(source.shape, dtype=np.int64)
        for tap, coefficient in enumerate(DBWFB2_COEFFICIENTS.tolist()):
            accumulator += int(coefficient) * extended[..., tap : tap + 128]
        accumulator = accumulator[..., 0::2]
    if accumulator.min() < -(1 << 31) or accumulator.max() > (1 << 31) - 1:
        raise OverflowError("FIR accumulator exceeded signed 32-bit range")
    return as_i16(np.right_shift(accumulator, SHIFT))


def primary_image(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row = filtered_decimated_lines(raw, np.zeros(128), centered=False)
    ll = filtered_decimated_lines(row.T, np.zeros(64), centered=False).T
    return row, ll


def alignment_only_image(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centered phase with independent constant extension per transaction."""

    row = filtered_decimated_lines(raw, raw[:, 0], centered=True)
    column_lines = row.T
    ll = filtered_decimated_lines(
        column_lines, column_lines[:, 0], centered=True
    ).T
    return row, ll


def full_state_image(
    raw: np.ndarray, transaction_last: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Exact controller behavior verified by the ten-image FPGA trace."""

    row_leading = np.empty(128, dtype=np.int64)
    row_leading[0] = int(transaction_last)
    row_leading[1:] = raw[:-1, -1]
    row = filtered_decimated_lines(raw, row_leading, centered=True)

    column_lines = row.T
    column_leading = np.empty(64, dtype=np.int64)
    column_leading[0] = int(raw[-1, -1])
    column_leading[1:] = column_lines[:-1, -1]
    ll = filtered_decimated_lines(column_lines, column_leading, centered=True).T
    return row, ll, int(column_lines[-1, -1])


def quantize(ll: np.ndarray) -> np.ndarray:
    return np.clip(ll.astype(np.int32) - 127, -128, 127).astype(np.int8)


def load_raw_images() -> np.ndarray:
    images = np.empty((IMAGE_COUNT, RAW_SIZE, RAW_SIZE), dtype=np.uint8)
    for part, relative in enumerate(RAW_HEADER_RELATIVE):
        start = 800 * part
        wanted = set(range(start, start + 800))
        found = extract_selected_raw(REPOSITORY / relative, wanted)
        if set(found) != wanted:
            missing = sorted(wanted - set(found))
            raise ValueError(f"{relative}: missing raw indices {missing[:10]}")
        for index in range(start, start + 800):
            images[index] = found[index]
    return images


def load_hls_packed_words() -> np.ndarray:
    words = np.empty((IMAGE_COUNT, 1024), dtype=np.uint32)
    count = 0
    in_array = False
    with HLS_PACKED.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if not in_array:
                if "const uint32_t test_images" in line:
                    in_array = True
                continue
            for token in re.findall(r"0x[0-9A-Fa-f]+", line):
                if count >= words.size:
                    raise ValueError("packed HLS header contains excess words")
                words.reshape(-1)[count] = int(token, 16)
                count += 1
    if count != words.size:
        raise ValueError(f"packed HLS header contains {count} words, expected {words.size}")
    return words


def int8_to_words(inputs: np.ndarray) -> np.ndarray:
    bytes_u8 = np.ascontiguousarray(inputs).view(np.uint8).reshape(IMAGE_COUNT, 1024, 4)
    return (
        bytes_u8[:, :, 0].astype(np.uint32)
        | (bytes_u8[:, :, 1].astype(np.uint32) << 8)
        | (bytes_u8[:, :, 2].astype(np.uint32) << 16)
        | (bytes_u8[:, :, 3].astype(np.uint32) << 24)
    )


def compile_runner() -> None:
    command = [
        "g++",
        "-O3",
        "-std=c++17",
        str(RUNNER_SOURCE),
        str(CNN_CPP),
        f"-I{CNN_HEADER.parent}",
        f"-I{CNN_WEIGHTS.parent}",
        "-o",
        str(RUNNER_EXE),
    ]
    subprocess.run(command, cwd=REPOSITORY, check=True)


def run_cnn(name: str, inputs: np.ndarray) -> np.ndarray:
    input_path = OUTPUT / f"variant_{name}_cnn_input.bin"
    prediction_path = OUTPUT / f"variant_{name}_predictions.bin"
    np.ascontiguousarray(inputs).tofile(input_path)
    subprocess.run([str(RUNNER_EXE), str(input_path), str(prediction_path)], check=True)
    predictions = np.fromfile(prediction_path, dtype=np.uint8)
    if predictions.shape != (IMAGE_COUNT,):
        raise ValueError(f"runner produced {predictions.shape} for Variant {name}")
    return predictions


def metric_row(
    name: str,
    prediction: np.ndarray,
    true: np.ndarray,
    fpga: np.ndarray,
    primary: np.ndarray,
) -> dict:
    return {
        "variant": name,
        "correct": int(np.count_nonzero(prediction == true)),
        "accuracy_percent": float(100 * np.mean(prediction == true)),
        "fpga_matches": int(np.count_nonzero(prediction == fpga)),
        "fpga_agreement_percent": float(100 * np.mean(prediction == fpga)),
        "python_primary_matches": int(np.count_nonzero(prediction == primary)),
        "python_primary_agreement_percent": float(100 * np.mean(prediction == primary)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    uart_records, uart_parts = audit_uart_logs()
    header_index, header_parts = audit_headers(uart_records)
    raw = load_raw_images()

    true = np.asarray([header_index[i]["true_label"] for i in range(IMAGE_COUNT)], dtype=np.uint8)
    primary_predictions = np.asarray(
        [header_index[i]["python_prediction"] for i in range(IMAGE_COUNT)], dtype=np.uint8
    )
    fpga_predictions = np.asarray(
        [uart_records[i]["fpga_prediction"] for i in range(IMAGE_COUNT)], dtype=np.uint8
    )

    variants = {
        "A": np.empty((IMAGE_COUNT, LL_SIZE, LL_SIZE), dtype=np.int8),
        "B": np.empty((IMAGE_COUNT, LL_SIZE, LL_SIZE), dtype=np.int8),
        "C": np.empty((IMAGE_COUNT, LL_SIZE, LL_SIZE), dtype=np.int8),
    }
    row_changed = 0
    row_abs_sum = 0
    row_max = 0
    row_coord_mask = np.zeros((128, 64), dtype=bool)
    row_images_changed = 0
    ll_changed = 0
    ll_abs_sum = 0
    ll_max = 0
    ll_coord_mask = np.zeros((64, 64), dtype=bool)
    ll_images_changed = 0
    ll_region_changed = {
        "leading_rows_excluding_corner": 0,
        "leading_columns_excluding_corner": 0,
        "leading_corner": 0,
        "interior": 0,
    }

    # Each released 800-image part was a separate application download/run.
    # The ten-image trace verified zero as the initial post-reset state.
    state = 0
    for index in range(IMAGE_COUNT):
        if index in (0, 800, 1600, 2400):
            state = 0
        row_a, ll_a = primary_image(raw[index])
        row_b, ll_b = alignment_only_image(raw[index])
        row_c, ll_c, state = full_state_image(raw[index], state)
        variants["A"][index] = quantize(ll_a)
        variants["B"][index] = quantize(ll_b)
        variants["C"][index] = quantize(ll_c)

        row_delta = row_c.astype(np.int32) - row_b.astype(np.int32)
        row_nonzero = row_delta != 0
        row_count = int(np.count_nonzero(row_nonzero))
        row_changed += row_count
        row_images_changed += int(row_count > 0)
        row_abs_sum += int(np.abs(row_delta).sum())
        row_max = max(row_max, int(np.abs(row_delta).max()))
        row_coord_mask |= row_nonzero
        if np.count_nonzero(row_nonzero[:, 2:]) != 0:
            raise AssertionError("B/C ROW difference escaped leading columns 0 and 1")

        ll_delta = ll_c.astype(np.int32) - ll_b.astype(np.int32)
        ll_nonzero = ll_delta != 0
        ll_count = int(np.count_nonzero(ll_nonzero))
        ll_changed += ll_count
        ll_images_changed += int(ll_count > 0)
        ll_abs_sum += int(np.abs(ll_delta).sum())
        ll_max = max(ll_max, int(np.abs(ll_delta).max()))
        ll_coord_mask |= ll_nonzero
        ll_region_changed["leading_corner"] += int(np.count_nonzero(ll_nonzero[:2, :2]))
        ll_region_changed["leading_rows_excluding_corner"] += int(
            np.count_nonzero(ll_nonzero[:2, 2:])
        )
        ll_region_changed["leading_columns_excluding_corner"] += int(
            np.count_nonzero(ll_nonzero[2:, :2])
        )
        ll_region_changed["interior"] += int(np.count_nonzero(ll_nonzero[2:, 2:]))

    # Variant A must reproduce the released 3,200 HLS input tensors exactly.
    packed_a = int8_to_words(variants["A"])
    released_packed = load_hls_packed_words()
    packed_exact = int(np.count_nonzero(packed_a == released_packed))
    if packed_exact != packed_a.size:
        where = np.argwhere(packed_a != released_packed)[0]
        raise ValueError(f"Variant A differs from released packed HLS input at {where.tolist()}")

    expected_text = HLS_EXPECTED.read_text(encoding="utf-8", errors="strict")
    expected_hls = parse_c_uint8_array(expected_text, "expected_int8_labels")
    compile_runner()
    predictions = {name: run_cnn(name, tensor) for name, tensor in variants.items()}
    if not np.array_equal(predictions["A"], expected_hls):
        remaining = np.flatnonzero(predictions["A"] != expected_hls)
        raise ValueError(f"frozen CNN validation failed at {remaining[:20].tolist()}")
    if not np.array_equal(predictions["A"], primary_predictions):
        raise ValueError("Variant A CNN predictions differ from raw-header primary predictions")

    metrics = [
        metric_row(name, predictions[name], true, fpga_predictions, primary_predictions)
        for name in ("A", "B", "C")
    ]
    pairwise = []
    for left, right in (("A", "B"), ("B", "C"), ("A", "C")):
        matches = int(np.count_nonzero(predictions[left] == predictions[right]))
        pairwise.append(
            {
                "pair": f"{left}_vs_{right}",
                "matches": matches,
                "differences": IMAGE_COUNT - matches,
                "agreement_percent": 100.0 * matches / IMAGE_COUNT,
            }
        )

    c_residual = np.flatnonzero(predictions["C"] != fpga_predictions).tolist()
    row_total = IMAGE_COUNT * 128 * 64
    ll_total = IMAGE_COUNT * 64 * 64
    row_coords = np.argwhere(row_coord_mask)
    ll_coords = np.argwhere(ll_coord_mask)
    tensor_effect = {
        "B_vs_C_ROW": {
            "changed_elements": row_changed,
            "total_elements": row_total,
            "changed_fraction": row_changed / row_total,
            "mean_absolute_difference_all_elements": row_abs_sum / row_total,
            "mean_absolute_difference_changed_elements": row_abs_sum / row_changed,
            "maximum_absolute_difference": row_max,
            "images_with_any_change": row_images_changed,
            "structurally_exposed_elements_columns_0_1": IMAGE_COUNT * 128 * 2,
            "structurally_exposed_fraction": (IMAGE_COUNT * 128 * 2) / row_total,
            "affected_row_indices": sorted(set(row_coords[:, 0].tolist())) if row_coords.size else [],
            "affected_column_indices": sorted(set(row_coords[:, 1].tolist())) if row_coords.size else [],
        },
        "B_vs_C_LL": {
            "changed_elements": ll_changed,
            "total_elements": ll_total,
            "changed_fraction": ll_changed / ll_total,
            "mean_absolute_difference_all_elements": ll_abs_sum / ll_total,
            "mean_absolute_difference_changed_elements": ll_abs_sum / ll_changed,
            "maximum_absolute_difference": ll_max,
            "images_with_any_change": ll_images_changed,
            "structurally_exposed_elements_border_union": IMAGE_COUNT * (2 * 64 + 62 * 2),
            "structurally_exposed_fraction": (IMAGE_COUNT * (2 * 64 + 62 * 2)) / ll_total,
            "changed_by_region": ll_region_changed,
            "affected_coordinate_rule": "row in {0,1} or column in {0,1}",
            "affected_row_indices": sorted(set(ll_coords[:, 0].tolist())) if ll_coords.size else [],
            "affected_column_indices": sorted(set(ll_coords[:, 1].tolist())) if ll_coords.size else [],
        },
    }
    if ll_region_changed["interior"] != 0:
        raise AssertionError("B/C LL difference escaped the leading two-row/two-column union")

    source_paths = [
        *[REPOSITORY / p for p in RAW_HEADER_RELATIVE],
        *[REPOSITORY / Path(part["source_log"]) for part in uart_parts],
        HLS_PACKED,
        HLS_EXPECTED,
        CNN_CPP,
        CNN_HEADER,
        CNN_WEIGHTS,
        RUNNER_SOURCE,
    ]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definitions": {
            "A": "frozen Python-primary forward complete-window phase with trailing replicate",
            "B": "centered even phase; each transaction independently uses constant extension from its own first and last samples",
            "C": "verified centered even phase with preceding-transaction state at the leading edge and current-tail replication",
        },
        "part_state_resets": [0, 800, 1600, 2400],
        "frozen_cnn_validation": {
            "variant_A_packed_words_exact": packed_exact,
            "variant_A_packed_words_total": int(packed_a.size),
            "variant_A_prediction_matches_expected": int(
                np.count_nonzero(predictions["A"] == expected_hls)
            ),
            "variant_A_prediction_total": IMAGE_COUNT,
        },
        "metrics": metrics,
        "pairwise_predictions": pairwise,
        "variant_C_fpga_residual_indices": c_residual,
        "tensor_effect": tensor_effect,
        "sources": [
            {
                "path": path.relative_to(REPOSITORY).as_posix(),
                "sha256": sha256(path),
            }
            for path in source_paths
        ],
    }
    (OUTPUT / "full3200_variant_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    with (OUTPUT / "full3200_variant_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "global_index",
                "true_label",
                "fpga_prediction",
                "frozen_python_primary_prediction",
                "variant_A_prediction",
                "variant_B_prediction",
                "variant_C_prediction",
            ]
        )
        for index in range(IMAGE_COUNT):
            writer.writerow(
                [
                    index,
                    int(true[index]),
                    int(fpga_predictions[index]),
                    int(primary_predictions[index]),
                    int(predictions["A"][index]),
                    int(predictions["B"][index]),
                    int(predictions["C"][index]),
                ]
            )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
