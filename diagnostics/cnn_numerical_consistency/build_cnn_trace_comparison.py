"""Compare NumPy and diagnostic HLS intermediate CNN tensors exactly."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cnn_trace_python import CHECKPOINTS, load_weights, trace_cnn


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
FIRST = REPOSITORY / "diagnostics/first_divergence"
sys.path.insert(0, str(FIRST))

from run_full3200_variants import load_hls_packed_words  # noqa: E402

GENERATED = HERE / "generated"
INPUT_PATH = GENERATED / "diagnostic10_cnn_input.bin"
HLS_PREFIX = GENERATED / "diagnostic10_hls"
PRIMARY_INPUT_PATH = GENERATED / "diagnostic10_python_primary_cnn_input.bin"
PRIMARY_HLS_PREFIX = GENERATED / "diagnostic10_python_primary_hls"
TRACE_EXE = GENERATED / "cnn_intermediate_trace.exe"
TRACE_CPP = HERE / "cnn_intermediate_trace.cpp"
SELECTION_PATH = REPOSITORY / "diagnostics/first_divergence/generated/selected_diagnostic_indices.csv"
WEIGHTS_PATH = REPOSITORY / "hardware/hls/dbwfb2/weights.h"
CNN_CPP = REPOSITORY / "hardware/hls/src/cnn_accel.cpp"
CNN_HEADER = REPOSITORY / "hardware/hls/src/cnn_accel.h"
OBSERVED_EXPECTED = (
    FIRST
    / "generated/first_diagnosis_complete_comparison/observed_controller_expected.npz"
)
ORDER = [8, 107, 844, 1670, 2408, 0, 1, 813, 1603, 2400]


def load_hls_checkpoint(prefix: Path, name: str, count: int) -> np.ndarray:
    shape, dtype = CHECKPOINTS[name]
    suffix = name
    path = Path(str(prefix) + f"_{suffix}.bin")
    array = np.fromfile(path, dtype=dtype)
    expected = count * (int(np.prod(shape)) if shape else 1)
    if array.size != expected:
        raise ValueError(f"{path}: {array.size} values, expected {expected}")
    return array.reshape((count,) + shape)


def compare(expected: np.ndarray, actual: np.ndarray) -> dict:
    if expected.shape != actual.shape:
        raise ValueError(f"shape mismatch: {expected.shape} vs {actual.shape}")
    delta = actual.astype(np.int64) - expected.astype(np.int64)
    mismatch = np.argwhere(delta != 0)
    first = mismatch[0].tolist() if mismatch.size else None
    return {
        "shape": list(expected.shape),
        "python_dtype": str(expected.dtype),
        "hls_dtype": str(actual.dtype),
        "total_elements": int(expected.size),
        "exact_matches": int(np.count_nonzero(delta == 0)),
        "mismatches": int(np.count_nonzero(delta != 0)),
        "exact_match_fraction": float(np.mean(delta == 0)),
        "maximum_absolute_difference": int(np.abs(delta).max()) if delta.size else 0,
        "first_mismatch_coordinate": first,
        "python_at_first_mismatch": int(expected[tuple(first)]) if first is not None else None,
        "hls_at_first_mismatch": int(actual[tuple(first)]) if first is not None else None,
    }


def main() -> None:
    selection = {
        int(row["global_index"]): row
        for row in csv.DictReader(SELECTION_PATH.open(encoding="utf-8"))
    }
    if set(ORDER) != set(selection):
        raise ValueError("diagnostic selection does not match frozen ten-image order")
    observed = np.load(OBSERVED_EXPECTED)
    hardware_inputs = np.stack([observed[f"index_{index}_CNN_INPUT"] for index in ORDER])
    hardware_inputs.tofile(INPUT_PATH)
    packed = load_hls_packed_words()[ORDER]
    primary_inputs = np.ascontiguousarray(packed).view(np.uint8).reshape(10, 4096).view(np.int8).reshape(10, 64, 64)
    primary_inputs.tofile(PRIMARY_INPUT_PATH)
    subprocess.run(
        [
            "g++",
            "-O3",
            "-std=c++17",
            str(TRACE_CPP),
            str(CNN_CPP),
            f"-I{CNN_HEADER.parent}",
            f"-I{WEIGHTS_PATH.parent}",
            "-o",
            str(TRACE_EXE),
        ],
        cwd=REPOSITORY,
        check=True,
    )
    subprocess.run(
        [str(TRACE_EXE), str(INPUT_PATH), str(HLS_PREFIX)],
        cwd=REPOSITORY,
        check=True,
    )
    subprocess.run(
        [str(TRACE_EXE), str(PRIMARY_INPUT_PATH), str(PRIMARY_HLS_PREFIX)],
        cwd=REPOSITORY,
        check=True,
    )
    weights = load_weights(WEIGHTS_PATH)
    input_variants = {
        "frozen_python_primary": (primary_inputs, PRIMARY_HLS_PREFIX),
        "hardware_equivalent": (hardware_inputs, HLS_PREFIX),
    }

    per_image = []
    aggregate = []
    csv_rows = []
    traces_by_variant = {}
    hls_by_variant = {}
    for input_variant, (inputs, prefix) in input_variants.items():
        python_traces = [trace_cnn(inputs[i], weights) for i in range(10)]
        hls = {name: load_hls_checkpoint(prefix, name, 10) for name in CHECKPOINTS}
        traces_by_variant[input_variant] = python_traces
        hls_by_variant[input_variant] = hls
        for name in CHECKPOINTS:
            python_stack = np.stack([trace[name] for trace in python_traces])
            stage_aggregate = compare(python_stack, hls[name])
            stage_aggregate.update({"input_variant": input_variant, "checkpoint": name})
            aggregate.append(stage_aggregate)
            for position, index in enumerate(ORDER):
                result = compare(python_stack[position], hls[name][position])
                result.update(
                    {
                        "input_variant": input_variant,
                        "global_index": index,
                        "checkpoint": name,
                    }
                )
                per_image.append(result)
                csv_rows.append(
                    {
                        "input_variant": input_variant,
                        "global_index": index,
                        "checkpoint": name,
                        "shape": "x".join(map(str, result["shape"])) or "scalar",
                        "python_dtype": result["python_dtype"],
                        "hls_dtype": result["hls_dtype"],
                        "total_elements": result["total_elements"],
                        "exact_matches": result["exact_matches"],
                        "mismatches": result["mismatches"],
                        "exact_match_fraction": result["exact_match_fraction"],
                        "maximum_absolute_difference": result["maximum_absolute_difference"],
                        "first_mismatch_coordinate": result["first_mismatch_coordinate"],
                    }
                )

    if any(row["mismatches"] for row in aggregate):
        failures = [row["checkpoint"] for row in aggregate if row["mismatches"]]
        raise ValueError(f"Python/HLS intermediate mismatches: {failures}")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_definitions": {
            "frozen_python_primary": "released packed Python-primary CNN inputs",
            "hardware_equivalent": "verified full-FPGA-state CNN inputs",
        },
        "execution_order": ORDER,
        "instrumentation_prediction_mismatches_vs_released_cnn_accel": 0,
        "aggregate": aggregate,
        "per_image": per_image,
    }
    (HERE / "cnn_intermediate_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (HERE / "cnn_intermediate_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    with (HERE / "diagnostic_indices.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "global_index",
            "true_label",
            "python_primary_prediction",
            "fpga_prediction",
            "selection_group",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in ORDER:
            row = selection[index]
            writer.writerow(
                {
                    "global_index": index,
                    "true_label": row["true_label"],
                    "python_primary_prediction": row["python_prediction"],
                    "fpga_prediction": row["fpga_prediction"],
                    "selection_group": row["selection_group"],
                }
            )

    with (HERE / "gap_exactness.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "input_variant",
            "global_index",
            "channel",
            "python_pool2_sum",
            "python_gap_sum",
            "hls_gap_sum",
            "python_gap_shifted",
            "hls_gap_shifted",
            "sum_exact",
            "shift_exact",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for input_variant in input_variants:
            python_traces = traces_by_variant[input_variant]
            hls = hls_by_variant[input_variant]
            for position, index in enumerate(ORDER):
                pool2 = python_traces[position]["POOL2"]
                recomputed = pool2.astype(np.uint32).sum(axis=(1, 2), dtype=np.uint32)
                for channel in range(32):
                    py_sum = int(python_traces[position]["GAP_SUM"][channel])
                    hls_sum = int(hls["GAP_SUM"][position, channel])
                    py_shift = int(python_traces[position]["GAP_SHIFTED"][channel])
                    hls_shift = int(hls["GAP_SHIFTED"][position, channel])
                    writer.writerow(
                        {
                            "input_variant": input_variant,
                            "global_index": index,
                            "channel": channel,
                            "python_pool2_sum": int(recomputed[channel]),
                            "python_gap_sum": py_sum,
                            "hls_gap_sum": hls_sum,
                            "python_gap_shifted": py_shift,
                            "hls_gap_shifted": hls_shift,
                            "sum_exact": recomputed[channel] == py_sum == hls_sum,
                            "shift_exact": py_shift == hls_shift == (py_sum >> 8),
                        }
                    )

    print("Python/HLS checkpoints exact:", len(aggregate), "of", len(aggregate))
    print("GAP channel comparisons:", len(input_variants) * 10 * 32)


if __name__ == "__main__":
    main()
