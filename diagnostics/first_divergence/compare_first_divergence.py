"""Parse FPGA UART checkpoints and localize the first numerical divergence."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from fpga_intent_preprocess import boundary_masks


HERE = Path(__file__).resolve().parent
DEFAULT_EXPECTED = HERE / "generated" / "expected_first_divergence.npz"

IMAGE_BEGIN_RE = re.compile(
    r"^TRACE_IMAGE_BEGIN,index=(\d+),true=(\d+),python=(\d+)\s*$"
)
STAGE_BEGIN_RE = re.compile(
    r"^(RAW|ROW|LL|CNN_INPUT)_BEGIN,rows=(\d+),cols=(\d+),dtype=([A-Za-z0-9_]+)\s*$"
)
STAGE_ROW_RE = re.compile(r"^(RAW|ROW|LL|CNN_INPUT)_ROW,(\d+),(.*)$")
STAGE_END_RE = re.compile(r"^(RAW|ROW|LL|CNN_INPUT)_END\s*$")
RESULT_RE = re.compile(
    r"^TRACE_RESULT,index=(\d+),true=(\d+),python=(\d+),fpga=(\d+)\s*$"
)
IMAGE_END_RE = re.compile(r"^TRACE_IMAGE_END,index=(\d+)\s*$")


def parse_uart(path: Path, allow_leading_incomplete: bool = False) -> dict[int, dict]:
    images: dict[int, dict] = {}
    current: dict | None = None
    stage_name: str | None = None
    stage_rows: dict[int, list[int]] = {}
    stage_shape: tuple[int, int] | None = None

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        line = raw_line.strip()
        match = IMAGE_BEGIN_RE.match(line)
        if match:
            if current is not None:
                raise ValueError(f"line {line_number}: nested TRACE_IMAGE_BEGIN")
            index, true_label, python = map(int, match.groups())
            if index in images:
                raise ValueError(f"line {line_number}: duplicate image index {index}")
            current = {
                "global_index": index,
                "true_label": true_label,
                "python_prediction": python,
                "stages": {},
            }
            images[index] = current
            continue

        match = STAGE_BEGIN_RE.match(line)
        if match:
            if current is None or stage_name is not None:
                if allow_leading_incomplete and current is None:
                    continue
                raise ValueError(f"line {line_number}: invalid stage begin")
            stage_name = match.group(1)
            stage_shape = (int(match.group(2)), int(match.group(3)))
            stage_rows = {}
            continue

        match = STAGE_ROW_RE.match(line)
        if match:
            if current is None or stage_name != match.group(1) or stage_shape is None:
                if allow_leading_incomplete and current is None:
                    continue
                raise ValueError(f"line {line_number}: row outside matching stage")
            row_index = int(match.group(2))
            if row_index in stage_rows:
                raise ValueError(f"line {line_number}: duplicate {stage_name} row {row_index}")
            values = [int(value) for value in match.group(3).split(",") if value != ""]
            if len(values) != stage_shape[1]:
                raise ValueError(
                    f"line {line_number}: {stage_name} row has {len(values)} values; "
                    f"expected {stage_shape[1]}"
                )
            stage_rows[row_index] = values
            continue

        match = STAGE_END_RE.match(line)
        if match:
            if current is None or stage_name != match.group(1) or stage_shape is None:
                if allow_leading_incomplete and current is None:
                    continue
                raise ValueError(f"line {line_number}: unmatched stage end")
            expected_rows = set(range(stage_shape[0]))
            if set(stage_rows) != expected_rows:
                missing = sorted(expected_rows - set(stage_rows))
                raise ValueError(
                    f"line {line_number}: incomplete {stage_name}; missing rows {missing[:10]}"
                )
            current["stages"][stage_name] = np.asarray(
                [stage_rows[row] for row in range(stage_shape[0])], dtype=np.int64
            )
            stage_name = None
            stage_shape = None
            stage_rows = {}
            continue

        match = RESULT_RE.match(line)
        if match:
            if current is None:
                if allow_leading_incomplete:
                    continue
                raise ValueError(f"line {line_number}: TRACE_RESULT outside image")
            index, true_label, python, fpga = map(int, match.groups())
            if (
                index != current["global_index"]
                or true_label != current["true_label"]
                or python != current["python_prediction"]
            ):
                raise ValueError(f"line {line_number}: TRACE_RESULT metadata conflict")
            current["fpga_prediction"] = fpga
            continue

        match = IMAGE_END_RE.match(line)
        if match:
            if current is None or int(match.group(1)) != current["global_index"]:
                if allow_leading_incomplete and current is None:
                    continue
                raise ValueError(f"line {line_number}: unmatched TRACE_IMAGE_END")
            if stage_name is not None:
                raise ValueError(f"line {line_number}: image ended inside {stage_name}")
            current = None

    if current is not None or stage_name is not None:
        raise ValueError("UART trace ended inside an image or stage")
    return images


def comparison(expected: np.ndarray, actual: np.ndarray) -> tuple[dict, np.ndarray]:
    expected_i = np.asarray(expected, dtype=np.int64)
    actual_i = np.asarray(actual, dtype=np.int64)
    if expected_i.shape != actual_i.shape:
        return (
            {
                "shape_expected": list(expected_i.shape),
                "shape_actual": list(actual_i.shape),
                "total_elements": int(expected_i.size),
                "exact_matches": 0,
                "mismatch_count": -1,
                "match_percent": 0.0,
                "first_mismatch": "SHAPE_MISMATCH",
                "expected_at_first": None,
                "fpga_at_first": None,
                "max_absolute_error": None,
                "mean_absolute_error": None,
                "plus_one_differences": None,
                "minus_one_differences": None,
            },
            np.ones(expected_i.shape, dtype=bool),
        )
    delta = actual_i - expected_i
    mismatch = delta != 0
    count = int(np.count_nonzero(mismatch))
    if count:
        coordinate = tuple(int(x) for x in np.argwhere(mismatch)[0])
        expected_first = int(expected_i[coordinate])
        actual_first = int(actual_i[coordinate])
        first = list(coordinate)
    else:
        first = None
        expected_first = None
        actual_first = None
    return (
        {
            "shape_expected": list(expected_i.shape),
            "shape_actual": list(actual_i.shape),
            "total_elements": int(expected_i.size),
            "exact_matches": int(expected_i.size - count),
            "mismatch_count": count,
            "match_percent": 100.0 * (expected_i.size - count) / expected_i.size,
            "first_mismatch": first,
            "expected_at_first": expected_first,
            "fpga_at_first": actual_first,
            "max_absolute_error": int(np.max(np.abs(delta))) if delta.size else 0,
            "mean_absolute_error": float(np.mean(np.abs(delta))) if delta.size else 0.0,
            "plus_one_differences": int(np.count_nonzero(delta == 1)),
            "minus_one_differences": int(np.count_nonzero(delta == -1)),
        },
        mismatch,
    )


def boundary_counts(mask: np.ndarray) -> dict[str, int]:
    masks = boundary_masks(tuple(mask.shape))
    return {name: int(np.count_nonzero(mask & region)) for name, region in masks.items()}


def classify_first_divergence(stage_results: dict[str, dict], prediction_differs: bool) -> str:
    for stage, meaning in (
        ("RAW", "RAW differs: input/header/write-path issue"),
        ("ROW", "RAW matches, ROW differs: row DBWFB2/boundary/integer-arithmetic path"),
        ("LL", "ROW matches, LL differs: transpose/reorder/column DBWFB2/column-boundary path"),
        ("CNN_INPUT", "LL matches, CNN_INPUT differs: PS centering/clipping/packing path"),
    ):
        result = stage_results.get(stage)
        if result is None:
            return f"INCOMPLETE TRACE: {stage} checkpoint missing"
        if result["mismatch_count"]:
            return meaning
    if prediction_differs:
        return "CNN_INPUT matches but prediction differs: CNN hardware path requires a separate experiment"
    return "No divergence in dumped checkpoints or final prediction"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("uart_trace", type=Path)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="recover complete traces from an explicitly incomplete capture",
    )
    args = parser.parse_args()

    uart_path = args.uart_trace.resolve()
    expected_path = args.expected.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else uart_path.parent / f"{uart_path.stem}_comparison"
    )
    output.mkdir(parents=True, exist_ok=True)

    trace = parse_uart(uart_path, allow_leading_incomplete=args.allow_subset)
    archive = np.load(expected_path)
    expected_indices = archive["global_indices"].astype(int).tolist()
    if set(trace) != set(expected_indices) and not args.allow_subset:
        raise ValueError(
            f"trace indices {sorted(trace)} differ from expected indices {sorted(expected_indices)}"
        )
    if args.allow_subset:
        unexpected = set(trace) - set(expected_indices)
        if unexpected:
            raise ValueError(f"trace contains unexpected indices: {sorted(unexpected)}")
        missing = sorted(set(expected_indices) - set(trace))
        print(
            "WARNING: incomplete capture recovery mode; "
            f"complete indices={sorted(trace)}, missing indices={missing}"
        )
        expected_indices = [index for index in expected_indices if index in trace]
    position = {index: offset for offset, index in enumerate(expected_indices)}
    stage_keys = {"RAW": "raw", "ROW": "row", "LL": "ll", "CNN_INPUT": "cnn_input"}

    aggregate: dict = {
        "uart_trace": str(uart_path),
        "expected_archive": str(expected_path),
        "images": [],
    }
    flat_stage_rows: list[dict] = []
    mismatch_locations: list[dict] = []

    for index in expected_indices:
        offset = position[index]
        image_trace = trace[index]
        if image_trace["true_label"] != int(archive["true_labels"][offset]):
            raise ValueError(f"true-label conflict at index {index}")
        if image_trace["python_prediction"] != int(archive["python_predictions"][offset]):
            raise ValueError(f"Python-prediction conflict at index {index}")

        image_result: dict = {
            "global_index": index,
            "true_label": image_trace["true_label"],
            "python_prediction": image_trace["python_prediction"],
            "original_logged_fpga_prediction": int(archive["logged_fpga_predictions"][offset]),
            "diagnostic_fpga_prediction": image_trace.get("fpga_prediction"),
            "stages": {},
        }
        stage_mismatch_masks: dict[str, np.ndarray] = {}

        for stage, key in stage_keys.items():
            actual = image_trace["stages"].get(stage)
            if actual is None:
                continue
            result, mismatch = comparison(archive[key][offset], actual)
            if stage in ("ROW", "LL", "CNN_INPUT") and result["mismatch_count"] >= 0:
                result["boundary_mismatch_counts"] = boundary_counts(mismatch)
            image_result["stages"][stage] = result
            stage_mismatch_masks[stage] = mismatch
            flat_stage_rows.append(
                {
                    "global_index": index,
                    "stage": stage,
                    "exact_matches": result["exact_matches"],
                    "mismatches": result["mismatch_count"],
                    "match_percent": result["match_percent"],
                    "max_absolute_error": result["max_absolute_error"],
                    "mean_absolute_error": result["mean_absolute_error"],
                    "first_mismatch": result["first_mismatch"],
                    "expected_at_first": result["expected_at_first"],
                    "fpga_at_first": result["fpga_at_first"],
                }
            )
            if result["mismatch_count"] > 0:
                expected_values = np.asarray(archive[key][offset], dtype=np.int64)
                actual_values = np.asarray(actual, dtype=np.int64)
                for coordinate in np.argwhere(mismatch):
                    coord = tuple(int(value) for value in coordinate)
                    mismatch_locations.append(
                        {
                            "global_index": index,
                            "stage": stage,
                            "row": coord[0],
                            "column": coord[1],
                            "python_value": int(expected_values[coord]),
                            "fpga_value": int(actual_values[coord]),
                            "difference_fpga_minus_python": int(actual_values[coord] - expected_values[coord]),
                        }
                    )

        expected_preclip = archive["preclip"][offset].astype(np.int64)
        actual_ll = image_trace["stages"].get("LL")
        clipping: dict = {
            "python_preclip_min": int(expected_preclip.min()),
            "python_preclip_max": int(expected_preclip.max()),
            "python_low_saturation_count": int(np.count_nonzero(expected_preclip < -128)),
            "python_high_saturation_count": int(np.count_nonzero(expected_preclip > 127)),
        }
        if actual_ll is not None:
            actual_preclip = actual_ll.astype(np.int64) - 127
            clipping.update(
                {
                    "fpga_preclip_min": int(actual_preclip.min()),
                    "fpga_preclip_max": int(actual_preclip.max()),
                    "fpga_low_saturation_count": int(np.count_nonzero(actual_preclip < -128)),
                    "fpga_high_saturation_count": int(np.count_nonzero(actual_preclip > 127)),
                }
            )
            ll_diff = actual_ll.astype(np.int64) != archive["ll"][offset].astype(np.int64)
            clipped_equal = np.clip(actual_preclip, -128, 127) == np.clip(
                expected_preclip, -128, 127
            )
            clipping["ll_mismatches_hidden_by_clipping"] = int(
                np.count_nonzero(ll_diff & clipped_equal)
            )
            cnn_actual = image_trace["stages"].get("CNN_INPUT")
            if cnn_actual is not None:
                cnn_diff = cnn_actual.astype(np.int64) != archive["cnn_input"][offset].astype(np.int64)
                either_saturated = (
                    (actual_preclip < -128)
                    | (actual_preclip > 127)
                    | (expected_preclip < -128)
                    | (expected_preclip > 127)
                )
                clipping["cnn_input_mismatches_with_either_side_saturated"] = int(
                    np.count_nonzero(cnn_diff & either_saturated)
                )
        image_result["clipping"] = clipping

        prediction = image_trace.get("fpga_prediction")
        prediction_differs = prediction is None or prediction != image_trace["python_prediction"]
        image_result["first_divergence"] = classify_first_divergence(
            image_result["stages"], prediction_differs
        )
        aggregate["images"].append(image_result)

        print(f"\nIMAGE {index}: {image_result['first_divergence']}")
        print("Stage       Exact matches  Mismatches  Max abs error  First mismatch")
        for stage in stage_keys:
            result = image_result["stages"].get(stage)
            if result is None:
                print(f"{stage:<12} MISSING")
            else:
                print(
                    f"{stage:<12} {result['exact_matches']:>13}  "
                    f"{result['mismatch_count']:>10}  {result['max_absolute_error']:>13}  "
                    f"{result['first_mismatch']}"
                )

    # Aggregate each stage without obscuring per-image first-divergence results.
    aggregate["stage_totals"] = {}
    for stage in stage_keys:
        rows = [row for row in flat_stage_rows if row["stage"] == stage]
        aggregate["stage_totals"][stage] = {
            "images_present": len(rows),
            "exact_matches": int(sum(row["exact_matches"] for row in rows)),
            "mismatches": int(sum(row["mismatches"] for row in rows)),
        }

    with (output / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
        handle.write("\n")

    with (output / "comparison_stage_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_stage_rows[0]))
        writer.writeheader()
        writer.writerows(flat_stage_rows)

    location_fields = [
        "global_index",
        "stage",
        "row",
        "column",
        "python_value",
        "fpga_value",
        "difference_fpga_minus_python",
    ]
    with (output / "mismatch_locations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=location_fields)
        writer.writeheader()
        writer.writerows(mismatch_locations)

    print(f"\nWrote comparison outputs to {output}")


if __name__ == "__main__":
    main()
