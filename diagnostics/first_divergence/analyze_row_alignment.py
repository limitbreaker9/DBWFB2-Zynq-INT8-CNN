"""Explore full-rate phase alignment after a verified RAW checkpoint match."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from compare_first_divergence import parse_uart
from fpga_intent_preprocess import (
    DBWFB2_COEFFICIENTS,
    fir_fullrate_trailing_replicate,
    preprocess_observed_controller_sequence,
)


def leading_centered_outputs(raw: np.ndarray, left_values: np.ndarray) -> np.ndarray:
    """Centered outputs n=0,2 using four repeated leading values per row."""

    source = raw.astype(np.int64)
    left = np.repeat(np.asarray(left_values, dtype=np.int64)[:, None], 4, axis=1)
    right = np.repeat(source[:, -1:], 4, axis=1)
    extended = np.concatenate((left, source, right), axis=1)
    outputs = []
    for start in (0, 2):
        accumulator = np.zeros(source.shape[0], dtype=np.int64)
        for tap, coefficient in enumerate(DBWFB2_COEFFICIENTS.tolist()):
            accumulator += int(coefficient) * extended[:, start + tap]
        outputs.append(np.right_shift(accumulator, 10).astype(np.int16))
    return np.stack(outputs, axis=1).astype(np.int64)


def centered_decimated_line(line: np.ndarray, leading_value: int) -> np.ndarray:
    """Centered even-phase output with controller state at the leading edge."""

    source = np.asarray(line, dtype=np.int64)
    extended = np.concatenate(
        (
            np.full(4, int(leading_value), dtype=np.int64),
            source,
            np.full(4, int(source[-1]), dtype=np.int64),
        )
    )
    output = np.empty(64, dtype=np.int16)
    for output_index, center in enumerate(range(0, 128, 2)):
        window = extended[center : center + 9]
        accumulator = int(np.sum(window * DBWFB2_COEFFICIENTS))
        output[output_index] = np.int16(accumulator >> 10)
    return output


def controller_sequence_image(
    raw: np.ndarray, transaction_last: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Model the observed state carried between successive 1-D transactions."""

    row = np.empty((128, 64), dtype=np.int16)
    state = int(transaction_last)
    for row_index in range(128):
        line = raw[row_index]
        row[row_index] = centered_decimated_line(line, state)
        state = int(line[-1])

    ll = np.empty((64, 64), dtype=np.int16)
    for column_index in range(64):
        line = row[:, column_index]
        ll[:, column_index] = centered_decimated_line(line, state)
        state = int(line[-1])
    return row, ll, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("uart_trace", type=Path)
    parser.add_argument("--min-offset", type=int, default=-16)
    parser.add_argument("--max-offset", type=int, default=16)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    traces = parse_uart(args.uart_trace.resolve())
    print("offset means: FPGA ROW[:,c] compared with Python full-rate FIR[:,2*c+offset]")

    totals: dict[int, dict[str, float]] = {}
    for offset in range(args.min_offset, args.max_offset + 1):
        totals[offset] = {"elements": 0, "exact": 0, "abs_error": 0.0, "max": 0.0}

    # Preserve execution order rather than sorting by global index; controller
    # state carries from the last transaction of one image into the next.
    execution_order = list(traces)
    transaction_last = 0
    sequence_totals = {"row_exact": 0, "row_elements": 0, "ll_exact": 0, "ll_elements": 0}
    per_image: list[dict] = []
    raw_sequence = [traces[index]["stages"]["RAW"].astype(np.uint8) for index in execution_order]
    observed_models = preprocess_observed_controller_sequence(raw_sequence)
    model_archive: dict[str, np.ndarray] = {}

    for sequence_position, index in enumerate(execution_order):
        trace = traces[index]
        raw = trace["stages"]["RAW"].astype(np.uint8)
        actual = trace["stages"]["ROW"].astype(np.int64)
        fullrate = fir_fullrate_trailing_replicate(raw, axis=1).astype(np.int64)
        image_scores: list[tuple[float, int, int, int]] = []

        for offset in range(args.min_offset, args.max_offset + 1):
            columns = np.arange(64)
            positions = 2 * columns + offset
            valid = (positions >= 0) & (positions < 128)
            expected = fullrate[:, positions[valid]]
            observed = actual[:, valid]
            delta = observed - expected
            elements = int(delta.size)
            exact = int(np.count_nonzero(delta == 0))
            absolute = float(np.abs(delta).sum())
            maximum = int(np.abs(delta).max()) if elements else 0
            mae = absolute / elements if elements else float("inf")
            image_scores.append((mae, -exact, offset, maximum))
            totals[offset]["elements"] += elements
            totals[offset]["exact"] += exact
            totals[offset]["abs_error"] += absolute
            totals[offset]["max"] = max(totals[offset]["max"], maximum)

        best = min(image_scores)
        print(
            f"index={index}: best_offset={best[2]:+d}, "
            f"MAE={best[0]:.6f}, exact={-best[1]}, max_abs={best[3]}"
        )

        # Isolate the second pass by using the captured ROW tensor itself as
        # the column-filter input. This avoids attributing the row-stage phase
        # difference to the column arithmetic.
        actual_ll = trace["stages"]["LL"].astype(np.int64)
        column_fullrate = fir_fullrate_trailing_replicate(actual, axis=0).astype(np.int64)
        column_scores: list[tuple[float, int, int, int]] = []
        for offset in range(args.min_offset, args.max_offset + 1):
            rows = np.arange(64)
            positions = 2 * rows + offset
            valid = (positions >= 0) & (positions < 128)
            expected_ll = column_fullrate[positions[valid], :]
            observed_ll = actual_ll[valid, :]
            delta = observed_ll - expected_ll
            elements = int(delta.size)
            exact = int(np.count_nonzero(delta == 0))
            mae = float(np.abs(delta).mean()) if elements else float("inf")
            maximum = int(np.abs(delta).max()) if elements else 0
            column_scores.append((mae, -exact, offset, maximum))
        column_best = min(column_scores)

        actual_cnn = trace["stages"]["CNN_INPUT"].astype(np.int64)
        packed_from_actual_ll = np.clip(actual_ll - 127, -128, 127)
        pack_delta = actual_cnn - packed_from_actual_ll
        print(
            f"           column best_offset={column_best[2]:+d}, "
            f"MAE={column_best[0]:.6f}, exact={-column_best[1]}, "
            f"max_abs={column_best[3]}; "
            f"PS pack exact={int(np.count_nonzero(pack_delta == 0))}/4096, "
            f"max_abs={int(np.abs(pack_delta).max())}"
        )

        # The -4 phase requires two leading centered outputs whose windows
        # extend before raw column zero. Compare plausible controller states.
        leading_actual = actual[:, :2]
        leading_candidates = {
            "zero": np.zeros(128, dtype=np.int64),
            "first_pixel": raw[:, 0].astype(np.int64),
            "previous_row_last": np.concatenate(
                (np.asarray([raw[0, 0]], dtype=np.int64), raw[:-1, -1].astype(np.int64))
            ),
        }
        leading_text = []
        for name, left_values in leading_candidates.items():
            candidate = leading_centered_outputs(raw, left_values)
            # Row zero has unknown state from the transaction before this image;
            # exclude it for the previous-row test.
            begin = 1 if name == "previous_row_last" else 0
            delta = leading_actual[begin:] - candidate[begin:]
            leading_text.append(
                f"{name}:exact={int(np.count_nonzero(delta == 0))}/{delta.size},"
                f"MAE={float(np.abs(delta).mean()):.4f}"
            )
        print("           leading models: " + "; ".join(leading_text))

        sequence_model = observed_models[sequence_position]
        sequence_row = sequence_model.row
        sequence_ll = sequence_model.ll
        transaction_last = sequence_model.final_transaction_last
        sequence_row_delta = actual - sequence_row.astype(np.int64)
        sequence_ll_delta = actual_ll - sequence_ll.astype(np.int64)
        row_exact = int(np.count_nonzero(sequence_row_delta == 0))
        ll_exact = int(np.count_nonzero(sequence_ll_delta == 0))
        sequence_totals["row_exact"] += row_exact
        sequence_totals["row_elements"] += sequence_row_delta.size
        sequence_totals["ll_exact"] += ll_exact
        sequence_totals["ll_elements"] += sequence_ll_delta.size
        print(
            f"           state-carry model: ROW={row_exact}/{sequence_row_delta.size}, "
            f"LL={ll_exact}/{sequence_ll_delta.size}, "
            f"ROW max_abs={int(np.abs(sequence_row_delta).max())}, "
            f"LL max_abs={int(np.abs(sequence_ll_delta).max())}"
        )
        per_image.append(
            {
                "execution_position": sequence_position,
                "global_index": index,
                "primary_reference_first_difference": "ROW",
                "row_phase_offset": int(best[2]),
                "row_phase_valid_exact": int(-best[1]),
                "row_phase_valid_elements": int(128 * np.count_nonzero((2 * np.arange(64) + best[2] >= 0) & (2 * np.arange(64) + best[2] < 128))),
                "column_phase_offset": int(column_best[2]),
                "column_phase_valid_exact": int(-column_best[1]),
                "column_phase_valid_elements": int(64 * np.count_nonzero((2 * np.arange(64) + column_best[2] >= 0) & (2 * np.arange(64) + column_best[2] < 128))),
                "ps_pack_exact": int(np.count_nonzero(pack_delta == 0)),
                "ps_pack_elements": int(pack_delta.size),
                "observed_model_row_exact": row_exact,
                "observed_model_row_elements": int(sequence_row_delta.size),
                "observed_model_ll_exact": ll_exact,
                "observed_model_ll_elements": int(sequence_ll_delta.size),
                "row_max_abs_error": int(np.abs(sequence_row_delta).max()),
                "ll_max_abs_error": int(np.abs(sequence_ll_delta).max()),
            }
        )
        model_archive[f"index_{index}_ROW"] = sequence_row
        model_archive[f"index_{index}_LL"] = sequence_ll
        model_archive[f"index_{index}_CNN_INPUT"] = sequence_model.cnn_input

    aggregate: list[tuple[float, int, int, float, int]] = []
    for offset, values in totals.items():
        elements = int(values["elements"])
        exact = int(values["exact"])
        mae = values["abs_error"] / elements if elements else float("inf")
        aggregate.append((mae, -exact, offset, 100.0 * exact / elements, int(values["max"])))

    print("\nAggregate best offsets:")
    print("offset,MAE,exact_percent,max_abs")
    for mae, _, offset, exact_percent, maximum in sorted(aggregate)[:10]:
        print(f"{offset:+d},{mae:.6f},{exact_percent:.6f},{maximum}")

    print(
        "\nState-carry sequence aggregate: "
        f"ROW={sequence_totals['row_exact']}/{sequence_totals['row_elements']}, "
        f"LL={sequence_totals['ll_exact']}/{sequence_totals['ll_elements']}"
    )

    if args.output_dir is not None:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(args.uart_trace.resolve().read_bytes()).hexdigest()
        report = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "uart_trace": str(args.uart_trace.resolve()),
            "uart_trace_sha256": digest,
            "execution_order": execution_order,
            "primary_reference_first_difference": "ROW",
            "observed_row_phase_offset": -4,
            "interpretation": (
                "The integrated controller emits centered even-phase outputs. "
                "The first two outputs use state carried from the preceding 1-D "
                "transaction; the final two use trailing last-sample replication."
            ),
            "state_carry_sequence_aggregate": sequence_totals,
            "per_image": per_image,
        }
        (output / "observed_controller_alignment.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        with (output / "observed_controller_alignment.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
            writer.writeheader()
            writer.writerows(per_image)
        np.savez_compressed(output / "observed_controller_expected.npz", **model_archive)


if __name__ == "__main__":
    main()
