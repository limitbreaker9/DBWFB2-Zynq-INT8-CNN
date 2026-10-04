"""Audit the released chain and generate preprocessing first-divergence vectors.

Run from any directory:

    python diagnostics/first_divergence/generate_first_divergence_vectors.py

Explicit indices can be supplied with ``--indices``.  With no explicit list,
the deterministic 5-mismatch/5-match selection documented in the generated
manifest is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from fpga_intent_preprocess import preprocess_raw_image


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "generated"

UART_RELATIVE = [
    Path("results/logs/dbwfb2/uart/dbwfb2_full3200_part0_0000_0799.txt"),
    Path("results/logs/dbwfb2/uart/dbwfb2_full3200_part1_0800_1599.txt"),
    Path("results/logs/dbwfb2/uart/dbwfb2_full3200_part2_1600_2399.txt"),
    Path("results/logs/dbwfb2/uart/dbwfb2_full3200_part3_2400_3199.txt"),
]
RAW_HEADER_RELATIVE = [
    Path(f"hardware/deployment/dbwfb2/raw3200_parts/test_vectors_raw_3200_part{part}_dbwfb2.h")
    for part in range(4)
]
HLS_LOG_RELATIVE = Path("results/logs/dbwfb2/hls/dbwfb2_hls_csim_3200.txt")
HLS_EXPECTED_RELATIVE = Path("hardware/hls/dbwfb2/expected_int8_3200.h")

PRED_RE = re.compile(r"^PRED,(\d+),(\d+),(\d+),(\d+),(\d+)\s*$", re.MULTILINE)
SUMMARY_RE = re.compile(r"^SUMMARY_ACC,(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)\s*$", re.MULTILINE)
PART_RE = re.compile(r"^SUMMARY_PART,(\d+),(\d+),(\d+),(\d+)\s*$", re.MULTILINE)
HEADER_IMAGE_RE = re.compile(r"//\s*local=(\d+),\s*test_index=(\d+),\s*true=([A-Za-z0-9_]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_identity(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(REPOSITORY).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def parse_c_uint8_array(text: str, name: str) -> np.ndarray:
    pattern = re.compile(
        rf"const\s+uint8_t\s+{re.escape(name)}\s*\[[^\]]+\]\s*=\s*\{{(.*?)\}};",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"array {name!r} not found")
    return np.asarray([int(x) for x in re.findall(r"\d+", match.group(1))], dtype=np.uint8)


def header_metadata(path: Path) -> dict:
    prefix_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if "const uint8_t test_images_raw" in line:
                break
            prefix_lines.append(line)
    prefix = "".join(prefix_lines)

    def macro(name: str) -> int:
        match = re.search(rf"^#define\s+{name}\s+(\d+)\s*$", prefix, re.MULTILINE)
        if not match:
            raise ValueError(f"macro {name} missing from {path}")
        return int(match.group(1))

    return {
        "n": macro("N_TEST_IMAGES"),
        "start": macro("TEST_PART_START"),
        "labels": parse_c_uint8_array(prefix, "test_labels"),
        "expected": parse_c_uint8_array(prefix, "expected_labels"),
        "expected_alt": parse_c_uint8_array(prefix, "expected_labels_alt"),
    }


def extract_selected_raw(path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    found: dict[int, np.ndarray] = {}
    current_index: int | None = None
    collecting = False
    values: list[int] = []

    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            comment = HEADER_IMAGE_RE.search(line)
            if comment:
                current_index = int(comment.group(2))
                collecting = current_index in wanted
                values = []
                continue
            if not collecting:
                continue
            if "{" in line and not values:
                continue
            if "}" in line:
                if current_index is None:
                    raise AssertionError("image terminator without an index")
                if len(values) != 128 * 128:
                    raise ValueError(
                        f"{path}: index {current_index} contains {len(values)} raw values"
                    )
                array = np.asarray(values, dtype=np.uint8).reshape(128, 128)
                found[current_index] = array
                collecting = False
                current_index = None
                values = []
                continue
            values.extend(int(x) for x in re.findall(r"\d+", line))
    return found


def audit_uart_logs() -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    parts: list[dict] = []

    for expected_part, relative in enumerate(UART_RELATIVE):
        path = REPOSITORY / relative
        text = path.read_text(encoding="utf-8", errors="replace")
        pred_rows = [tuple(map(int, match)) for match in PRED_RE.findall(text)]
        summaries = [tuple(map(int, match)) for match in SUMMARY_RE.findall(text)]
        identities = [tuple(map(int, match)) for match in PART_RE.findall(text)]
        if len(pred_rows) != 800 or len(summaries) != 1 or len(identities) != 1:
            raise ValueError(
                f"{relative}: expected 800 PRED rows, one SUMMARY_ACC, and one SUMMARY_PART"
            )
        part, start, end, count = identities[0]
        if (part, start, end, count) != (
            expected_part,
            expected_part * 800,
            expected_part * 800 + 799,
            800,
        ):
            raise ValueError(f"unexpected part mapping in {relative}: {identities[0]}")
        if [row[0] for row in pred_rows] != list(range(start, end + 1)):
            raise ValueError(f"PRED global indices do not match SUMMARY_PART in {relative}")

        scheduled, completed, correct_summary, sw_summary, alt_summary, hw_errors = summaries[0]
        correct_rows = sum(true == fpga for _, true, _, _, fpga in pred_rows)
        sw_rows = sum(python == fpga for _, _, python, _, fpga in pred_rows)
        alt_rows = sum(alt == fpga for _, _, _, alt, fpga in pred_rows)
        if (scheduled, completed) != (800, 800):
            raise ValueError(f"incomplete UART part {relative}: {summaries[0]}")
        if (correct_rows, sw_rows, alt_rows) != (correct_summary, sw_summary, alt_summary):
            raise ValueError(f"per-image PRED rows disagree with summary in {relative}")
        if hw_errors != 0:
            raise ValueError(f"hardware errors reported in {relative}: {hw_errors}")

        for global_index, true, python, alt, fpga in pred_rows:
            records.append(
                {
                    "global_index": global_index,
                    "true_label": true,
                    "python_prediction": python,
                    "alt_prediction": alt,
                    "fpga_prediction": fpga,
                    "match": python == fpga,
                    "part": part,
                    "source_log": relative.as_posix(),
                }
            )
        parts.append(
            {
                "part": part,
                "start": start,
                "end": end,
                "records": len(pred_rows),
                "fpga_correct_from_pred": correct_rows,
                "python_matches_from_pred": sw_rows,
                "hardware_errors": hw_errors,
                "source_log": relative.as_posix(),
                "sha256": sha256(path),
            }
        )

    records.sort(key=lambda row: row["global_index"])
    indices = [row["global_index"] for row in records]
    if indices != list(range(3200)):
        raise ValueError("four UART logs do not cover global indices 0..3199 exactly once")
    return records, parts


def audit_headers(records: list[dict]) -> tuple[dict[int, dict], list[dict]]:
    metadata_by_index: dict[int, dict] = {}
    header_records: list[dict] = []

    for part, relative in enumerate(RAW_HEADER_RELATIVE):
        path = REPOSITORY / relative
        meta = header_metadata(path)
        if meta["n"] != 800 or meta["start"] != part * 800:
            raise ValueError(f"unexpected header mapping in {relative}")
        for key in ("labels", "expected", "expected_alt"):
            if meta[key].shape != (800,):
                raise ValueError(f"{relative}: {key} contains {meta[key].size} entries")
        for local in range(800):
            global_index = meta["start"] + local
            metadata_by_index[global_index] = {
                "part": part,
                "local_index": local,
                "true_label": int(meta["labels"][local]),
                "python_prediction": int(meta["expected"][local]),
                "alt_prediction": int(meta["expected_alt"][local]),
                "source_header": relative.as_posix(),
            }
        header_records.append(
            {
                "part": part,
                "start": meta["start"],
                "end": meta["start"] + meta["n"] - 1,
                "source_header": relative.as_posix(),
                "sha256": sha256(path),
            }
        )

    for row in records:
        header = metadata_by_index[row["global_index"]]
        for field in ("true_label", "python_prediction", "alt_prediction"):
            if row[field] != header[field]:
                raise ValueError(
                    f"UART/header {field} conflict at global index {row['global_index']}"
                )

    hls_expected_path = REPOSITORY / HLS_EXPECTED_RELATIVE
    hls_text = hls_expected_path.read_text(encoding="utf-8", errors="strict")
    hls_expected = parse_c_uint8_array(hls_text, "expected_int8_labels")
    concatenated = np.asarray(
        [metadata_by_index[index]["python_prediction"] for index in range(3200)],
        dtype=np.uint8,
    )
    if hls_expected.shape != (3200,) or not np.array_equal(hls_expected, concatenated):
        raise ValueError("raw-header Python predictions differ from HLS expected_int8_3200.h")

    return metadata_by_index, header_records


def audit_hls_log() -> dict:
    path = REPOSITORY / HLS_LOG_RELATIVE
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = re.findall(
        r"^Image\s+(\d+)\s*\|.*?\|\s*true:(PASS|FAIL)\s*\|\s*SW:(MATCH|MISMATCH)\s*$",
        text,
        re.MULTILINE,
    )
    if len(rows) != 3200:
        raise ValueError(f"expected 3200 HLS per-image records, found {len(rows)}")
    indices = [int(row[0]) for row in rows]
    if indices != list(range(3200)):
        raise ValueError("HLS log image indices do not cover 0..3199 exactly once")
    correct = sum(result == "PASS" for _, result, _ in rows)
    matches = sum(result == "MATCH" for _, _, result in rows)
    return {
        "images": len(rows),
        "correct_from_per_image_log": correct,
        "accuracy_percent": 100.0 * correct / len(rows),
        "python_matches_from_per_image_log": matches,
        "python_agreement_percent": 100.0 * matches / len(rows),
        "source_log": HLS_LOG_RELATIVE.as_posix(),
        "sha256": sha256(path),
    }


def deterministic_selection(records: list[dict]) -> tuple[list[int], dict[int, str]]:
    mismatches = [row for row in records if not row["match"]]
    matches = [row for row in records if row["match"]]
    selected_mismatch: list[dict] = []

    # One mismatch for each true class, deliberately mapped to a different
    # released 800-image part.  Within each constraint, choose the lowest index.
    for part, true_label in enumerate(range(4)):
        candidates = [
            row for row in mismatches if row["part"] == part and row["true_label"] == true_label
        ]
        if not candidates:
            raise ValueError(f"no mismatch for true class {true_label} in part {part}")
        selected_mismatch.append(min(candidates, key=lambda row: row["global_index"]))

    transitions = {
        (row["python_prediction"], row["fpga_prediction"]) for row in selected_mismatch
    }
    fifth_mismatch = next(
        row
        for row in mismatches
        if row not in selected_mismatch
        and (row["python_prediction"], row["fpga_prediction"]) not in transitions
    )
    selected_mismatch.append(fifth_mismatch)

    selected_match: list[dict] = []
    for part, true_label in enumerate(range(4)):
        candidates = [
            row for row in matches if row["part"] == part and row["true_label"] == true_label
        ]
        if not candidates:
            raise ValueError(f"no match for true class {true_label} in part {part}")
        selected_match.append(min(candidates, key=lambda row: row["global_index"]))

    triples = {
        (row["true_label"], row["python_prediction"], row["fpga_prediction"])
        for row in selected_match
    }
    fifth_match = next(
        row
        for row in matches
        if row not in selected_match
        and (row["true_label"], row["python_prediction"], row["fpga_prediction"])
        not in triples
    )
    selected_match.append(fifth_match)

    mismatch_indices = sorted(row["global_index"] for row in selected_mismatch)
    match_indices = sorted(row["global_index"] for row in selected_match)
    groups = {index: "mismatch" for index in mismatch_indices}
    groups.update({index: "match" for index in match_indices})
    return mismatch_indices + match_indices, groups


def write_diagnostic_header(path: Path, rows: list[dict], arrays: dict[int, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("// GENERATED FIRST-DIVERGENCE DIAGNOSTIC VECTORS\n")
        handle.write("// Source: released 4x800 DBWFB2 raw headers; values unchanged\n")
        handle.write("#pragma once\n#include <stdint.h>\n\n")
        handle.write(f"#define N_TEST_IMAGES {len(rows)}\n")
        handle.write("#define IMG_PIXELS 16384\n")
        handle.write("#define FIRST_DIVERGENCE_DIAGNOSTIC 1\n\n")

        def small_array(type_name: str, name: str, values: Iterable[int]) -> None:
            materialized = [int(value) for value in values]
            handle.write(f"const {type_name} {name}[N_TEST_IMAGES] = {{")
            handle.write(", ".join(map(str, materialized)))
            handle.write("};\n\n")

        small_array("uint32_t", "test_global_indices", (row["global_index"] for row in rows))
        small_array("uint8_t", "test_labels", (row["true_label"] for row in rows))
        small_array("uint8_t", "expected_labels", (row["python_prediction"] for row in rows))
        small_array("uint8_t", "expected_labels_alt", (row["alt_prediction"] for row in rows))
        small_array("uint8_t", "logged_fpga_labels", (row["fpga_prediction"] for row in rows))

        handle.write("const uint8_t test_images_raw[N_TEST_IMAGES][IMG_PIXELS] = {\n")
        for row_number, row in enumerate(rows):
            global_index = row["global_index"]
            handle.write(
                f"  // local={row_number}, test_index={global_index}, "
                f"true={row['true_label']}, set={row['selection_group']}\n  {{\n    "
            )
            flat = arrays[global_index].reshape(-1)
            for position, value in enumerate(flat.tolist()):
                handle.write(str(int(value)))
                if position != flat.size - 1:
                    handle.write(", ")
                if (position + 1) % 16 == 0 and position != flat.size - 1:
                    handle.write("\n    ")
            handle.write("\n  }")
            if row_number != len(rows) - 1:
                handle.write(",")
            handle.write("\n")
        handle.write("};\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", nargs="+", type=int, help="explicit global test indices")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    uart_records, part_audit = audit_uart_logs()
    by_index = {row["global_index"]: row for row in uart_records}
    header_index, header_audit = audit_headers(uart_records)
    hls_audit = audit_hls_log()

    fpga_correct = sum(row["true_label"] == row["fpga_prediction"] for row in uart_records)
    python_matches = sum(row["match"] for row in uart_records)
    python_correct = sum(
        header_index[index]["true_label"] == header_index[index]["python_prediction"]
        for index in range(3200)
    )
    hardware_errors = sum(part["hardware_errors"] for part in part_audit)
    chain = {
        "images": len(uart_records),
        "python_true_int8_correct_from_header_arrays": python_correct,
        "python_true_int8_accuracy_percent": 100.0 * python_correct / len(uart_records),
        "fpga_correct_from_3200_pred_records": fpga_correct,
        "fpga_accuracy_percent": 100.0 * fpga_correct / len(uart_records),
        "fpga_python_matches_from_3200_pred_records": python_matches,
        "fpga_python_agreement_percent": 100.0 * python_matches / len(uart_records),
        "fpga_python_mismatches": len(uart_records) - python_matches,
        "hardware_errors_from_four_uart_summaries": hardware_errors,
    }
    expected_chain = (2497, 3006, 194, 0)
    actual_chain = (
        chain["fpga_correct_from_3200_pred_records"],
        chain["fpga_python_matches_from_3200_pred_records"],
        chain["fpga_python_mismatches"],
        chain["hardware_errors_from_four_uart_summaries"],
    )
    if actual_chain != expected_chain:
        raise ValueError(f"current-chain audit failed: expected {expected_chain}, got {actual_chain}")
    if python_correct != 2488:
        raise ValueError(f"Python-primary header accuracy count is {python_correct}, not 2488")
    if hls_audit["correct_from_per_image_log"] != 2488:
        raise ValueError("HLS per-image accuracy count is not 2488")
    if hls_audit["python_matches_from_per_image_log"] != 3200:
        raise ValueError("HLS per-image Python agreement count is not 3200")

    if args.indices:
        selected_indices = args.indices
        if len(selected_indices) != len(set(selected_indices)):
            raise ValueError("explicit indices contain a duplicate")
        if any(index not in by_index for index in selected_indices):
            raise ValueError("explicit index is outside the released 0..3199 range")
        groups = {
            index: ("match" if by_index[index]["match"] else "mismatch")
            for index in selected_indices
        }
    else:
        selected_indices, groups = deterministic_selection(uart_records)

    selected_rows: list[dict] = []
    for index in selected_indices:
        row = dict(by_index[index])
        row.update(header_index[index])
        row["fpga_prediction"] = by_index[index]["fpga_prediction"]
        row["match"] = bool(by_index[index]["match"])
        row["selection_group"] = groups[index]
        selected_rows.append(row)

    wanted = set(selected_indices)
    raw_by_index: dict[int, np.ndarray] = {}
    for relative in RAW_HEADER_RELATIVE:
        raw_by_index.update(extract_selected_raw(REPOSITORY / relative, wanted))
    if set(raw_by_index) != wanted:
        raise ValueError(f"missing selected raw images: {sorted(wanted - set(raw_by_index))}")

    results = {index: preprocess_raw_image(raw_by_index[index]) for index in selected_indices}

    header_path = output / "first_divergence_vectors.h"
    write_diagnostic_header(header_path, selected_rows, raw_by_index)
    generated_raw = extract_selected_raw(header_path, wanted)

    identity_rows: list[dict] = []
    for row in selected_rows:
        index = row["global_index"]
        identical = index in generated_raw and np.array_equal(raw_by_index[index], generated_raw[index])
        if not identical:
            raise ValueError(f"generated header raw vector differs at index {index}")
        identity_rows.append(
            {
                "global_index": index,
                "source_header": row["source_header"],
                "source_local_index": row["local_index"],
                "raw_values": int(raw_by_index[index].size),
                "value_for_value_identical": "PASS",
                "raw_sha256": hashlib.sha256(raw_by_index[index].tobytes()).hexdigest(),
            }
        )

    np.savez_compressed(
        output / "expected_first_divergence.npz",
        global_indices=np.asarray(selected_indices, dtype=np.int32),
        true_labels=np.asarray([row["true_label"] for row in selected_rows], dtype=np.uint8),
        python_predictions=np.asarray(
            [row["python_prediction"] for row in selected_rows], dtype=np.uint8
        ),
        logged_fpga_predictions=np.asarray(
            [row["fpga_prediction"] for row in selected_rows], dtype=np.uint8
        ),
        expected_match=np.asarray([row["match"] for row in selected_rows], dtype=np.bool_),
        raw=np.stack([results[index].raw for index in selected_indices]),
        row=np.stack([results[index].row for index in selected_indices]),
        ll=np.stack([results[index].ll for index in selected_indices]),
        preclip=np.stack([results[index].preclip for index in selected_indices]),
        cnn_input=np.stack([results[index].cnn_input for index in selected_indices]),
        packed_words=np.stack([results[index].packed_words for index in selected_indices]),
        low_saturation_count=np.asarray(
            [results[index].low_saturation_count for index in selected_indices], dtype=np.int32
        ),
        high_saturation_count=np.asarray(
            [results[index].high_saturation_count for index in selected_indices], dtype=np.int32
        ),
    )

    with (output / "selected_diagnostic_indices.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "selection_group",
            "global_index",
            "part",
            "local_index",
            "true_label",
            "python_prediction",
            "fpga_prediction",
            "match",
            "source_header",
            "source_log",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected_rows)

    with (output / "expected_first_divergence_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "global_index",
            "selection_group",
            "raw_min",
            "raw_max",
            "row_min",
            "row_max",
            "ll_min",
            "ll_max",
            "preclip_min",
            "preclip_max",
            "low_saturation_count",
            "high_saturation_count",
            "saturation_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected_rows:
            index = row["global_index"]
            result = results[index]
            writer.writerow(
                {
                    "global_index": index,
                    "selection_group": row["selection_group"],
                    "raw_min": int(result.raw.min()),
                    "raw_max": int(result.raw.max()),
                    "row_min": int(result.row.min()),
                    "row_max": int(result.row.max()),
                    "ll_min": int(result.ll.min()),
                    "ll_max": int(result.ll.max()),
                    "preclip_min": int(result.preclip.min()),
                    "preclip_max": int(result.preclip.max()),
                    "low_saturation_count": result.low_saturation_count,
                    "high_saturation_count": result.high_saturation_count,
                    "saturation_fraction": (
                        result.low_saturation_count + result.high_saturation_count
                    )
                    / 4096.0,
                }
            )

    with (output / "header_identity_verification.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(identity_rows[0]))
        writer.writeheader()
        writer.writerows(identity_rows)

    package_components_relative = [
        Path("software/int8_packages/dbwfb2/selected_package.json"),
        Path("hardware/deployment/dbwfb2/weights.h"),
        Path("hardware/deployment/dbwfb2/cnn_accel_shifts.txt"),
    ]
    package_components = [REPOSITORY / path for path in package_components_relative]
    package_manifest = json.loads(package_components[0].read_text(encoding="utf-8"))

    source_of_truth = {
        "purpose": "first-divergence diagnostic source-of-truth; publication files unchanged",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": package_manifest["protocol_version"],
        "selected_package": {
            "arm": package_manifest["arm"],
            "seed": package_manifest["selected_seed_frozen_before_test"],
            "gain": package_manifest["input_gain_frozen_before_test"],
            "C1_SHIFT": package_manifest["C1_SHIFT"],
            "C2_SHIFT": package_manifest["C2_SHIFT"],
            "identity_sha256": combined_identity(package_components),
            "components": [
                {"path": relative.as_posix(), "sha256": sha256(REPOSITORY / relative)}
                for relative in package_components_relative
            ],
        },
        "authoritative_raw_headers": header_audit,
        "authoritative_python_prediction_source": {
            "description": "expected_labels arrays in the four raw headers; byte-for-byte label sequence also present in HLS expected_int8_3200.h",
            "hls_expected_path": HLS_EXPECTED_RELATIVE.as_posix(),
            "hls_expected_sha256": sha256(REPOSITORY / HLS_EXPECTED_RELATIVE),
        },
        "authoritative_fpga_uart_parts": part_audit,
        "current_chain_recomputed_from_primary_records": chain,
        "hls_c_simulation_recomputed_from_3200_per_image_lines": hls_audit,
        "meaning_of_2488_over_3200": {
            "python_true_int8_accuracy": "2488/3200 = 77.75%",
            "hls_c_simulation_accuracy": "2488/3200 = 77.75%",
            "python_hls_prediction_agreement": "3200/3200 = 100%",
            "note": "2488/3200 is an accuracy count against true labels, not an agreement count",
        },
        "selection_algorithm": {
            "mismatch": "earliest mismatch for (part,true_class)=(0,0),(1,1),(2,2),(3,3), then earliest remaining mismatch with a new Python-to-FPGA transition",
            "match": "earliest match for (part,true_class)=(0,0),(1,1),(2,2),(3,3), then earliest remaining match with a new (true,Python,FPGA) triple",
        },
        "selected_indices": selected_rows,
        "generated_artifacts": {
            "header": header_path.relative_to(REPOSITORY).as_posix(),
            "header_sha256": sha256(header_path),
            "expected_npz": (output / "expected_first_divergence.npz")
            .relative_to(REPOSITORY)
            .as_posix(),
        },
    }
    with (HERE / "current_chain_source_of_truth.json").open("w", encoding="utf-8") as handle:
        json.dump(source_of_truth, handle, indent=2)
        handle.write("\n")

    print("CURRENT CHAIN AUDIT: PASS")
    print(f"Python true-INT8 correct: {python_correct}/3200 = {100.0*python_correct/3200:.2f}%")
    print(f"FPGA correct: {fpga_correct}/3200 = {100.0*fpga_correct/3200:.5f}%")
    print(f"Python/FPGA matches: {python_matches}/3200 = {100.0*python_matches/3200:.4f}%")
    print(f"Mismatches: {3200-python_matches}; hardware errors: {hardware_errors}")
    print(
        f"HLS accuracy: {hls_audit['correct_from_per_image_log']}/3200 = "
        f"{hls_audit['accuracy_percent']:.2f}%"
    )
    print(
        f"Python/HLS agreement: {hls_audit['python_matches_from_per_image_log']}/3200 = "
        f"{hls_audit['python_agreement_percent']:.2f}%"
    )
    print("Selected indices:", selected_indices)
    for row in identity_rows:
        print(
            f"index {row['global_index']}: source raw vector == diagnostic raw vector: "
            f"{row['value_for_value_identical']}"
        )


if __name__ == "__main__":
    main()
