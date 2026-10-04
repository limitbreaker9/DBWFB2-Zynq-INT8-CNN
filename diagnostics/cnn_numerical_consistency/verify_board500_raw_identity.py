"""Verify that Board500 raw vectors equal the mapped full-test raw vectors."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
FIRST = REPOSITORY / "diagnostics/first_divergence"
sys.path.insert(0, str(FIRST))

from generate_first_divergence_vectors import (  # noqa: E402
    RAW_HEADER_RELATIVE,
    extract_selected_raw,
)


BOARD_HEADER = REPOSITORY / "hardware/deployment/dbwfb2/test_vectors_raw_500_dbwfb2.h"
BOARD_MAP = REPOSITORY / "results/board500_indices.csv"
COMMENT = re.compile(r"//\s*board_order=(\d+),\s*test_index=(\d+),\s*true=([A-Za-z0-9_]+)")


def parse_board_header() -> dict[int, tuple[int, np.ndarray]]:
    output: dict[int, tuple[int, np.ndarray]] = {}
    board_order = test_index = None
    collecting = False
    values: list[int] = []
    with BOARD_HEADER.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            match = COMMENT.search(line)
            if match:
                board_order, test_index = int(match.group(1)), int(match.group(2))
                collecting = True
                values = []
                continue
            if not collecting:
                continue
            if "{" in line and not values:
                continue
            if "}" in line:
                if len(values) != 128 * 128:
                    raise ValueError(f"Board500 image {board_order} has {len(values)} values")
                output[int(board_order)] = (
                    int(test_index),
                    np.asarray(values, dtype=np.uint8).reshape(128, 128),
                )
                collecting = False
                continue
            values.extend(int(token) for token in re.findall(r"\d+", line))
    if set(output) != set(range(500)):
        raise ValueError("Board500 header does not contain board_order 0..499 exactly once")
    return output


def main() -> None:
    mapping = list(csv.DictReader(BOARD_MAP.open(encoding="utf-8")))
    board = parse_board_header()
    wanted_by_part = [set() for _ in range(4)]
    for row in mapping:
        index = int(row["four_class_test_index"])
        wanted_by_part[index // 800].add(index)
    full = {}
    for part, wanted in enumerate(wanted_by_part):
        full.update(extract_selected_raw(REPOSITORY / RAW_HEADER_RELATIVE[part], wanted))
    for position, row in enumerate(mapping):
        mapped = int(row["four_class_test_index"])
        header_index, array = board[position]
        if header_index != mapped:
            raise ValueError(f"Board500 mapping conflict at position {position}")
        if not np.array_equal(array, full[mapped]):
            where = np.argwhere(array != full[mapped])[0].tolist()
            raise ValueError(f"Board500 raw mismatch at position {position}, coordinate {where}")
    print("Board500 raw identity: 500/500 images exact (8,192,000/8,192,000 bytes)")


if __name__ == "__main__":
    main()
