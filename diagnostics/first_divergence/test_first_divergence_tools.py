"""Lightweight local tests for the diagnostic parser and numerical archive."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

import numpy as np

from compare_first_divergence import comparison, parse_uart


HERE = Path(__file__).resolve().parent


class DiagnosticToolsTest(unittest.TestCase):
    def test_reference_shaped_uart_trace_round_trip(self) -> None:
        archive = np.load(HERE / "generated" / "expected_first_divergence.npz")
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = Path(temporary) / "synthetic_trace.txt"
            with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
                for offset, index in enumerate(archive["global_indices"].astype(int)):
                    true_label = int(archive["true_labels"][offset])
                    python = int(archive["python_predictions"][offset])
                    fpga = int(archive["logged_fpga_predictions"][offset])
                    handle.write(
                        f"TRACE_IMAGE_BEGIN,index={index},true={true_label},python={python}\n"
                    )
                    for marker, key, dtype in (
                        ("RAW", "raw", "uint32_readback"),
                        ("ROW", "row", "int16"),
                        ("LL", "ll", "int16"),
                        ("CNN_INPUT", "cnn_input", "int8"),
                    ):
                        array = archive[key][offset]
                        handle.write(
                            f"{marker}_BEGIN,rows={array.shape[0]},"
                            f"cols={array.shape[1]},dtype={dtype}\n"
                        )
                        for row_index, row in enumerate(array):
                            values = ",".join(map(str, row.astype(int).tolist()))
                            handle.write(f"{marker}_ROW,{row_index},{values}\n")
                        handle.write(f"{marker}_END\n")
                    handle.write(
                        f"TRACE_RESULT,index={index},true={true_label},"
                        f"python={python},fpga={fpga}\n"
                    )
                    handle.write(f"TRACE_IMAGE_END,index={index}\n")

            parsed = parse_uart(trace_path)
            self.assertEqual(set(parsed), set(archive["global_indices"].astype(int)))
            for offset, index in enumerate(archive["global_indices"].astype(int)):
                for marker, key in (
                    ("RAW", "raw"),
                    ("ROW", "row"),
                    ("LL", "ll"),
                    ("CNN_INPUT", "cnn_input"),
                ):
                    stats, _ = comparison(archive[key][offset], parsed[index]["stages"][marker])
                    self.assertEqual(stats["mismatch_count"], 0)

            output = Path(temporary) / "comparison"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(HERE / "compare_first_divergence.py"),
                    str(trace_path),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "comparison_summary.json").is_file())
            self.assertTrue((output / "comparison_stage_summary.csv").is_file())
            self.assertTrue((output / "mismatch_locations.csv").is_file())


if __name__ == "__main__":
    unittest.main()
