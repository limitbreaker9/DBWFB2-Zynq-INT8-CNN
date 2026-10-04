"""Replay the included Lena firmware transaction order using its original RTL.

Requires Icarus Verilog on PATH. All generated files go to a new output
directory, leaving the included source and captured results unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idle", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")
    args = parser.parse_args()
    work = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    rtl = work / "rtl"
    header = work / "source/lena_data.h"
    firmware = work / "source/main.c"
    source_text = header.read_text(encoding="utf-8")
    data = re.search(r"lena_image\[256\]\[256\]\s*=\s*\{(.*?)\};", source_text, re.S)
    assert data, "Expected the included 256x256 Lena array"
    numbers = [int(n) for n in re.findall(r"\d+", data[1])]
    assert len(numbers) == 65536 and all(0 <= n <= 255 for n in numbers)
    pixel_bytes = bytes(numbers)
    names = ["dbwfb2_97_bram_ctrl.v", "dbwfb2_97_conv.v", "delay_chain.v",
             "lpf_shiftadd.v", "hpf_shiftadd.v", "output_norm.v"]
    source_paths = [rtl / n for n in names]
    xci_dir = work / "configuration"
    xcis = [xci_dir / (n + ".xci") for n in ["PL_blk_mem_gen_0_0", "PL_blk_mem_gen_0_1"]]
    bram_settings = {}
    for xci in xcis:
        parameters = json.loads(xci.read_text())["ip_inst"]["parameters"]["component_parameters"]
        selected = {n: parameters[n][0]["value"] for n in ["READ_LATENCY_B", "Read_Width_B",
                    "Register_PortB_Output_of_Memory_Primitives", "Register_PortB_Output_of_Memory_Core",
                    "Enable_32bit_Address"]}
        assert selected["READ_LATENCY_B"] == "1"
        assert selected["Register_PortB_Output_of_Memory_Primitives"] == "false"
        assert selected["Register_PortB_Output_of_Memory_Core"] == "false"
        bram_settings[str(xci)] = selected
    output.mkdir(parents=True, exist_ok=False)
    (output / "lena_pixels.hex").write_text("\n".join(f"{n:02x}" for n in numbers) + "\n", encoding="utf-8")
    executable = output / "lena_sim.vvp"
    compile_command = [args.iverilog, "-g2012", "-s", "lena_tb", "-o", str(executable),
                       str(work / "lena_tb.sv"), *map(str, source_paths)]
    compiled = subprocess.run(compile_command, cwd=output, capture_output=True, text=True, check=True)
    run_command = [args.vvp, str(executable), f"+idle={args.idle}"]
    simulated = subprocess.run(run_command, cwd=output, capture_output=True, text=True, check=True)
    rows = [[int(n) for n in line.split()] for line in (output / "subbands.txt").read_text().splitlines()]
    assert len(rows) == 16384 and all(len(row) == 4 for row in rows)
    assert all(-32768 <= n <= 32767 for row in rows for n in row)
    bands = ["LL", "LH", "HL", "HH"]
    raw = {band: sum(float(row[i]) * float(row[i]) for row in rows) for i, band in enumerate(bands)}
    total = sum(raw.values())
    normalized = {band: raw[band] / total * 100.0 for band in bands}
    expected = dict(zip(bands, [99.429778, 0.119650, 0.329864, 0.120710]))
    differences = {band: abs(normalized[band] - expected[band]) for band in bands}
    originals = [header, firmware, *source_paths, *xcis]
    result = {
        "method": "Icarus simulation of original RTL and firmware execution order",
        "input_dimensions": [256, 256], "subband_dimensions": [128, 128],
        "input_pixel_sha256": hashlib.sha256(pixel_bytes).hexdigest(),
        "source_file_sha256": {f.relative_to(work).as_posix(): sha(f) for f in originals},
        "lpf": {"coefficients": [27, -17, -78, 273, 614, 273, -78, -17, 27], "arithmetic_shift": 10},
        "hpf": {"coefficients": [3, -2, -19, 36, -19, -2, 3], "arithmetic_shift": 6},
        "boundary": "Original controller trailing last-sample replication; BRAM output holds between transactions; original persistent FIR state",
        "decimation": "Firmware retains even horizontal column addresses and even vertical row addresses; original controller pipe_fill=9",
        "arithmetic": "Original signed-16 taps, signed-17 symmetric sums, signed-32 shift/add and registers; arithmetic shift then low-16 two's-complement truncation; no rounding or saturation",
        "bram_settings": bram_settings,
        "idle_cycles": args.idle,
        "input_writes": "One per clock before each transaction; at least 256 idle clocks, no inter-transaction reset",
        "firmware_sequence": "256 horizontal transactions followed by 128 even columns, each with low-stream then high-stream transaction",
        "raw_energy_double": raw, "total_energy_double": total,
        "normalized_percent": normalized, "expected_percent": expected,
        "absolute_percentage_point_difference": differences,
        "maximum_absolute_percentage_point_difference": max(differences.values()),
        "near_exact_to_six_decimals": all(round(normalized[b], 6) == expected[b] for b in bands),
        "matches_firmware_five_decimal_display": all(round(normalized[b], 5) == round(expected[b], 5) for b in bands),
        "matches_manuscript_two_decimal_display": all(round(normalized[b], 2) == round(expected[b], 2) for b in bands),
        "precision_note": "Near-exact, not identical at all six target decimals. No phase, boundary, coefficient, sign, or truncation parameter was fitted to the target percentages.",
        "simulation_stdout": simulated.stdout, "compile_stderr": compiled.stderr,
        "testbench_sha256": sha(work / "lena_tb.sv"), "script_sha256": sha(Path(__file__)),
        "subbands_sha256": sha(output / "subbands.txt"),
        "compile_command": compile_command, "run_command": run_command,
    }
    (output / f"reproduction_idle_{args.idle}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
