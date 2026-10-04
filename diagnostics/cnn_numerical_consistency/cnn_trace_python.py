"""Independent NumPy implementation of the frozen deployed INT8 CNN trace."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


CHECKPOINTS = {
    "CNN_INPUT": ((64, 64), np.int8),
    "CONV1_ACC": ((16, 64, 64), np.int32),
    "CONV1_SHIFTED": ((16, 64, 64), np.int32),
    "CONV1_ACT": ((16, 64, 64), np.uint8),
    "POOL1": ((16, 32, 32), np.uint8),
    "CONV2_ACC": ((32, 32, 32), np.int32),
    "CONV2_SHIFTED": ((32, 32, 32), np.int32),
    "CONV2_ACT": ((32, 32, 32), np.uint8),
    "POOL2": ((32, 16, 16), np.uint8),
    "GAP_SUM": ((32,), np.uint32),
    "GAP_SHIFTED": ((32,), np.uint32),
    "LOGITS": ((4,), np.int32),
    "PREDICTION": ((), np.uint8),
}


def parse_c_array(text: str, name: str, dtype: np.dtype) -> np.ndarray:
    match = re.search(
        rf"const\s+(?:int8_t|int32_t)\s+{re.escape(name)}\s*\[[^\]]+\]\s*=\s*\{{(.*?)\}};",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"weight array {name!r} not found")
    return np.asarray([int(value) for value in re.findall(r"-?\d+", match.group(1))], dtype=dtype)


def load_weights(path: Path) -> dict[str, np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="strict")
    arrays = {
        "c1_w": parse_c_array(text, "c1_w", np.int8).reshape(3, 3, 16),
        "c1_b": parse_c_array(text, "c1_b", np.int32),
        "c2_w": parse_c_array(text, "c2_w", np.int8).reshape(3, 3, 16, 32),
        "c2_b": parse_c_array(text, "c2_b", np.int32),
        "d_w": parse_c_array(text, "d_w", np.int8).reshape(32, 4),
        "d_b": parse_c_array(text, "d_b", np.int32),
    }
    expected = {"c1_b": 16, "c2_b": 32, "d_b": 4}
    for name, size in expected.items():
        if arrays[name].size != size:
            raise ValueError(f"{name} has {arrays[name].size} values, expected {size}")
    return arrays


def maxpool2x2(tensor: np.ndarray) -> np.ndarray:
    channels, height, width = tensor.shape
    return tensor.reshape(channels, height // 2, 2, width // 2, 2).max(axis=(2, 4))


def trace_cnn(cnn_input: np.ndarray, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    x = np.asarray(cnn_input, dtype=np.int8)
    if x.shape != (64, 64):
        raise ValueError(f"expected CNN input (64,64), got {x.shape}")

    padded = np.pad(x.astype(np.int32), ((1, 1), (1, 1)), mode="constant")
    c1_hwc = np.broadcast_to(weights["c1_b"], (64, 64, 16)).copy()
    for ky in range(3):
        for kx in range(3):
            c1_hwc += (
                padded[ky : ky + 64, kx : kx + 64, None]
                * weights["c1_w"][ky, kx].astype(np.int32)
            )
    c1_acc = c1_hwc.transpose(2, 0, 1).astype(np.int32)
    c1_shift = np.right_shift(c1_acc, 9).astype(np.int32)
    c1_act = np.where(c1_shift <= 0, 0, np.where(c1_shift > 127, 127, c1_shift)).astype(np.uint8)
    pool1 = maxpool2x2(c1_act)

    padded2 = np.pad(pool1.astype(np.int32), ((0, 0), (1, 1), (1, 1)), mode="constant")
    c2_hwo = np.broadcast_to(weights["c2_b"], (32, 32, 32)).copy()
    for ky in range(3):
        for kx in range(3):
            patch = padded2[:, ky : ky + 32, kx : kx + 32]
            c2_hwo += np.einsum(
                "ihw,io->hwo",
                patch,
                weights["c2_w"][ky, kx].astype(np.int32),
                dtype=np.int32,
                optimize=True,
            )
    c2_acc = c2_hwo.transpose(2, 0, 1).astype(np.int32)
    c2_shift = np.right_shift(c2_acc, 8).astype(np.int32)
    c2_act = np.where(c2_shift <= 0, 0, np.where(c2_shift > 127, 127, c2_shift)).astype(np.uint8)
    pool2 = maxpool2x2(c2_act)

    # The deployed HLS implementation uses uint32_t because POOL2 is uint8.
    # Consequently GAP sums cannot be negative and >>8 is an unsigned logical
    # shift; for these nonnegative values it is exactly floor(sum/256).
    gap_sum = pool2.astype(np.uint32).sum(axis=(1, 2), dtype=np.uint32)
    gap_shift = np.right_shift(gap_sum, 8).astype(np.uint32)
    logits64 = weights["d_b"].astype(np.int64) + (
        gap_shift.astype(np.int64)[:, None] * weights["d_w"].astype(np.int64)
    ).sum(axis=0)
    if logits64.min() < np.iinfo(np.int32).min or logits64.max() > np.iinfo(np.int32).max:
        raise OverflowError("dense accumulator exceeds int32")
    logits = logits64.astype(np.int32)
    prediction = np.asarray(int(np.argmax(logits)), dtype=np.uint8)

    return {
        "CNN_INPUT": x,
        "CONV1_ACC": c1_acc,
        "CONV1_SHIFTED": c1_shift,
        "CONV1_ACT": c1_act,
        "POOL1": pool1,
        "CONV2_ACC": c2_acc,
        "CONV2_SHIFTED": c2_shift,
        "CONV2_ACT": c2_act,
        "POOL2": pool2,
        "GAP_SUM": gap_sum,
        "GAP_SHIFTED": gap_shift,
        "LOGITS": logits,
        "PREDICTION": prediction,
    }
