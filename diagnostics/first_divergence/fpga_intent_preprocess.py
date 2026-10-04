"""Numerical references used by the DBWFB2 first-divergence experiment.

``preprocess_raw_image`` preserves the frozen Python-primary convention used
to generate the public software predictions.  The board trace subsequently
showed that the integrated controller uses a different stream alignment.  The
observed controller behavior is implemented separately by
``preprocess_observed_controller_sequence`` so that the two references are not
conflated.  Neither routine models CNN internals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DBWFB2_COEFFICIENTS = np.asarray(
    [27, -17, -78, 273, 614, 273, -78, -17, 27], dtype=np.int64
)
NORMALIZATION_SHIFT = 10
FILTER_TAPS = int(DBWFB2_COEFFICIENTS.size)
INPUT_SIZE = 128
OUTPUT_SIZE = 64
INPUT_GAIN = 1
INPUT_ZERO_POINT = 127

# For an output window beginning at full-rate position n, samples n..n+8 are
# consumed.  With a 128-sample input, windows beginning at n >= 120 require the
# controller's trailing-edge replicate extension.  Even-index decimation maps
# those positions to output indices 60..63.
FULL_RATE_TRAILING_BOUNDARY_START = INPUT_SIZE - FILTER_TAPS + 1  # 120
DECIMATED_TRAILING_BOUNDARY_START = FULL_RATE_TRAILING_BOUNDARY_START // 2  # 60


@dataclass(frozen=True)
class PreprocessResult:
    raw: np.ndarray
    row: np.ndarray
    ll: np.ndarray
    preclip: np.ndarray
    cnn_input: np.ndarray
    packed_words: np.ndarray
    low_saturation_count: int
    high_saturation_count: int


@dataclass(frozen=True)
class ObservedControllerResult:
    """Intermediate tensors reproduced from the observed controller stream."""

    raw: np.ndarray
    row: np.ndarray
    ll: np.ndarray
    preclip: np.ndarray
    cnn_input: np.ndarray
    packed_words: np.ndarray
    low_saturation_count: int
    high_saturation_count: int
    final_transaction_last: int


def _as_signed_int16(values: np.ndarray) -> np.ndarray:
    """Apply the RTL's final 16-bit two's-complement truncation."""

    unsigned = np.asarray(values, dtype=np.int64) & 0xFFFF
    signed = np.where(unsigned >= 0x8000, unsigned - 0x10000, unsigned)
    return signed.astype(np.int16)


def fir_fullrate_trailing_replicate(values: np.ndarray, axis: int) -> np.ndarray:
    """Apply the released 9-tap integer FIR at every full-rate output position.

    The controller suppresses pipeline-fill outputs, numbers the first complete
    9-sample window as output zero, and drains the final eight positions by
    repeating the last input sample.  ``>>> 10`` is an arithmetic right shift;
    NumPy's signed integer right shift has the same behavior for negative
    values.  The result is then truncated to signed 16 bits as in
    ``output_norm.v``.
    """

    source = np.asarray(values)
    if source.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {source.shape}")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    if source.shape[axis] != INPUT_SIZE:
        raise ValueError(
            f"filtered axis must contain {INPUT_SIZE} samples, got {source.shape[axis]}"
        )

    moved = np.moveaxis(source.astype(np.int64, copy=False), axis, -1)
    tail = np.repeat(moved[..., -1:], FILTER_TAPS - 1, axis=-1)
    extended = np.concatenate((moved, tail), axis=-1)
    accumulator = np.zeros_like(moved, dtype=np.int64)

    for tap, coefficient in enumerate(DBWFB2_COEFFICIENTS.tolist()):
        accumulator += int(coefficient) * extended[..., tap : tap + INPUT_SIZE]

    # For the released uint8 row input and int16 column input this sum is well
    # inside signed 32-bit range.  Failing here protects against silently using
    # this routine with an incompatible tensor.
    if accumulator.min() < -(1 << 31) or accumulator.max() > (1 << 31) - 1:
        raise OverflowError("FIR accumulator exceeds the released signed-32-bit path")

    shifted = np.right_shift(accumulator, NORMALIZATION_SHIFT)
    return np.moveaxis(_as_signed_int16(shifted), -1, axis)


def pack_int8_little_endian(cnn_input: np.ndarray) -> np.ndarray:
    """Pack four signed INT8 pixels per uint32 exactly as the PS firmware does."""

    q = np.asarray(cnn_input, dtype=np.int8)
    if q.shape != (OUTPUT_SIZE, OUTPUT_SIZE):
        raise ValueError(f"expected {(OUTPUT_SIZE, OUTPUT_SIZE)}, got {q.shape}")
    bytes_u8 = q.view(np.uint8).reshape(-1, 4).astype(np.uint32)
    return (
        bytes_u8[:, 0]
        | (bytes_u8[:, 1] << 8)
        | (bytes_u8[:, 2] << 16)
        | (bytes_u8[:, 3] << 24)
    ).astype(np.uint32)


def preprocess_raw_image(raw_image: np.ndarray) -> PreprocessResult:
    """Generate checkpoints under the frozen Python-primary convention."""

    raw = np.asarray(raw_image)
    if raw.shape != (INPUT_SIZE, INPUT_SIZE):
        raise ValueError(f"expected {(INPUT_SIZE, INPUT_SIZE)}, got {raw.shape}")
    if np.any(raw < 0) or np.any(raw > 255):
        raise ValueError("raw image contains a value outside uint8 range")
    raw = raw.astype(np.uint8, copy=True)

    row_full = fir_fullrate_trailing_replicate(raw, axis=1)
    row = row_full[:, 0::2].copy()
    if row.shape != (INPUT_SIZE, OUTPUT_SIZE):
        raise AssertionError(f"unexpected ROW shape {row.shape}")

    column_full = fir_fullrate_trailing_replicate(row, axis=0)
    ll = column_full[0::2, :].copy()
    if ll.shape != (OUTPUT_SIZE, OUTPUT_SIZE):
        raise AssertionError(f"unexpected LL shape {ll.shape}")

    # Firmware freezes gain at one, so this is exactly LL - 127 followed by
    # signed-INT8 clipping.  There is no fractional rounding at this stage.
    preclip = ll.astype(np.int32) - INPUT_ZERO_POINT
    low_count = int(np.count_nonzero(preclip < -128))
    high_count = int(np.count_nonzero(preclip > 127))
    cnn_input = np.clip(preclip, -128, 127).astype(np.int8)
    packed_words = pack_int8_little_endian(cnn_input)

    return PreprocessResult(
        raw=raw,
        row=row,
        ll=ll,
        preclip=preclip,
        cnn_input=cnn_input,
        packed_words=packed_words,
        low_saturation_count=low_count,
        high_saturation_count=high_count,
    )


def _observed_centered_decimated_line(
    line: np.ndarray, transaction_last: int
) -> np.ndarray:
    """Reproduce one observed 1-D controller transaction.

    The trace proves that the controller emits centered even-phase samples.
    Its first two windows use the final sample left by the preceding 1-D
    transaction, repeated four times.  The final two windows repeat the final
    sample of the current transaction.  Accumulation, arithmetic shift, and
    signed-16 truncation are otherwise identical to the frozen reference.
    """

    source = np.asarray(line, dtype=np.int64)
    if source.shape != (INPUT_SIZE,):
        raise ValueError(f"expected one {INPUT_SIZE}-sample line, got {source.shape}")
    extended = np.concatenate(
        (
            np.full(4, int(transaction_last), dtype=np.int64),
            source,
            np.full(4, int(source[-1]), dtype=np.int64),
        )
    )
    accumulator = np.empty(OUTPUT_SIZE, dtype=np.int64)
    for output_index, center in enumerate(range(0, INPUT_SIZE, 2)):
        accumulator[output_index] = int(
            np.sum(extended[center : center + FILTER_TAPS] * DBWFB2_COEFFICIENTS)
        )
    return _as_signed_int16(np.right_shift(accumulator, NORMALIZATION_SHIFT))


def preprocess_observed_controller_sequence(
    raw_images: list[np.ndarray] | tuple[np.ndarray, ...],
    initial_transaction_last: int = 0,
) -> list[ObservedControllerResult]:
    """Reproduce the traced controller sequence in its actual execution order.

    Controller delay/state is reset only before the first diagnostic image.
    It then carries through 128 row transactions, 64 column transactions, and
    into the next image.  This behavior was verified against all 10 captured
    images; it is not an assumed boundary convention.
    """

    state = int(initial_transaction_last)
    outputs: list[ObservedControllerResult] = []
    for raw_image in raw_images:
        raw = np.asarray(raw_image)
        if raw.shape != (INPUT_SIZE, INPUT_SIZE):
            raise ValueError(f"expected {(INPUT_SIZE, INPUT_SIZE)}, got {raw.shape}")
        if np.any(raw < 0) or np.any(raw > 255):
            raise ValueError("raw image contains a value outside uint8 range")
        raw = raw.astype(np.uint8, copy=True)

        row = np.empty((INPUT_SIZE, OUTPUT_SIZE), dtype=np.int16)
        for row_index in range(INPUT_SIZE):
            row[row_index] = _observed_centered_decimated_line(raw[row_index], state)
            state = int(raw[row_index, -1])

        ll = np.empty((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.int16)
        for column_index in range(OUTPUT_SIZE):
            ll[:, column_index] = _observed_centered_decimated_line(
                row[:, column_index], state
            )
            state = int(row[-1, column_index])

        preclip = ll.astype(np.int32) - INPUT_ZERO_POINT
        low_count = int(np.count_nonzero(preclip < -128))
        high_count = int(np.count_nonzero(preclip > 127))
        cnn_input = np.clip(preclip, -128, 127).astype(np.int8)
        outputs.append(
            ObservedControllerResult(
                raw=raw,
                row=row,
                ll=ll,
                preclip=preclip,
                cnn_input=cnn_input,
                packed_words=pack_int8_little_endian(cnn_input),
                low_saturation_count=low_count,
                high_saturation_count=high_count,
                final_transaction_last=state,
            )
        )
    return outputs


def boundary_masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Return implementation-derived interior/right/bottom/corner masks."""

    rows, cols = shape
    rr, cc = np.indices(shape)
    right = cc >= DECIMATED_TRAILING_BOUNDARY_START

    if rows == INPUT_SIZE:  # ROW checkpoint: only horizontal filtering is done.
        return {
            "interior": ~right,
            "right_boundary": right,
        }
    if rows == OUTPUT_SIZE and cols == OUTPUT_SIZE:
        bottom = rr >= DECIMATED_TRAILING_BOUNDARY_START
        return {
            "interior": ~(right | bottom),
            "right_boundary": right & ~bottom,
            "bottom_boundary": bottom & ~right,
            "corner": right & bottom,
        }
    raise ValueError(f"no released boundary definition for shape {shape}")
