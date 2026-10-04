# First-divergence firmware changes (manual application only)

The released firmware file is **not modified** by this diagnostic package. Apply
the following temporary changes manually to a diagnostic copy of:

`firmware/dbwfb2/main_full3200.c`

The snippets use the current released symbol names and memory flow. They dump
the input-BRAM readback, `Matrix_L`, `Subband_LL`, the actual packed CPU buffer,
and the final prediction. Diagnostic UART output invalidates latency results;
do not use timing summaries from this run.

## 1. Supply the diagnostic header

Copy:

`diagnostics/first_divergence/generated/first_divergence_vectors.h`

into the same Vitis application source directory as the diagnostic copy of
`main.c`.

### Location

Replace the complete block beginning at:

```c
#define TEST_PART 3
```

and ending after:

```c
#if N_TEST_IMAGES != 800
    #error "Each full-test part must contain exactly 800 images"
#endif
```

### Replacement

```c
#include "first_divergence_vectors.h"

#if !defined(FIRST_DIVERGENCE_DIAGNOSTIC)
    #error "The first-divergence diagnostic header is required"
#endif

#if N_TEST_IMAGES != 10
    #error "The frozen first-divergence set must contain exactly 10 images"
#endif
```

## 2. Add dump helpers

### Location

Immediately after these existing declarations:

```c
static int16_t  Matrix_L[128][64];
static int16_t  Subband_LL[64][64];
static uint32_t cnn_input[1024] __attribute__((aligned(64)));
```

### Addition

```c
/* Diagnostic-only, machine-readable checkpoint dumps. */
static void trace_raw_bram_row(int row)
{
    xil_printf("RAW_ROW,%d", row);
    for (int col = 0; col < 128; col++) {
        u32 word = Xil_In32(BRAM_IN_BASE + ((u32)col << 2));
        xil_printf(",%u", (unsigned int)word);
    }
    xil_printf("\r\n");
}

static void trace_matrix_i16(
    const char *stage,
    const int16_t *data,
    int rows,
    int cols
)
{
    xil_printf("%s_BEGIN,rows=%d,cols=%d,dtype=int16\r\n",
               stage, rows, cols);

    for (int row = 0; row < rows; row++) {
        xil_printf("%s_ROW,%d", stage, row);
        for (int col = 0; col < cols; col++) {
            xil_printf(",%d", (int)data[row * cols + col]);
        }
        xil_printf("\r\n");
    }

    xil_printf("%s_END\r\n", stage);
}

static void trace_cnn_input_i8(void)
{
    /* uint8_t/int8_t aliasing is permitted; on the Zynq PS this exposes the
     * same little-endian byte order packed explicitly by the firmware. */
    const int8_t *pixels = (const int8_t *)cnn_input;

    xil_printf("CNN_INPUT_BEGIN,rows=64,cols=64,dtype=int8\r\n");
    for (int row = 0; row < 64; row++) {
        xil_printf("CNN_INPUT_ROW,%d", row);
        for (int col = 0; col < 64; col++) {
            xil_printf(",%d", (int)pixels[row * 64 + col]);
        }
        xil_printf("\r\n");
    }
    xil_printf("CNN_INPUT_END\r\n");
}
```

## 3. Replace the part-specific banner lines

### Location

In the experiment banner, replace:

```c
    xil_printf(" Test scope        : full independent 3200-image test\r\n");
    xil_printf(" Current part      : %d / 3\r\n", TEST_PART);
    xil_printf(" Global test range : %d .. %d\r\n",
               TEST_PART_START,
               TEST_PART_START + N_TEST_IMAGES - 1);
```

### Replacement

```c
    xil_printf(" Test scope        : first-divergence diagnostic\r\n");
    xil_printf(" Images            : 5 FPGA/Python mismatches + 5 matches\r\n");
```

## 4. Begin each image trace

### Location

In the main inference loop, immediately after the existing pre-timing
human-readable `xil_printf(...)` block that begins with:

```c
        xil_printf(
            "[%4d/%4d] true=%-8s sw_ref=%-8s sw_alt=%-8s  ",
```

### Addition

```c
        xil_printf(
            "\r\nTRACE_IMAGE_BEGIN,index=%u,true=%u,python=%u\r\n",
            (unsigned int)test_global_indices[img],
            (unsigned int)true_label,
            (unsigned int)sw_ref
        );
```

## 5. Dump RAW input-BRAM readback

### Location A

Immediately before the existing row loop:

```c
        for (int r = 0; r < 128 && !dwt_err; r++) {
```

### Addition

```c
        xil_printf("RAW_BEGIN,rows=128,cols=128,dtype=uint32_readback\r\n");
```

### Location B

Inside that row loop, immediately after:

```c
            u64 t1 = gtimer_read();
            img_row_ps_write += (t1 - t0);
```

### Addition

```c
            /* Read back the 128 words actually written for this row before
             * the PL transaction begins and before this BRAM row is reused. */
            trace_raw_bram_row(r);
```

### Location C

Immediately after the closing brace of the complete row loop and immediately
before:

```c
        if (dwt_err) {
```

### Addition

```c
        xil_printf("RAW_END\r\n");
```

## 6. Dump `Matrix_L` after the row pass

### Location

Immediately after the first successful row-pass error check:

```c
        if (dwt_err) {
            hw_err_count++;
            continue;
        }
```

and before the `DWT PASS 2: COLUMNS` comment.

### Addition

```c
        trace_matrix_i16("ROW", &Matrix_L[0][0], 128, 64);
```

## 7. Dump `Subband_LL` after the column pass

### Location

Immediately after the second successful `if (dwt_err) ... continue;` block and
before the `DEPLOYMENT INPUT QUANTIZATION` comment.

### Addition

```c
        trace_matrix_i16("LL", &Subband_LL[0][0], 64, 64);
```

## 8. Dump the actual packed CNN input

### Location

Immediately after:

```c
        u64 t1 = gtimer_read();
        img_quant_pack = (t1 - t0);
```

and before the `D-CACHE FLUSH` comment.

### Addition

```c
        trace_cnn_input_i8();
```

## 9. Use the noncontiguous global index in `PRED`

### Location

In the existing machine-readable `PRED` call, replace only:

```c
            TEST_PART_START + img,
```

with:

```c
            (int)test_global_indices[img],
```

## 10. Add result/end markers

### Location

Immediately after the existing complete `PRED,...` `xil_printf(...)` call.

### Addition

```c
        xil_printf(
            "TRACE_RESULT,index=%u,true=%u,python=%u,fpga=%u\r\n",
            (unsigned int)test_global_indices[img],
            (unsigned int)true_label,
            (unsigned int)sw_ref,
            (unsigned int)pred
        );
        xil_printf(
            "TRACE_IMAGE_END,index=%u\r\n",
            (unsigned int)test_global_indices[img]
        );
```

## 11. Replace the part summary

### Location

Replace the complete existing block:

```c
    xil_printf(
        "SUMMARY_PART,%d,%d,%d,%d\r\n",
        TEST_PART,
        TEST_PART_START,
        TEST_PART_START + N_TEST_IMAGES - 1,
        N_TEST_IMAGES
    );
```

### Replacement

```c
    xil_printf("SUMMARY_DIAGNOSTIC,%d\r\n", N_TEST_IMAGES);
```

## Expected UART markers

Each successful image must contain, in order:

```text
TRACE_IMAGE_BEGIN,index=...
RAW_BEGIN,rows=128,cols=128,dtype=uint32_readback
RAW_ROW,...
RAW_END
ROW_BEGIN,rows=128,cols=64,dtype=int16
ROW_ROW,...
ROW_END
LL_BEGIN,rows=64,cols=64,dtype=int16
LL_ROW,...
LL_END
CNN_INPUT_BEGIN,rows=64,cols=64,dtype=int8
CNN_INPUT_ROW,...
CNN_INPUT_END
TRACE_RESULT,index=...,true=...,python=...,fpga=...
TRACE_IMAGE_END,index=...
```

If a timeout occurs, preserve the incomplete log. The parser will reject it
rather than silently treating a partial trace as evidence.
