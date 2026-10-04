// =============================================================================
// File   : main_avgpool_500_70sec.c
//
// Purpose:
//   Same-flow hardware baseline for a 2x2 AvgPool RTL front-end followed by
//   the SAME compact INT8 CNN architecture used in the DBWFB2 experiment.
//
// Frozen AvgPool deployment configuration:
//   - 4 STL-10 classes: airplane, car, dog, ship
//   - Frozen board subset: 500 images, 125/class
//   - AvgPool deployment model selected on VALIDATION only: seed 42
//   - Input quantization gain: 1
//   - CNN C1_SHIFT: 9
//   - CNN C2_SHIFT: 7
//
// FAIR-HARDWARE-COMPARISON RULE:
//   Keep the same board, 100 MHz clock, PS/AXI/BRAM/GPIO structure, CNN
//   architecture, HLS directives, board-500 images, and timing methodology.
//   The preprocessing RTL and its validation-selected weights/shifts are the
//   method-specific parts.
//
// AVGPOOL RTL NUMERICAL FORMAT:
//   avgpool_2x2_bram_ctrl_exact.v intentionally avoids intermediate rounding.
//
//   Row pass:
//       horizontal_sum = x[2k] + x[2k+1]              (0..510)
//
//   Column pass:
//       sum4 = horizontal_sum_top + horizontal_sum_bottom
//            = p00 + p01 + p10 + p11                  (0..1020)
//
//   The output BRAM therefore stores the exact FOUR-PIXEL SUM, with an implicit
//   divide-by-4.  The PS performs the final deployment quantization as:
//
//       q = RNE((sum4 - 508) / 4)
//
//   because software deployment computes:
//
//       q = np.rint((AvgPool - 127) * gain), gain=1
//         = np.rint(sum4/4 - 127)
//         = np.rint((sum4 - 508)/4)
//
//   RNE means round-to-nearest, ties-to-even, matching numpy.rint for all
//   possible integer sum4 values.
//
// SAME-FLOW TWO-PASS MEMORY SCHEDULE:
//   The AvgPool RTL writes pair results at BRAM indices 0,2,4,...,126.
//   Therefore the existing PS stride-2 output-BRAM reads are retained:
//
//   Row pass:
//      1) PS writes 128 raw uint8 pixels to input BRAM
//      2) AvgPool RTL forms 64 exact horizontal pair sums
//      3) PS reads output indices 0,2,...,126 -> Matrix_HSUM[128][64]
//
//   Column pass:
//      4) PS writes each 128-element Matrix_HSUM column to input BRAM
//      5) AvgPool RTL forms 64 exact vertical pair sums
//      6) PS reads output indices 0,2,...,126 -> Sum4[64][64]
//
//   CNN stage:
//      7) exact AvgPool-to-INT8 RNE quantization + packing
//      8) D-cache flush
//      9) AXI-Lite CNN input-pointer setup
//     10) CNN start -> done -> class-result read
//
// TIMING:
//   ARM Cortex-A9 global-timer values are GLOBAL-TIMER TICKS (~333 MHz),
//   NOT 100 MHz FPGA/PL cycles.  Reported milliseconds exclude UART printing
//   and exclude the startup delay.
//
// SINGLE-USB-CABLE WORKFLOW:
//   A 70-second startup delay occurs AFTER initialization and BEFORE experiment
//   printing/timing, allowing one USB cable to be moved from JTAG to UART.
//
// HEADER:
//   test_vectors_raw_500_avgpool.h must contain:
//      N_TEST_IMAGES
//      test_labels[]
//      expected_labels[]       : frozen AvgPool Python true-INT8 reference
//      test_images_raw[][]     : frozen 128x128 uint8 images
//
// IMPORTANT:
//   This file is intended specifically for avgpool_2x2_bram_ctrl_exact.v.
// =============================================================================

#include <stdio.h>
#include <stdint.h>

#include "xparameters.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xcnn_accel.h"

// Frozen 500-image AvgPool deployment header.
#include "test_vectors_raw_500_avgpool.h"


/* ============================================================================
 * ARM Cortex-A9 Global Timer
 * ----------------------------------------------------------------------------
 * Zynq-7000 global timer base address is fixed at 0xF8F00200.
 * Timer increments at CPU clock / 2 (approximately 333 MHz on ZedBoard).
 *
 * DO NOT call these values FPGA cycles in the paper.
 * ========================================================================== */
#define GTIMER_BASE   0xF8F00200UL
#define GTIMER_LO     (*(volatile u32*)(GTIMER_BASE + 0x00U))
#define GTIMER_HI     (*(volatile u32*)(GTIMER_BASE + 0x04U))
#define GTIMER_CTRL   (*(volatile u32*)(GTIMER_BASE + 0x08U))
#define GTIMER_CPS    (XPAR_CPU_CORE_CLOCK_FREQ_HZ / 2U)


/* Read the 64-bit global timer safely across the two 32-bit registers. */
static u64 gtimer_read(void)
{
    u32 hi, lo, hi2;

    do {
        hi  = GTIMER_HI;
        lo  = GTIMER_LO;
        hi2 = GTIMER_HI;
    } while (hi != hi2);

    return ((u64)hi << 32) | (u64)lo;
}


/* Convert ARM global-timer ticks to microseconds.
 * Safe for the short per-image latencies measured here. */
static u32 ticks_to_us(u64 ticks, u64 ticks_per_sec)
{
    return (u32)((ticks * 1000000ULL) / ticks_per_sec);
}


/* Integer-only percentage helper.
 * Returns percentage x10:
 *   pct10(1,2) = 500 -> print as 50.0%.
 */
static int pct10(int num, int den)
{
    return (den > 0) ? (num * 1000 / den) : 0;
}


/* Busy-wait startup delay for the single-cable JTAG -> UART workflow.
 *
 * This function is called before the experiment begins, so it does not enter
 * any inference-latency measurement.
 */
static void startup_delay_seconds(u32 seconds)
{
    u64 start = gtimer_read();
    u64 wait_ticks = (u64)GTIMER_CPS * (u64)seconds;

    while ((gtimer_read() - start) < wait_ticks) {
        /* intentional busy wait */
    }
}


/* ============================================================================
 * Classes
 * ========================================================================== */
static const char *CLASS_NAMES[4] = {
    "airplane",
    "car",
    "dog",
    "ship"
};


/* ============================================================================
 * Memory-mapped hardware addresses
 *
 * These are the same BRAM and GPIO peripherals used by the validated original
 * design. The preprocessing RTL itself is NOT an AXI-Lite slave; PS control is
 * performed through AXI GPIO and data movement through the AXI BRAM controllers.
 * ========================================================================== */
#define BRAM_IN_BASE   XPAR_AXI_BRAM_CTRL_0_BASEADDR
#define BRAM_OUT_BASE  XPAR_AXI_BRAM_CTRL_1_BASEADDR
#define GPIO_BASE      XPAR_AXI_GPIO_0_BASEADDR

#define GPIO_CH1_DATA  (GPIO_BASE + 0x00U)
#define GPIO_CH1_TRI   (GPIO_BASE + 0x04U)
#define GPIO_CH2_DATA  (GPIO_BASE + 0x08U)
#define GPIO_CH2_TRI   (GPIO_BASE + 0x0CU)


/* Existing working polling limits.
 *
 * These are software-loop limits, NOT cycle counts.
 * Hardware error count is reported at the end of the run.
 */
#define CNN_TIMEOUT  2000000
#define AVGPOOL_TIMEOUT   500000


/* 70 seconds gives enough time to move one USB cable JTAG -> UART. */
#define STARTUP_DELAY_SEC  70U


/* ============================================================================
 * Buffers
 *
 * Matrix_HSUM:
 *   128 rows x 64 horizontal pair sums.
 *   Each value is exact: 0..510.
 *
 * Sum4:
 *   64 x 64 exact 2x2 four-pixel sums.
 *   Each value is exact: 0..1020, representing AvgPool = Sum4 / 4.
 *
 * cnn_input:
 *   4096 signed INT8 pixels packed four bytes per uint32
 *   = 1024 x 32-bit words.
 * ========================================================================== */
static uint16_t Matrix_HSUM[128][64];
static uint16_t Sum4[64][64];
static uint32_t cnn_input[1024] __attribute__((aligned(64)));


/* ============================================================================
 * Start one 1-D AvgPool pair-sum hardware transaction.
 *
 * GPIO channel 1:
 *   bit 0      : start
 *   remaining  : pixel_count << 1
 *
 * GPIO channel 2:
 *   bit 0      : done
 *
 * Returns:
 *    0 -> success
 *   -1 -> timeout
 *
 * For pixel_count=128 the RTL consumes adjacent pairs and writes 64 sums at
 * output BRAM sample indices 0,2,4,...,126.  The PS then performs the same
 * stride-2 reads used by the DBWFB2 same-flow implementation.
 * ========================================================================== */
static int run_hw_avgpool(uint32_t pixel_count)
{
    Xil_Out32(GPIO_CH1_TRI,  0x00000000u);
    Xil_Out32(GPIO_CH2_TRI,  0xFFFFFFFFu);

    Xil_Out32(GPIO_CH1_DATA, (pixel_count << 1) | 0x1u);

    int t = AVGPOOL_TIMEOUT;

    while ((Xil_In32(GPIO_CH2_DATA) & 0x1u) == 0u) {
        if (--t <= 0) {
            Xil_Out32(GPIO_CH1_DATA, (pixel_count << 1) | 0x0u);
            return -1;
        }
    }

    Xil_Out32(GPIO_CH1_DATA, (pixel_count << 1) | 0x0u);
    return 0;
}


/* ============================================================================
 * Signed round-to-nearest-even division by 4.
 *
 * Exactly matches numpy.rint(n / 4.0) for integer n.
 *
 * Here:
 *     n = Sum4 - 508
 *
 * Sum4 is 0..1020, therefore n is -508..512.
 * ========================================================================== */
static int32_t round_div4_rne(int32_t n)
{
    /* q = floor(n/4), rem in {0,1,2,3}. */
    int32_t q;

    if (n >= 0) {
        q = n / 4;
    }
    else {
        q = - ((-n + 3) / 4);
    }

    int32_t rem = n - (q * 4);

    if (rem < 2) {
        return q;
    }

    if (rem > 2) {
        return q + 1;
    }

    /* Exact half-way case: choose the even integer. */
    return ((q & 1) == 0) ? q : (q + 1);
}


/* ============================================================================
 * main
 * ========================================================================== */
int main(void)
{
    /* Ensure ARM global timer is enabled. */
    GTIMER_CTRL = GTIMER_CTRL | 0x1U;


    /* ------------------------------------------------------------------------
     * Initialize HLS CNN driver
     * --------------------------------------------------------------------- */
    XCnn_accel cnn;
    int init_status;

#ifdef XPAR_CNN_ACCEL_0_DEVICE_ID
    init_status = XCnn_accel_Initialize(
        &cnn,
        XPAR_CNN_ACCEL_0_DEVICE_ID
    );
#elif defined(XPAR_XCNN_ACCEL_0_DEVICE_ID)
    init_status = XCnn_accel_Initialize(
        &cnn,
        XPAR_XCNN_ACCEL_0_DEVICE_ID
    );
#else
    init_status = XCnn_accel_Initialize(
        &cnn,
        0
    );
#endif

    if (init_status != XST_SUCCESS) {
        /* If UART is not attached yet this message may not be visible,
         * but return immediately rather than running with an invalid driver. */
        xil_printf("[FATAL] CNN Driver Initialization Failed!\r\n");
        return -1;
    }


    /* ------------------------------------------------------------------------
     * CNN AXI-Lite address assignment
     *
     * Current HLS IP exposes two AXI-Lite slave interfaces:
     *
     *   Control_BaseAddress = s_axi_control
     *                        HLS ap_start / ap_done / class output control
     *
     *   Ctrl_BaseAddress    = s_axi_CTRL
     *                        secondary HLS control/data interface
     *
     * These addresses must match the CURRENT Vivado Address Editor.
     * They are retained from the validated working design because the interface
     * topology has not changed.
     * --------------------------------------------------------------------- */
    cnn.Control_BaseAddress = 0x40020000u;
    cnn.Ctrl_BaseAddress    = 0x40010000u;


    /* ------------------------------------------------------------------------
     * Single-cable delay
     *
     * Program/run through JTAG, then keep board power ON, move the cable to
     * UART, open the serial terminal, and wait for this delay to finish.
     * --------------------------------------------------------------------- */
    startup_delay_seconds(STARTUP_DELAY_SEC);


    /* ------------------------------------------------------------------------
     * Experiment banner
     * --------------------------------------------------------------------- */
    xil_printf("\r\n");
    xil_printf("============================================================\r\n");
    xil_printf(" AVGPOOL SAME-FLOW INT8 CNN FPGA VALIDATION\r\n");
    xil_printf(" Raw 128x128 -> exact 2x2 AvgPool RTL -> INT8 CNN\r\n");
    xil_printf(" Images            : %d\r\n", N_TEST_IMAGES);
    xil_printf(" Input gain        : 1\r\n");
    xil_printf(" CNN C1_SHIFT      : 9\r\n");
    xil_printf(" CNN C2_SHIFT      : 7  (compiled inside AvgPool HLS IP)\r\n");
    xil_printf(" Board subset      : frozen random stratified 500 (125/class)\r\n");
    xil_printf(" AvgPool RTL format: exact four-pixel sum / implicit divide-by-4\r\n");
    xil_printf("============================================================\r\n");


    /* =========================================================================
     * Accuracy / agreement counters
     * ====================================================================== */

    int class_correct[4] = {0, 0, 0, 0};
    int class_total[4]   = {0, 0, 0, 0};

    int total_correct     = 0;
    int total_completed   = 0;
    int hw_err_count      = 0;

    /* Prediction-level agreement with the frozen AvgPool Python
     * true-INT8 software reference in expected_labels[]. */
    int sw_ref_match = 0;


    /* =========================================================================
     * Global timing accumulators
     *
     * All values are ARM global-timer TICKS.
     *
     * To avoid partial accounting on timeout/error paths, every image first
     * accumulates into local per-image counters. Global counters are updated
     * only after the complete AvgPool + CNN inference succeeds.
     * ====================================================================== */

    u64 total_ticks = 0ULL;

    u64 row_ps_write_ticks = 0ULL;
    u64 row_pool_wall_ticks = 0ULL;
    u64 row_ps_read_ticks  = 0ULL;

    u64 col_ps_write_ticks = 0ULL;
    u64 col_pool_wall_ticks = 0ULL;
    u64 col_ps_read_ticks  = 0ULL;

    u64 quant_pack_ticks   = 0ULL;
    u64 cache_flush_ticks  = 0ULL;
    u64 cnn_setup_ticks    = 0ULL;
    u64 cnn_exec_ticks     = 0ULL;


    /* =========================================================================
     * Main inference loop
     * ====================================================================== */
    for (int img = 0; img < N_TEST_IMAGES; img++) {

        const uint8_t true_label = test_labels[img];
        const uint8_t sw_ref     = expected_labels[img];

        if (true_label < 4u) {
            class_total[true_label]++;
        }


        /* --------------------------------------------------------------------
         * Per-image local timing accumulators.
         * These are committed to the global statistics only on success.
         * ----------------------------------------------------------------- */
        u64 img_row_ps_write = 0ULL;
        u64 img_row_pool_wall = 0ULL;
        u64 img_row_ps_read  = 0ULL;

        u64 img_col_ps_write = 0ULL;
        u64 img_col_pool_wall = 0ULL;
        u64 img_col_ps_read  = 0ULL;

        u64 img_quant_pack   = 0ULL;
        u64 img_cache_flush  = 0ULL;
        u64 img_cnn_setup    = 0ULL;
        u64 img_cnn_exec     = 0ULL;


        /* Print BEFORE timing so UART output is excluded from inference time. */
        xil_printf(
            "[%4d/%4d] true=%-8s sw_ref=%-8s  ",
            img + 1,
            N_TEST_IMAGES,
            (true_label < 4u) ? CLASS_NAMES[true_label] : "???",
            (sw_ref < 4u)     ? CLASS_NAMES[sw_ref]     : "???"
        );


        /* Entire successful preprocessing + CNN wall time. */
        u64 t_image_start = gtimer_read();

        int pool_err = 0;


        /* ====================================================================
         * AVGPOOL PASS 1: ROWS
         *
         * For each of the 128 rows:
         *   A) PS writes 128 raw uint8 pixels to input BRAM.
         *   B) AvgPool RTL forms exact adjacent pair sums:
         *          x[0]+x[1], x[2]+x[3], ..., x[126]+x[127]
         *   C) RTL stores these at output sample indices 0,2,...,126.
         *   D) PS performs the existing stride-2 reads into Matrix_HSUM.
         *
         * Matrix_HSUM range: 0..510.
         * ================================================================= */
        for (int r = 0; r < 128 && !pool_err; r++) {

            u64 t0 = gtimer_read();

            for (int c = 0; c < 128; c++) {
                Xil_Out32(
                    BRAM_IN_BASE + ((u32)c << 2),
                    (uint32_t)test_images_raw[img][r * 128 + c]
                );
            }

            u64 t1 = gtimer_read();
            img_row_ps_write += (t1 - t0);


            t0 = gtimer_read();

            if (run_hw_avgpool(128) != 0) {
                t1 = gtimer_read();
                img_row_pool_wall += (t1 - t0);

                xil_printf("[AVGPOOL-ROW-TIMEOUT]\r\n");
                pool_err = 1;
                break;
            }

            t1 = gtimer_read();
            img_row_pool_wall += (t1 - t0);


            t0 = gtimer_read();

            for (int c = 0; c < 64; c++) {
                uint32_t w = Xil_In32(
                    BRAM_OUT_BASE + (((u32)c * 2U) << 2)
                );

                Matrix_HSUM[r][c] = (uint16_t)(w & 0xFFFFu);

                /* Legal horizontal-pair range for uint8 pixels. */
                if (Matrix_HSUM[r][c] > 510u) {
                    pool_err = 1;
                    break;
                }
            }

            t1 = gtimer_read();
            img_row_ps_read += (t1 - t0);

            if (pool_err) {
                xil_printf("[AVGPOOL-ROW-RANGE-ERROR]\r\n");
                break;
            }
        }

        if (pool_err) {
            hw_err_count++;
            continue;
        }


        /* ====================================================================
         * AVGPOOL PASS 2: COLUMNS
         *
         * For each of the 64 horizontal-output columns:
         *   A) PS writes 128 exact horizontal pair sums to input BRAM.
         *   B) AvgPool RTL adds adjacent rows, producing exact 2x2 sums:
         *
         *          Sum4 = p00 + p01 + p10 + p11
         *
         *   C) PS stride-2 reads 64 results into Sum4[64][64].
         *
         * Sum4 range: 0..1020.
         * AvgPool value is represented exactly as Sum4 / 4.
         * ================================================================= */
        for (int c = 0; c < 64 && !pool_err; c++) {

            u64 t0 = gtimer_read();

            for (int r = 0; r < 128; r++) {
                Xil_Out32(
                    BRAM_IN_BASE + ((u32)r << 2),
                    (uint32_t)Matrix_HSUM[r][c]
                );
            }

            u64 t1 = gtimer_read();
            img_col_ps_write += (t1 - t0);


            t0 = gtimer_read();

            if (run_hw_avgpool(128) != 0) {
                t1 = gtimer_read();
                img_col_pool_wall += (t1 - t0);

                xil_printf("[AVGPOOL-COL-TIMEOUT]\r\n");
                pool_err = 1;
                break;
            }

            t1 = gtimer_read();
            img_col_pool_wall += (t1 - t0);


            t0 = gtimer_read();

            for (int r = 0; r < 64; r++) {
                uint32_t w = Xil_In32(
                    BRAM_OUT_BASE + (((u32)r * 2U) << 2)
                );

                Sum4[r][c] = (uint16_t)(w & 0xFFFFu);

                if (Sum4[r][c] > 1020u) {
                    pool_err = 1;
                    break;
                }
            }

            t1 = gtimer_read();
            img_col_ps_read += (t1 - t0);

            if (pool_err) {
                xil_printf("[AVGPOOL-COL-RANGE-ERROR]\r\n");
                break;
            }
        }

        if (pool_err) {
            hw_err_count++;
            continue;
        }


        /* ====================================================================
         * AVGPOOL -> TRUE INT8 INPUT QUANTIZATION
         *
         * Software deployment:
         *
         *      AvgPool = (p00+p01+p10+p11) / 4
         *      q       = np.rint(AvgPool - 127)
         *      q       = clip(q, -128, 127)
         *
         * RTL retains the exact four-pixel sum Sum4, so firmware computes:
         *
         *      q = RNE((Sum4 - 508) / 4)
         *
         * where RNE = round-to-nearest, ties-to-even.
         *
         * This avoids intermediate rounding and exactly reproduces numpy.rint
         * for the integer-valued raw 128x128 image supplied to the FPGA.
         *
         * Four signed INT8 values are packed into one uint32:
         *      bits  7:0   = pixel 0
         *      bits 15:8   = pixel 1
         *      bits 23:16  = pixel 2
         *      bits 31:24  = pixel 3
         * ================================================================= */
        u64 t0 = gtimer_read();

        for (int r = 0; r < 64; r++) {

            for (int c = 0; c < 64; c += 4) {

                uint8_t p[4];

                for (int i = 0; i < 4; i++) {

                    int32_t numerator =
                        (int32_t)Sum4[r][c + i] - 508;

                    int32_t val =
                        round_div4_rne(numerator);

                    if (val > 127) {
                        val = 127;
                    }
                    else if (val < -128) {
                        val = -128;
                    }

                    p[i] = (uint8_t)(val & 0xFF);
                }

                cnn_input[(r * 16) + (c >> 2)] =
                      (u32)p[0]
                    | ((u32)p[1] << 8)
                    | ((u32)p[2] << 16)
                    | ((u32)p[3] << 24);
            }
        }

        u64 t1 = gtimer_read();
        img_quant_pack = (t1 - t0);


        /* ====================================================================
         * D-CACHE FLUSH
         *
         * cnn_input is written by the ARM and read by the HLS accelerator
         * through its AXI master interface. Flush the cache so DDR contains the
         * newly packed data visible to the PL.
         * ================================================================= */
        t0 = gtimer_read();

        Xil_DCacheFlushRange(
            (INTPTR)cnn_input,
            sizeof(cnn_input)
        );

        t1 = gtimer_read();
        img_cache_flush = (t1 - t0);


        /* ====================================================================
         * CNN AXI-LITE SETUP
         *
         * Program the input-buffer address separately from accelerator execution
         * so the software-control overhead is visible.
         * ================================================================= */
        t0 = gtimer_read();

        XCnn_accel_Set_in_bram(
            &cnn,
            (u64)(UINTPTR)cnn_input
        );

        t1 = gtimer_read();
        img_cnn_setup = (t1 - t0);


        /* ====================================================================
         * CNN EXECUTION WALL TIME
         *
         * Measures:
         *      XCnn_accel_Start()
         *          ->
         *      PS polling XCnn_accel_IsDone()
         *          ->
         *      XCnn_accel_Get_cls_out()
         *
         * This is ARM-observed start-to-result elapsed time.
         *
         * It should NOT be called "measured 100 MHz PL cycles".
         *
         * Independent HLS synthesis result for this frozen CNN:
         *      latency = 6,329,660 HLS cycles
         *      target  = 100 MHz
         *      nominal HLS latency ~= 63.30 ms
         * ================================================================= */
        t0 = gtimer_read();

        XCnn_accel_Start(&cnn);

        int to = CNN_TIMEOUT;

        while ((to > 0) && !XCnn_accel_IsDone(&cnn)) {
            to--;
        }

        if (to == 0) {
            xil_printf("[CNN-TIMEOUT]\r\n");
            hw_err_count++;
            continue;
        }

        u32 pred = XCnn_accel_Get_cls_out(&cnn);

        t1 = gtimer_read();
        img_cnn_exec = (t1 - t0);


        /* Entire successful image processing interval. */
        u64 t_image_end = gtimer_read();
        u64 img_total = t_image_end - t_image_start;


        /* ====================================================================
         * Commit timing statistics ONLY after successful end-to-end inference.
         * ================================================================= */
        total_ticks += img_total;

        row_ps_write_ticks += img_row_ps_write;
        row_pool_wall_ticks += img_row_pool_wall;
        row_ps_read_ticks  += img_row_ps_read;

        col_ps_write_ticks += img_col_ps_write;
        col_pool_wall_ticks += img_col_pool_wall;
        col_ps_read_ticks  += img_col_ps_read;

        quant_pack_ticks   += img_quant_pack;
        cache_flush_ticks  += img_cache_flush;
        cnn_setup_ticks    += img_cnn_setup;
        cnn_exec_ticks     += img_cnn_exec;

        total_completed++;


        /* ====================================================================
         * Score classification and software-reference agreement.
         * ================================================================= */
        int ok_true = ((int)pred == (int)true_label);
        int ok_sw   = ((int)pred == (int)sw_ref);

        if (ok_true) {

            total_correct++;

            if (true_label < 4u) {
                class_correct[true_label]++;
            }
        }

        if (ok_sw) {
            sw_ref_match++;
        }



        /* Human-readable line. */
        xil_printf(
            "-> %-8s [true:%s | SW:%s]\r\n",
            (pred < 4u) ? CLASS_NAMES[pred] : "???",
            ok_true ? "PASS"  : "FAIL",
            ok_sw   ? "MATCH" : "MISMATCH"
        );


        /* Machine-readable prediction line:
         *   PRED,board_order,true_label,sw_ref,pred
         */
        xil_printf(
            "PRED,%d,%d,%d,%u\r\n",
            img,
            (int)true_label,
            (int)sw_ref,
            pred
        );
    }


    /* =========================================================================
     * ACCURACY / AGREEMENT SUMMARY
     * ====================================================================== */
    xil_printf("\r\n");
    xil_printf("============================================================\r\n");
    xil_printf(" ACCURACY / AGREEMENT SUMMARY\r\n");
    xil_printf("============================================================\r\n");
    xil_printf(" Scheduled images : %d\r\n", N_TEST_IMAGES);
    xil_printf(" Completed images : %d\r\n", total_completed);
    xil_printf(" HW errors        : %d\r\n", hw_err_count);
    xil_printf("------------------------------------------------------------\r\n");


    /* Per-class denominator remains the frozen 125/class scheduled set.
     * Therefore a timeout would transparently reduce end-to-end class success. */
    for (int c = 0; c < 4; c++) {

        int p = pct10(
            class_correct[c],
            class_total[c]
        );

        xil_printf(
            " %-10s : %4d / %4d = %3d.%d%%\r\n",
            CLASS_NAMES[c],
            class_correct[c],
            class_total[c],
            p / 10,
            p % 10
        );
    }


    /* Overall success on the complete scheduled 500-image set.
     * If there are no HW errors this is identical to completed-only accuracy. */
    int p_all = pct10(
        total_correct,
        N_TEST_IMAGES
    );

    int p_completed = pct10(
        total_correct,
        total_completed
    );

    int p_sw = pct10(
        sw_ref_match,
        total_completed
    );



    xil_printf("------------------------------------------------------------\r\n");

    xil_printf(
        " vs TRUE (all scheduled) : %4d / %4d = %3d.%d%%\r\n",
        total_correct,
        N_TEST_IMAGES,
        p_all / 10,
        p_all % 10
    );

    xil_printf(
        " vs TRUE (completed)     : %4d / %4d = %3d.%d%%\r\n",
        total_correct,
        total_completed,
        p_completed / 10,
        p_completed % 10
    );

    xil_printf(
        " vs SW INT8 reference    : %4d / %4d = %3d.%d%%\r\n",
        sw_ref_match,
        total_completed,
        p_sw / 10,
        p_sw % 10
    );



    /* Machine-readable summary line. */
    xil_printf(
        "SUMMARY_ACC_AVGPOOL,%d,%d,%d,%d,%d\r\n",
        N_TEST_IMAGES,
        total_completed,
        total_correct,
        sw_ref_match,
        hw_err_count
    );


    /* =========================================================================
     * LATENCY SUMMARY
     * ====================================================================== */
    xil_printf("\r\n");
    xil_printf("============================================================\r\n");
    xil_printf(" LATENCY SUMMARY (%d successful images)\r\n", total_completed);
    xil_printf("============================================================\r\n");


    if (total_completed > 0) {

        const u64 cps = (u64)GTIMER_CPS;


        /* --------------------------------------------------------------------
         * Average ARM global-timer ticks per successful image.
         * ----------------------------------------------------------------- */
        u64 avg_total_ticks =
            total_ticks / (u64)total_completed;

        u64 avg_row_write_ticks =
            row_ps_write_ticks / (u64)total_completed;

        u64 avg_row_pool_ticks =
            row_pool_wall_ticks / (u64)total_completed;

        u64 avg_row_read_ticks =
            row_ps_read_ticks / (u64)total_completed;

        u64 avg_col_write_ticks =
            col_ps_write_ticks / (u64)total_completed;

        u64 avg_col_pool_ticks =
            col_pool_wall_ticks / (u64)total_completed;

        u64 avg_col_read_ticks =
            col_ps_read_ticks / (u64)total_completed;

        u64 avg_quant_ticks =
            quant_pack_ticks / (u64)total_completed;

        u64 avg_flush_ticks =
            cache_flush_ticks / (u64)total_completed;

        u64 avg_cnn_setup_ticks =
            cnn_setup_ticks / (u64)total_completed;

        u64 avg_cnn_exec_ticks =
            cnn_exec_ticks / (u64)total_completed;


        /* --------------------------------------------------------------------
         * Convert each average to microseconds.
         * ----------------------------------------------------------------- */
        u32 total_us =
            ticks_to_us(avg_total_ticks, cps);

        u32 row_write_us =
            ticks_to_us(avg_row_write_ticks, cps);

        u32 row_pool_us =
            ticks_to_us(avg_row_pool_ticks, cps);

        u32 row_read_us =
            ticks_to_us(avg_row_read_ticks, cps);

        u32 col_write_us =
            ticks_to_us(avg_col_write_ticks, cps);

        u32 col_pool_us =
            ticks_to_us(avg_col_pool_ticks, cps);

        u32 col_read_us =
            ticks_to_us(avg_col_read_ticks, cps);

        u32 quant_us =
            ticks_to_us(avg_quant_ticks, cps);

        u32 flush_us =
            ticks_to_us(avg_flush_ticks, cps);

        u32 cnn_setup_us =
            ticks_to_us(avg_cnn_setup_ticks, cps);

        u32 cnn_exec_us =
            ticks_to_us(avg_cnn_exec_ticks, cps);


        /* Aggregate preprocessing and CNN software-visible stage times. */
        u32 row_total_us =
            row_write_us +
            row_pool_us +
            row_read_us;

        u32 col_total_us =
            col_write_us +
            col_pool_us +
            col_read_us;

        u32 avgpool_system_total_us =
            row_total_us +
            col_total_us;

        u32 cnn_system_total_us =
            quant_us +
            flush_us +
            cnn_setup_us +
            cnn_exec_us;


        /* Sum of explicitly instrumented substages.
         * Difference from end-to-end time is timer/control/loop residual. */
        u32 accounted_us =
            avgpool_system_total_us +
            cnn_system_total_us;

        u32 residual_us =
            (total_us > accounted_us)
                ? (total_us - accounted_us)
                : 0U;


        /* Throughput from measured end-to-end time. */
        u32 fps_i =
            (total_us > 0U)
                ? (u32)(1000000UL / total_us)
                : 0U;

        u32 fps_f =
            (total_us > 0U)
                ? (u32)((10000000UL / total_us) % 10U)
                : 0U;


        /* Total accumulated successful-image inference time. */
        u32 measured_total_ms =
            (u32)((total_ticks * 1000ULL) / cps);


        /* --------------------------------------------------------------------
         * Print helper macro:
         * Convert integer microseconds into X.YY ms without floating point.
         *
         * Example:
         *     63295 us -> 63.29 ms
         * ----------------------------------------------------------------- */
#define PRINT_TIME_MS(label, value_us)                                      \
        do {                                                                \
            u32 _us = (value_us);                                           \
            xil_printf(                                                     \
                " %-30s : %u.%02u ms  (%u us)\r\n",                        \
                (label),                                                    \
                _us / 1000U,                                                \
                (_us % 1000U) / 10U,                                       \
                _us                                                        \
            );                                                              \
        } while (0)


        xil_printf("------------------------------------------------------------\r\n");
        PRINT_TIME_MS("End-to-end", total_us);
        xil_printf(" Throughput                     : %u.%u FPS\r\n", fps_i, fps_f);
        xil_printf(
            " Accumulated measured time       : %u ms\r\n",
            measured_total_ms
        );


        xil_printf("\r\n");
        xil_printf(" ROW PASS\r\n");
        xil_printf("------------------------------------------------------------\r\n");
        PRINT_TIME_MS("PS input-BRAM writes", row_write_us);
        PRINT_TIME_MS("AvgPool start->done wall", row_pool_us);
        PRINT_TIME_MS("PS stride-2 BRAM reads", row_read_us);
        PRINT_TIME_MS("Row-pass system total", row_total_us);


        xil_printf("\r\n");
        xil_printf(" COLUMN PASS\r\n");
        xil_printf("------------------------------------------------------------\r\n");
        PRINT_TIME_MS("PS transpose/reorder + writes", col_write_us);
        PRINT_TIME_MS("AvgPool start->done wall", col_pool_us);
        PRINT_TIME_MS("PS stride-2 BRAM reads", col_read_us);
        PRINT_TIME_MS("Column-pass system total", col_total_us);


        xil_printf("\r\n");
        xil_printf(" PREPROCESSING TOTAL\r\n");
        xil_printf("------------------------------------------------------------\r\n");
        PRINT_TIME_MS("AvgPool system total", avgpool_system_total_us);


        xil_printf("\r\n");
        xil_printf(" CNN PATH\r\n");
        xil_printf("------------------------------------------------------------\r\n");
        PRINT_TIME_MS("INT8 quantization + packing", quant_us);
        PRINT_TIME_MS("D-cache flush", flush_us);
        PRINT_TIME_MS("CNN pointer/control setup", cnn_setup_us);
        PRINT_TIME_MS("CNN start->done/result wall", cnn_exec_us);
        PRINT_TIME_MS("CNN system total", cnn_system_total_us);


        xil_printf("\r\n");
        xil_printf(" TIMING ACCOUNTING\r\n");
        xil_printf("------------------------------------------------------------\r\n");
        PRINT_TIME_MS("Explicitly instrumented sum", accounted_us);
        PRINT_TIME_MS("Residual timer/control overhead", residual_us);


        /* --------------------------------------------------------------------
         * ARM GLOBAL-TIMER TICKS
         *
         * These are printed for reproducibility but explicitly named GT ticks.
         * They MUST NOT be called 100 MHz PL cycles in the manuscript.
         *
         * Per-image averages are small enough to print as u32 using xil_printf.
         * ----------------------------------------------------------------- */
        xil_printf("\r\n");
        xil_printf(" AVG ARM GLOBAL-TIMER TICKS (NOT PL CYCLES)\r\n");
        xil_printf("------------------------------------------------------------\r\n");

        xil_printf(
            " End-to-end                    : %u GT ticks\r\n",
            (u32)(avg_total_ticks & 0xFFFFFFFFULL)
        );

        xil_printf(
            " Row PS writes                 : %u GT ticks\r\n",
            (u32)(avg_row_write_ticks & 0xFFFFFFFFULL)
        );

        xil_printf(
            " Row AvgPool start->done       : %u GT ticks\r\n",
            (u32)(avg_row_pool_ticks & 0xFFFFFFFFULL)
        );

        xil_printf(
            " Row PS reads                  : %u GT ticks\r\n",
            (u32)(avg_row_read_ticks & 0xFFFFFFFFULL)
        );

        xil_printf(
            " Col transpose/write           : %u GT ticks\r\n",
            (u32)(avg_col_write_ticks & 0xFFFFFFFFULL)
        );

        xil_printf(
            " Col AvgPool start->done       : %u GT ticks\r\n",
            (u32)(avg_col_pool_ticks & 0xFFFFFFFFULL)
        );

        xil_printf(
            " Col PS reads                  : %u GT ticks\r\n",
            (u32)(avg_col_read_ticks & 0xFFFFFFFFULL)
        );

        xil_printf(
            " Quantization/packing          : %u GT ticks\r\n",
            (u32)(avg_quant_ticks & 0xFFFFFFFFULL)
        );

        xil_printf(
            " Cache flush                   : %u GT ticks\r\n",
            (u32)(avg_flush_ticks & 0xFFFFFFFFULL)
        );

        xil_printf(
            " CNN setup                     : %u GT ticks\r\n",
            (u32)(avg_cnn_setup_ticks & 0xFFFFFFFFULL)
        );

        xil_printf(
            " CNN start->done/result        : %u GT ticks\r\n",
            (u32)(avg_cnn_exec_ticks & 0xFFFFFFFFULL)
        );


        /* Machine-readable timing summary.
         *
         * Format:
         * SUMMARY_TIME_US,
         * total,
         * row_write,row_avgpool,row_read,
         * col_write,col_avgpool,col_read,
         * quant,flush,cnn_setup,cnn_exec,
         * residual
         */
        xil_printf(
            "SUMMARY_TIME_US,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\r\n",
            total_us,
            row_write_us,
            row_pool_us,
            row_read_us,
            col_write_us,
            col_pool_us,
            col_read_us,
            quant_us,
            flush_us,
            cnn_setup_us,
            cnn_exec_us,
            residual_us
        );


#undef PRINT_TIME_MS
    }
    else {
        xil_printf(
            " ERROR: no successful images were available for timing.\r\n"
        );
    }


    xil_printf("============================================================\r\n");
    xil_printf(" END OF SAME-FLOW AVGPOOL BASELINE RUN\r\n");
    xil_printf("============================================================\r\n");

    return 0;
}
