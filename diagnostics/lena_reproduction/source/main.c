#include <stdio.h>
#include "xparameters.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "lena_data.h" // Contains the 256x256 test image array 'lena_image'

// --- Hardware Memory and GPIO Mapping Definitions ---
#define BRAM_IN_BASE     XPAR_AXI_BRAM_CTRL_0_BASEADDR
#define BRAM_OUT_BASE    XPAR_AXI_BRAM_CTRL_1_BASEADDR
#define GPIO_BASE        XPAR_AXI_GPIO_0_BASEADDR
#define GPIO_CH1_DATA    (GPIO_BASE + 0x00)
#define GPIO_CH2_DATA    (GPIO_BASE + 0x08)

// --- Global Subband Matrix Storage ---
int16_t Subband_LL[128][128];
int16_t Subband_LH[128][128];
int16_t Subband_HL[128][128];
int16_t Subband_HH[128][128];

// Intermediate buffers for 2D forward transform steps
int16_t Matrix_L[256][256];
int16_t Matrix_H[256][256];

// =========================================================================
// HARDWARE ACCELERATOR CONTROL INTERFACE
// =========================================================================
void run_256pixel_accelerator() {
    uint32_t num_pixels = 256;
    uint32_t pixel_count_shifted = (num_pixels & 0xFFFF) << 1;

    Xil_Out32(GPIO_CH1_DATA, pixel_count_shifted | 0x01); // Assert START
    while ((Xil_In32(GPIO_CH2_DATA) & 0x01) == 0);        // Wait for DONE
    Xil_Out32(GPIO_CH1_DATA, pixel_count_shifted);        // De-assert START
}

// =========================================================================
// MAIN ROUTINE ENTRY
// =========================================================================
int main() {
    Xil_ICacheEnable();
    Xil_DCacheEnable();

    printf("\r\n=========================================================\r\n");
    printf("   DBWFB2 9/7 WAVELET HARDWARE — SUBBAND ENERGY TEST    \r\n");
    printf("=========================================================\r\n");

    // --- Hardware Step 1: Row Transform ---
    // Feed each row into the accelerator, store LP and HP outputs
    for (int r = 0; r < 256; r++) {
        for (int c = 0; c < 256; c++) {
            Xil_Out32(BRAM_IN_BASE + (c * 4), (uint32_t)lena_image[r][c]);
        }
        run_256pixel_accelerator();
        for (int c = 0; c < 256; c++) {
            uint32_t result = Xil_In32(BRAM_OUT_BASE + (c * 4));
            Matrix_L[r][c] = (int16_t)(result & 0xFFFF);
            Matrix_H[r][c] = (int16_t)((result >> 16) & 0xFFFF);
        }
    }

    // --- Hardware Step 2: Column Transform with Downsampling ---
    // Only even columns are processed (horizontal ↓2)
    // Only even rows are read back       (vertical   ↓2)
    int subband_c = 0;
    for (int c = 0; c < 256; c++) {
        if (c % 2 == 0) {

            // LL and LH — column transform on Matrix_L
            for (int r = 0; r < 256; r++) {
                Xil_Out32(BRAM_IN_BASE + (r * 4), (uint32_t)((uint16_t)Matrix_L[r][c]));
            }
            run_256pixel_accelerator();
            int subband_r = 0;
            for (int r = 0; r < 256; r++) {
                if (r % 2 == 0) {
                    uint32_t result = Xil_In32(BRAM_OUT_BASE + (r * 4));
                    Subband_LL[subband_r][subband_c] = (int16_t)(result & 0xFFFF);
                    Subband_LH[subband_r][subband_c] = (int16_t)((result >> 16) & 0xFFFF);
                    subband_r++;
                }
            }

            // HL and HH — column transform on Matrix_H
            for (int r = 0; r < 256; r++) {
                Xil_Out32(BRAM_IN_BASE + (r * 4), (uint32_t)((uint16_t)Matrix_H[r][c]));
            }
            run_256pixel_accelerator();
            subband_r = 0;
            for (int r = 0; r < 256; r++) {
                if (r % 2 == 0) {
                    uint32_t result = Xil_In32(BRAM_OUT_BASE + (r * 4));
                    Subband_HL[subband_r][subband_c] = (int16_t)(result & 0xFFFF);
                    Subband_HH[subband_r][subband_c] = (int16_t)((result >> 16) & 0xFFFF);
                    subband_r++;
                }
            }

            subband_c++;
        }
    }

    // --- Subband Energy Calculation ---
    // Each sample squared and summed; percentage shows energy compaction
    double energy_LL = 0, energy_LH = 0, energy_HL = 0, energy_HH = 0;
    for (int r = 0; r < 128; r++) {
        for (int c = 0; c < 128; c++) {
            energy_LL += (double)Subband_LL[r][c] * Subband_LL[r][c];
            energy_LH += (double)Subband_LH[r][c] * Subband_LH[r][c];
            energy_HL += (double)Subband_HL[r][c] * Subband_HL[r][c];
            energy_HH += (double)Subband_HH[r][c] * Subband_HH[r][c];
        }
    }
    double total_energy = energy_LL + energy_LH + energy_HL + energy_HH;

    printf("\r\n=== SUBBAND ENERGY COMPACTION (compare with paper Table VIII) ===\r\n");
    printf("------------------------------------------------------------------\r\n");
    printf("Subband | Energy (%%)   | Paper DBWFB2 9/7 (%%) \r\n");
    printf("------------------------------------------------------------------\r\n");
    printf("  LL    |  %9.5f   |  99.24830             \r\n", (energy_LL / total_energy) * 100.0);
    printf("  LH    |  %9.5f   |   0.14840             \r\n", (energy_LH / total_energy) * 100.0);
    printf("  HL    |  %9.5f   |   0.49380             \r\n", (energy_HL / total_energy) * 100.0);
    printf("  HH    |  %9.5f   |   0.10950             \r\n", (energy_HH / total_energy) * 100.0);
    printf("------------------------------------------------------------------\r\n");
    printf("  Total |  100.00000   |  100.00000            \r\n");
    printf("------------------------------------------------------------------\r\n");

    Xil_DCacheDisable();
    Xil_ICacheDisable();
    return 0;
}