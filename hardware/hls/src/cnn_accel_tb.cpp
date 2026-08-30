// ============================================================================
// File : cnn_accel_tb.cpp
//
// DBWFB2 CNN-only C-simulation testbench
//
// Tests all 3,200 untouched STL-10 test images.
//
// Checks:
//   1. HLS prediction vs TRUE class label
//   2. HLS prediction vs Python true-INT8 software reference
//
// IMPORTANT:
// This testbench does NOT include DBWFB2 preprocessing.
// Input is already 64x64 signed INT8, packed 4 pixels per uint32_t.
// ============================================================================

#include <cstdio>
#include <cstdint>

#include "cnn_accel.h"
#include "test_vectors_3200_cnn_only.h"
#include "expected_int8_3200.h"


static const char *CLASS_NAMES[4] = {
    "airplane",
    "car",
    "dog",
    "ship"
};


int main()
{
    // ------------------------------------------------------------------------
    // Accuracy against real STL-10 labels
    // ------------------------------------------------------------------------
    int total_correct = 0;

    int class_correct[4] = {0, 0, 0, 0};
    int class_total[4]   = {0, 0, 0, 0};


    // ------------------------------------------------------------------------
    // Numerical agreement with Python true-INT8 reference
    // ------------------------------------------------------------------------
    int sw_match_count = 0;
    int sw_mismatch_count = 0;


    printf("\n");
    printf("============================================================\n");
    printf(" DBWFB2 INT8 CNN HLS C-SIMULATION\n");
    printf(" Images: %d\n", N_TEST_IMAGES);
    printf("============================================================\n");


    if (N_TEST_IMAGES != N_EXPECTED_INT8) {
        printf("[ERROR] Test-vector count and expected-label count differ.\n");
        printf("N_TEST_IMAGES  = %d\n", N_TEST_IMAGES);
        printf("N_EXPECTED_INT8 = %d\n", N_EXPECTED_INT8);
        return 1;
    }


    for (int i = 0; i < N_TEST_IMAGES; i++) {

        uint8_t pred_cls = 0xFF;

        uint8_t true_cls =
            test_labels[i];

        uint8_t sw_ref_cls =
            expected_int8_labels[i];


        if (true_cls < 4) {
            class_total[true_cls]++;
        }


        // --------------------------------------------------------------------
        // Run HLS accelerator
        //
        // test_images[i]:
        //     1024 uint32_t words
        //     4 signed INT8 pixels packed per word
        //     total = 4096 pixels = 64x64
        // --------------------------------------------------------------------
        cnn_accel(
            test_images[i],
            &pred_cls
        );


        // --------------------------------------------------------------------
        // Accuracy against true STL-10 class
        // --------------------------------------------------------------------
        bool correct =
            (pred_cls == true_cls);

        if (correct) {

            total_correct++;

            if (true_cls < 4) {
                class_correct[true_cls]++;
            }
        }


        // --------------------------------------------------------------------
        // Agreement with Python true-INT8 reference
        // --------------------------------------------------------------------
        bool sw_match =
            (pred_cls == sw_ref_cls);

        if (sw_match) {
            sw_match_count++;
        }
        else {
            sw_mismatch_count++;
        }


        // --------------------------------------------------------------------
        // Progress output
        // --------------------------------------------------------------------
        printf(
            "Image %4d | true=%-8s | SW_INT8=%-8s | HLS=%-8s "
            "| true:%s | SW:%s\n",

            i,

            (true_cls < 4)
                ? CLASS_NAMES[true_cls]
                : "UNKNOWN",

            (sw_ref_cls < 4)
                ? CLASS_NAMES[sw_ref_cls]
                : "UNKNOWN",

            (pred_cls < 4)
                ? CLASS_NAMES[pred_cls]
                : "UNKNOWN",

            correct
                ? "PASS"
                : "FAIL",

            sw_match
                ? "MATCH"
                : "MISMATCH"
        );


        // Machine-readable disagreement line
        if (!sw_match) {

            printf(
                "MISMATCH,%d,%u,%u,%u\n",
                i,
                (unsigned)true_cls,
                (unsigned)sw_ref_cls,
                (unsigned)pred_cls
            );
        }
    }


    // ========================================================================
    // TRUE-LABEL ACCURACY
    // ========================================================================

    printf("\n");
    printf("============================================================\n");
    printf(" TRUE-LABEL ACCURACY\n");
    printf("============================================================\n");

    for (int c = 0; c < 4; c++) {

        float acc = 0.0f;

        if (class_total[c] > 0) {

            acc =
                100.0f *
                ((float)class_correct[c] /
                 (float)class_total[c]);
        }

        printf(
            " %-10s : %4d / %4d = %.2f%%\n",
            CLASS_NAMES[c],
            class_correct[c],
            class_total[c],
            acc
        );
    }


    float overall_acc =
        100.0f *
        ((float)total_correct /
         (float)N_TEST_IMAGES);


    printf("------------------------------------------------------------\n");

    printf(
        " Overall HLS accuracy : %d / %d = %.2f%%\n",
        total_correct,
        N_TEST_IMAGES,
        overall_acc
    );


    // ========================================================================
    // PYTHON <-> HLS AGREEMENT
    // ========================================================================

    float agreement =
        100.0f *
        ((float)sw_match_count /
         (float)N_TEST_IMAGES);


    printf("\n");
    printf("============================================================\n");
    printf(" PYTHON TRUE-INT8 <-> HLS AGREEMENT\n");
    printf("============================================================\n");

    printf(
        " Matches    : %d / %d\n",
        sw_match_count,
        N_TEST_IMAGES
    );

    printf(
        " Mismatches : %d\n",
        sw_mismatch_count
    );

    printf(
        " Agreement  : %.4f%%\n",
        agreement
    );

    printf("============================================================\n");


    // ------------------------------------------------------------------------
    // C-simulation fails if HLS differs from
    // the Python true-INT8 numerical reference.
    //
    // Classification errors relative to the true label are normal.
    // Python/HLS numerical disagreements are not.
    // ------------------------------------------------------------------------

    if (sw_mismatch_count == 0) {

        printf(
            "\n[PASS] Python true-INT8 and HLS C-sim agree "
            "on all %d images.\n\n",
            N_TEST_IMAGES
        );

        return 0;
    }
    else {

        printf(
            "\n[FAIL] Python/HLS numerical mismatch detected "
            "on %d images.\n\n",
            sw_mismatch_count
        );

        return 1;
    }
}
