# DBWFB2 full-test deployment headers

The independent 3,200-image FPGA test is partitioned into four build-time packages of 800 raw 128x128 images:

| Part | Global indices | Header |
|---:|---:|---|
| 0 | 0-799 | `test_vectors_raw_3200_part0_dbwfb2.h` |
| 1 | 800-1599 | `test_vectors_raw_3200_part1_dbwfb2.h` |
| 2 | 1600-2399 | `test_vectors_raw_3200_part2_dbwfb2.h` |
| 3 | 2400-3199 | `test_vectors_raw_3200_part3_dbwfb2.h` |

Each header declares 800 images and the matching `TEST_PART_START`. The four disjoint ranges cover global indices 0 through 3199 exactly once. The accompanying manifest records each public header's size and SHA-256 hash.

Use `firmware/dbwfb2/main_full3200.c` and select the required `TEST_PART` at build time. Final UART captures and the derived aggregate metrics are under `results/logs/dbwfb2/uart/` and `results/`, respectively.
