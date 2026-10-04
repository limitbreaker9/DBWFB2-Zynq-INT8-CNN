#pragma once
#include <stdint.h>

// ── Network dimensions ─────────────────────────────────────────────────────
#define IN_H        64
#define IN_W        64
#define C1_OCH      16      // conv1 output channels
#define P1_H        32      // pool1 output H/W
#define P1_W        32
#define C2_ICH      16      // conv2 input channels  (= C1_OCH)
#define C2_OCH      32      // conv2 output channels
#define P2_H        16      // pool2 output H/W
#define P2_W        16
#define GAP_CH      32      // GAP channels          (= C2_OCH)
#define N_CLS       4       // number of output classes
#define KK          3       // conv kernel size

// ── Quantisation shifts ────────────────────────────────────────────────────
// Selected AvgPool seed-42 deployment, gain 1.
// These shifts are fixed by validation before independent-test evaluation.
#define C1_SHIFT    9
#define C2_SHIFT    7

// ── AXI input word count ───────────────────────────────────────────────────
// 64×64 INT8 pixels packed 4/word  →  1024 32-bit words
#define IN_WORDS    (IN_H * IN_W / 4)   // = 1024

// ── Top-level function ─────────────────────────────────────────────────────
void cnn_accel(
    const uint32_t *in_bram,    // m_axi  : read INT8 input (IN_WORDS words)
    uint8_t        *cls_out     // s_axilite : result class 0-3
);
