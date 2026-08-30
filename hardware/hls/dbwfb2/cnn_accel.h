#pragma once
#include <stdint.h>

// Network dimensions
#define IN_H        64
#define IN_W        64
#define C1_OCH      16      // conv1 output channels
#define P1_H        32      // pool1 output height
#define P1_W        32      // pool1 output width
#define C2_ICH      16      // conv2 input channels
#define C2_OCH      32      // conv2 output channels
#define P2_H        16      // pool2 output height
#define P2_W        16      // pool2 output width
#define GAP_CH      32      // global-average-pooling channels
#define N_CLS       4       // output classes
#define KK          3       // convolution kernel size

// DBWFB2 deployment quantization shifts
#define C1_SHIFT    9
#define C2_SHIFT    8

// 64 x 64 INT8 pixels packed four per 32-bit AXI word.
#define IN_WORDS    (IN_H * IN_W / 4)

void cnn_accel(
    const uint32_t *in_bram,
    uint8_t        *cls_out
);
