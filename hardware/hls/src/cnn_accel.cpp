// ============================================================================
// File   : cnn_accel.cpp
// Target : Vitis HLS 2020+ / Vivado HLS 2019.x   (Zynq-7000)
// Clock  : 100 MHz
//
// Pipeline : Conv1 → Pool1 → Conv2 → Pool2 → GAP → Dense → Argmax
// ============================================================================

#include "cnn_accel.h"
#include "weights.h"

void cnn_accel(
    const uint32_t *in_bram,    // m_axi : IN_WORDS 32-bit words of packed INT8
    uint8_t        *cls_out     // s_axilite : predicted class 0-3
) {
    // ── AXI interfaces ──────────────────────────────────────────────────────
    #pragma HLS INTERFACE m_axi     port=in_bram  bundle=MBUS  \
        offset=slave  depth=1024                                \
        max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE s_axilite port=cls_out  bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return   bundle=CTRL

    // ── Weight & Bias Storage Bindings ──────────────────────────────────────
    #pragma HLS bind_storage variable=c2_w type=ROM_1P impl=BRAM
    #pragma HLS bind_storage variable=c1_w type=ROM_1P impl=LUTRAM
    #pragma HLS bind_storage variable=c1_b type=ROM_1P impl=LUTRAM
    #pragma HLS bind_storage variable=c2_b type=ROM_1P impl=LUTRAM
    #pragma HLS bind_storage variable=d_w  type=ROM_1P impl=LUTRAM
    #pragma HLS ARRAY_PARTITION variable=d_b complete dim=1

    // ── Local Feature Maps ──────────────────────────────────────────────────
    int8_t  fm0[IN_H][IN_W];                     // SIGNED INT8 for Wavelet DWT Input
    uint8_t fm1[C1_OCH][IN_H][IN_W];             // ReLU bounded [0, 127]
    uint8_t fm2[C2_ICH][P1_H][P1_W];             
    uint8_t fm3[C2_OCH][P1_H][P1_W];             // ReLU bounded [0, 127]
    uint8_t fm4[C2_OCH][P2_H][P2_W];             

    #pragma HLS ARRAY_PARTITION variable=fm1 cyclic factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=fm1 cyclic factor=2 dim=3
    #pragma HLS ARRAY_PARTITION variable=fm3 cyclic factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=fm3 cyclic factor=2 dim=3

    #pragma HLS bind_storage variable=fm0 type=RAM_2P impl=BRAM
    #pragma HLS bind_storage variable=fm2 type=RAM_2P impl=BRAM
    #pragma HLS bind_storage variable=fm4 type=RAM_2P impl=BRAM

    // ────────────────────────────────────────────────────────────────────────
    // STEP 1 — READ INPUT BRAM 
    // ────────────────────────────────────────────────────────────────────────
    RD_ROW: for (int r = 0; r < IN_H; r++) {
        RD_COL: for (int c = 0; c < IN_W; c += 4) {
            #pragma HLS PIPELINE II=1
            uint32_t w = in_bram[r * (IN_W / 4) + (c >> 2)];
// 🟢 Use Option A (Standard Little-Endian):
fm0[r][c+0] = (int8_t)( w         & 0xFFu);
fm0[r][c+1] = (int8_t)((w >>  8)  & 0xFFu);
fm0[r][c+2] = (int8_t)((w >> 16)  & 0xFFu);
fm0[r][c+3] = (int8_t)((w >> 24)  & 0xFFu);

// ❌ Comment out Option B:
/*
fm0[r][c+0] = (int8_t)((w >> 24)  & 0xFFu);
fm0[r][c+1] = (int8_t)((w >> 16)  & 0xFFu);
fm0[r][c+2] = (int8_t)((w >>  8)  & 0xFFu);
fm0[r][c+3] = (int8_t)( w         & 0xFFu);
*/
            
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // STEP 2 — CONV1
    // ────────────────────────────────────────────────────────────────────────
    C1_OC: for (int oc = 0; oc < C1_OCH; oc++) {
        C1_OH: for (int oh = 0; oh < IN_H; oh++) {
            C1_OW: for (int ow = 0; ow < IN_W; ow++) {

                int32_t acc = (int32_t)c1_b[oc];

                C1_KY: for (int ky = 0; ky < KK; ky++) {
                    C1_KX: for (int kx = 0; kx < KK; kx++) {
                        #pragma HLS PIPELINE II=1
                        int ih = oh + ky - 1;
                        int iw = ow + kx - 1;
                        
                        // Preserve signed nature of DWT input
                        int8_t iv = (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W)
                                    ? fm0[ih][iw] : (int8_t)0;
                                     
                        acc += (int32_t)iv * (int32_t)c1_w[ky * 48 + kx * 16 + oc];
                    }
                }

                int32_t s = acc >> C1_SHIFT;
                fm1[oc][oh][ow] = (s <= 0)   ? (uint8_t)0   :
                                  (s > 127)  ? (uint8_t)127 :
                                               (uint8_t)s;
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // STEP 3 — MAXPOOL1
    // ────────────────────────────────────────────────────────────────────────
    P1_IC: for (int ic = 0; ic < C1_OCH; ic++) {
        P1_OH: for (int oh = 0; oh < P1_H; oh++) {
            P1_OW: for (int ow = 0; ow < P1_W; ow++) {
                #pragma HLS PIPELINE II=1
                uint8_t v00 = fm1[ic][oh*2+0][ow*2+0];
                uint8_t v01 = fm1[ic][oh*2+0][ow*2+1];
                uint8_t v10 = fm1[ic][oh*2+1][ow*2+0];
                uint8_t v11 = fm1[ic][oh*2+1][ow*2+1];
                uint8_t m0  = (v00 > v01) ? v00 : v01;
                uint8_t m1  = (v10 > v11) ? v10 : v11;
                fm2[ic][oh][ow] = (m0 > m1) ? m0 : m1;
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // STEP 4 — CONV2
    // ────────────────────────────────────────────────────────────────────────
    C2_OC: for (int oc = 0; oc < C2_OCH; oc++) {
        C2_OH: for (int oh = 0; oh < P1_H; oh++) {
            C2_OW: for (int ow = 0; ow < P1_W; ow++) {

                int32_t acc = (int32_t)c2_b[oc];

                C2_KY: for (int ky = 0; ky < KK; ky++) {
                    C2_KX: for (int kx = 0; kx < KK; kx++) {
                        C2_IC: for (int ic = 0; ic < C2_ICH; ic++) {
                            #pragma HLS PIPELINE II=1
                            int ih = oh + ky - 1;
                            int iw = ow + kx - 1;
                            uint8_t iv = (ih >= 0 && ih < P1_H && iw >= 0 && iw < P1_W)
                                         ? fm2[ic][ih][iw] : (uint8_t)0;
                            acc += (int32_t)iv * (int32_t)c2_w[ky*1536 + kx*512 + ic*32 + oc];
                        }
                    }
                }

                int32_t s = acc >> C2_SHIFT;
                fm3[oc][oh][ow] = (s <= 0)   ? (uint8_t)0   :
                                  (s > 127)  ? (uint8_t)127 :
                                               (uint8_t)s;
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // STEP 5 — MAXPOOL2
    // ────────────────────────────────────────────────────────────────────────
    P2_IC: for (int ic = 0; ic < C2_OCH; ic++) {
        P2_OH: for (int oh = 0; oh < P2_H; oh++) {
            P2_OW: for (int ow = 0; ow < P2_W; ow++) {
                #pragma HLS PIPELINE II=1
                uint8_t v00 = fm3[ic][oh*2+0][ow*2+0];
                uint8_t v01 = fm3[ic][oh*2+0][ow*2+1];
                uint8_t v10 = fm3[ic][oh*2+1][ow*2+0];
                uint8_t v11 = fm3[ic][oh*2+1][ow*2+1];
                uint8_t m0  = (v00 > v01) ? v00 : v01;
                uint8_t m1  = (v10 > v11) ? v10 : v11;
                fm4[ic][oh][ow] = (m0 > m1) ? m0 : m1;
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // STEP 6 — GLOBAL AVERAGE POOL
    // ────────────────────────────────────────────────────────────────────────
    uint32_t gap[GAP_CH];
    #pragma HLS ARRAY_PARTITION variable=gap complete dim=1

    GAP_IC: for (int ic = 0; ic < GAP_CH; ic++) {
        uint32_t s = 0;
        GAP_H: for (int h = 0; h < P2_H; h++) {
            GAP_W: for (int w = 0; w < P2_W; w++) {
                #pragma HLS PIPELINE II=1
                s += (uint32_t)fm4[ic][h][w];
            }
        }
        gap[ic] = s >> 8;
    }

    // ────────────────────────────────────────────────────────────────────────
    // STEP 7 — DENSE 32→4   +   ARGMAX
    // ────────────────────────────────────────────────────────────────────────
    int32_t logit[N_CLS];
    #pragma HLS ARRAY_PARTITION variable=logit complete dim=1

    DN_OC: for (int oc = 0; oc < N_CLS; oc++) {
        int32_t acc = d_b[oc];
        DN_IC: for (int ic = 0; ic < GAP_CH; ic++) {
            #pragma HLS PIPELINE II=1
            acc += (int32_t)gap[ic] * (int32_t)d_w[ic * N_CLS + oc];
        }
        logit[oc] = acc;
    }

    // Combinational Argmax Unrolling
    uint8_t best = 0;
    if (logit[1] > logit[0])                 best = 1;
    if (logit[2] > logit[(int)best])         best = 2;
    if (logit[3] > logit[(int)best])         best = 3;

    *cls_out = best;
}