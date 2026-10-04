// Diagnostic-only intermediate trace for the frozen DBWFB2 INT8 CNN.
//
// This source mirrors hardware/hls/src/cnn_accel.cpp operation-for-operation,
// exposes intermediate tensors, and checks every traced prediction against the
// unmodified released cnn_accel() linked into the same executable.

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "cnn_accel.h"
#include "weights.h"

static constexpr std::size_t INPUT_ELEMENTS = 64U * 64U;

template <typename T>
static void write_values(std::ofstream &stream, const std::vector<T> &values) {
    stream.write(
        reinterpret_cast<const char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T))
    );
    if (!stream) {
        throw std::runtime_error("binary checkpoint write failed");
    }
}

struct Audit {
    std::int32_t c1_acc_min = std::numeric_limits<std::int32_t>::max();
    std::int32_t c1_acc_max = std::numeric_limits<std::int32_t>::min();
    std::int32_t c1_shift_min = std::numeric_limits<std::int32_t>::max();
    std::int32_t c1_shift_max = std::numeric_limits<std::int32_t>::min();
    std::uint64_t c1_below_zero = 0, c1_equal_zero = 0;
    std::uint64_t c1_above_127 = 0, c1_equal_127 = 0;
    std::uint64_t c1_output_zero = 0, c1_output_127 = 0;

    std::int32_t c2_acc_min = std::numeric_limits<std::int32_t>::max();
    std::int32_t c2_acc_max = std::numeric_limits<std::int32_t>::min();
    std::int32_t c2_shift_min = std::numeric_limits<std::int32_t>::max();
    std::int32_t c2_shift_max = std::numeric_limits<std::int32_t>::min();
    std::uint64_t c2_below_zero = 0, c2_equal_zero = 0;
    std::uint64_t c2_above_127 = 0, c2_equal_127 = 0;
    std::uint64_t c2_output_zero = 0, c2_output_127 = 0;

    std::uint32_t gap_sum_min = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t gap_sum_max = 0;
    std::uint32_t gap_shift_min = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t gap_shift_max = 0;
    std::int32_t logit_min = std::numeric_limits<std::int32_t>::max();
    std::int32_t logit_max = std::numeric_limits<std::int32_t>::min();
};

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: cnn_intermediate_trace INPUT_INT8.bin OUTPUT_PREFIX [--aggregate-only]\n";
        return 2;
    }
    const bool aggregate_only = argc == 4 && std::string(argv[3]) == "--aggregate-only";
    if (argc == 4 && !aggregate_only) {
        std::cerr << "unknown option: " << argv[3] << "\n";
        return 2;
    }
    const std::string prefix = argv[2];

    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    if (!input) return 3;
    const std::streamsize byte_count = input.tellg();
    if (byte_count <= 0 || byte_count % static_cast<std::streamsize>(INPUT_ELEMENTS) != 0) {
        std::cerr << "input size is not a positive multiple of 4096 bytes\n";
        return 4;
    }
    input.seekg(0, std::ios::beg);
    const std::size_t images = static_cast<std::size_t>(byte_count) / INPUT_ELEMENTS;

    std::ofstream f_input, f_c1_acc, f_c1_shift, f_c1_act, f_pool1;
    std::ofstream f_c2_acc, f_c2_shift, f_c2_act, f_pool2;
    std::ofstream f_gap_sum, f_gap_shift, f_logits;
    if (!aggregate_only) {
        f_input.open(prefix + "_CNN_INPUT.bin", std::ios::binary | std::ios::trunc);
        f_c1_acc.open(prefix + "_CONV1_ACC.bin", std::ios::binary | std::ios::trunc);
        f_c1_shift.open(prefix + "_CONV1_SHIFTED.bin", std::ios::binary | std::ios::trunc);
        f_c1_act.open(prefix + "_CONV1_ACT.bin", std::ios::binary | std::ios::trunc);
        f_pool1.open(prefix + "_POOL1.bin", std::ios::binary | std::ios::trunc);
        f_c2_acc.open(prefix + "_CONV2_ACC.bin", std::ios::binary | std::ios::trunc);
        f_c2_shift.open(prefix + "_CONV2_SHIFTED.bin", std::ios::binary | std::ios::trunc);
        f_c2_act.open(prefix + "_CONV2_ACT.bin", std::ios::binary | std::ios::trunc);
        f_pool2.open(prefix + "_POOL2.bin", std::ios::binary | std::ios::trunc);
        f_gap_sum.open(prefix + "_GAP_SUM.bin", std::ios::binary | std::ios::trunc);
        f_gap_shift.open(prefix + "_GAP_SHIFTED.bin", std::ios::binary | std::ios::trunc);
        f_logits.open(prefix + "_LOGITS.bin", std::ios::binary | std::ios::trunc);
    }
    std::ofstream f_prediction(prefix + "_PREDICTION.bin", std::ios::binary | std::ios::trunc);
    if (!f_prediction) return 5;

    std::vector<std::int8_t> fm0(INPUT_ELEMENTS);
    std::vector<std::uint8_t> fm1(16U * 64U * 64U), fm2(16U * 32U * 32U);
    std::vector<std::uint8_t> fm3(32U * 32U * 32U), fm4(32U * 16U * 16U);
    std::vector<std::int32_t> c1_acc(16U * 64U * 64U), c1_shift(c1_acc.size());
    std::vector<std::int32_t> c2_acc(32U * 32U * 32U), c2_shift(c2_acc.size());
    std::vector<std::uint32_t> gap_sum(32), gap_shift(32);
    std::vector<std::int32_t> logits(4);
    std::vector<std::uint32_t> packed(1024);
    Audit audit;
    std::size_t released_prediction_mismatches = 0;

    auto i1 = [](int c, int r, int x) { return (c * 64 + r) * 64 + x; };
    auto i2 = [](int c, int r, int x) { return (c * 32 + r) * 32 + x; };
    auto i4 = [](int c, int r, int x) { return (c * 16 + r) * 16 + x; };

    for (std::size_t image = 0; image < images; ++image) {
        input.read(reinterpret_cast<char *>(fm0.data()), INPUT_ELEMENTS);
        if (!input) return 6;
        for (std::size_t word = 0; word < packed.size(); ++word) {
            const std::size_t base = 4U * word;
            packed[word] =
                static_cast<std::uint8_t>(fm0[base]) |
                (static_cast<std::uint32_t>(static_cast<std::uint8_t>(fm0[base + 1])) << 8U) |
                (static_cast<std::uint32_t>(static_cast<std::uint8_t>(fm0[base + 2])) << 16U) |
                (static_cast<std::uint32_t>(static_cast<std::uint8_t>(fm0[base + 3])) << 24U);
        }

        for (int oc = 0; oc < 16; ++oc) {
            for (int oh = 0; oh < 64; ++oh) {
                for (int ow = 0; ow < 64; ++ow) {
                    std::int32_t acc = c1_b[oc];
                    for (int ky = 0; ky < 3; ++ky) {
                        for (int kx = 0; kx < 3; ++kx) {
                            const int ih = oh + ky - 1;
                            const int iw = ow + kx - 1;
                            const std::int8_t value =
                                (ih >= 0 && ih < 64 && iw >= 0 && iw < 64)
                                    ? fm0[ih * 64 + iw] : static_cast<std::int8_t>(0);
                            acc += static_cast<std::int32_t>(value) *
                                   static_cast<std::int32_t>(c1_w[ky * 48 + kx * 16 + oc]);
                        }
                    }
                    const std::int32_t shifted = acc >> 9;
                    const std::uint8_t act = shifted <= 0 ? 0 :
                                             shifted > 127 ? 127 : static_cast<std::uint8_t>(shifted);
                    const int index = i1(oc, oh, ow);
                    c1_acc[index] = acc;
                    c1_shift[index] = shifted;
                    fm1[index] = act;
                    audit.c1_acc_min = std::min(audit.c1_acc_min, acc);
                    audit.c1_acc_max = std::max(audit.c1_acc_max, acc);
                    audit.c1_shift_min = std::min(audit.c1_shift_min, shifted);
                    audit.c1_shift_max = std::max(audit.c1_shift_max, shifted);
                    audit.c1_below_zero += shifted < 0;
                    audit.c1_equal_zero += shifted == 0;
                    audit.c1_above_127 += shifted > 127;
                    audit.c1_equal_127 += shifted == 127;
                    audit.c1_output_zero += act == 0;
                    audit.c1_output_127 += act == 127;
                }
            }
        }
        for (int ic = 0; ic < 16; ++ic) {
            for (int oh = 0; oh < 32; ++oh) {
                for (int ow = 0; ow < 32; ++ow) {
                    const std::uint8_t v00 = fm1[i1(ic, 2 * oh, 2 * ow)];
                    const std::uint8_t v01 = fm1[i1(ic, 2 * oh, 2 * ow + 1)];
                    const std::uint8_t v10 = fm1[i1(ic, 2 * oh + 1, 2 * ow)];
                    const std::uint8_t v11 = fm1[i1(ic, 2 * oh + 1, 2 * ow + 1)];
                    fm2[i2(ic, oh, ow)] = std::max(std::max(v00, v01), std::max(v10, v11));
                }
            }
        }
        for (int oc = 0; oc < 32; ++oc) {
            for (int oh = 0; oh < 32; ++oh) {
                for (int ow = 0; ow < 32; ++ow) {
                    std::int32_t acc = c2_b[oc];
                    for (int ky = 0; ky < 3; ++ky) {
                        for (int kx = 0; kx < 3; ++kx) {
                            for (int ic = 0; ic < 16; ++ic) {
                                const int ih = oh + ky - 1;
                                const int iw = ow + kx - 1;
                                const std::uint8_t value =
                                    (ih >= 0 && ih < 32 && iw >= 0 && iw < 32)
                                        ? fm2[i2(ic, ih, iw)] : static_cast<std::uint8_t>(0);
                                acc += static_cast<std::int32_t>(value) *
                                       static_cast<std::int32_t>(
                                           c2_w[ky * 1536 + kx * 512 + ic * 32 + oc]
                                       );
                            }
                        }
                    }
                    const std::int32_t shifted = acc >> 8;
                    const std::uint8_t act = shifted <= 0 ? 0 :
                                             shifted > 127 ? 127 : static_cast<std::uint8_t>(shifted);
                    const int index = i2(oc, oh, ow);
                    c2_acc[index] = acc;
                    c2_shift[index] = shifted;
                    fm3[index] = act;
                    audit.c2_acc_min = std::min(audit.c2_acc_min, acc);
                    audit.c2_acc_max = std::max(audit.c2_acc_max, acc);
                    audit.c2_shift_min = std::min(audit.c2_shift_min, shifted);
                    audit.c2_shift_max = std::max(audit.c2_shift_max, shifted);
                    audit.c2_below_zero += shifted < 0;
                    audit.c2_equal_zero += shifted == 0;
                    audit.c2_above_127 += shifted > 127;
                    audit.c2_equal_127 += shifted == 127;
                    audit.c2_output_zero += act == 0;
                    audit.c2_output_127 += act == 127;
                }
            }
        }
        for (int ic = 0; ic < 32; ++ic) {
            for (int oh = 0; oh < 16; ++oh) {
                for (int ow = 0; ow < 16; ++ow) {
                    const std::uint8_t v00 = fm3[i2(ic, 2 * oh, 2 * ow)];
                    const std::uint8_t v01 = fm3[i2(ic, 2 * oh, 2 * ow + 1)];
                    const std::uint8_t v10 = fm3[i2(ic, 2 * oh + 1, 2 * ow)];
                    const std::uint8_t v11 = fm3[i2(ic, 2 * oh + 1, 2 * ow + 1)];
                    fm4[i4(ic, oh, ow)] = std::max(std::max(v00, v01), std::max(v10, v11));
                }
            }
        }
        for (int ic = 0; ic < 32; ++ic) {
            std::uint32_t sum = 0;
            for (int h = 0; h < 16; ++h)
                for (int w = 0; w < 16; ++w)
                    sum += static_cast<std::uint32_t>(fm4[i4(ic, h, w)]);
            gap_sum[ic] = sum;
            gap_shift[ic] = sum >> 8;
            audit.gap_sum_min = std::min(audit.gap_sum_min, sum);
            audit.gap_sum_max = std::max(audit.gap_sum_max, sum);
            audit.gap_shift_min = std::min(audit.gap_shift_min, gap_shift[ic]);
            audit.gap_shift_max = std::max(audit.gap_shift_max, gap_shift[ic]);
        }
        for (int oc = 0; oc < 4; ++oc) {
            std::int32_t acc = d_b[oc];
            for (int ic = 0; ic < 32; ++ic)
                acc += static_cast<std::int32_t>(gap_shift[ic]) *
                       static_cast<std::int32_t>(d_w[ic * 4 + oc]);
            logits[oc] = acc;
            audit.logit_min = std::min(audit.logit_min, acc);
            audit.logit_max = std::max(audit.logit_max, acc);
        }
        std::uint8_t prediction = 0;
        if (logits[1] > logits[0]) prediction = 1;
        if (logits[2] > logits[prediction]) prediction = 2;
        if (logits[3] > logits[prediction]) prediction = 3;

        std::uint8_t released_prediction = 0xFF;
        cnn_accel(packed.data(), &released_prediction);
        released_prediction_mismatches += prediction != released_prediction;
        f_prediction.write(reinterpret_cast<const char *>(&prediction), 1);
        if (!aggregate_only) {
            write_values(f_input, fm0); write_values(f_c1_acc, c1_acc);
            write_values(f_c1_shift, c1_shift); write_values(f_c1_act, fm1);
            write_values(f_pool1, fm2); write_values(f_c2_acc, c2_acc);
            write_values(f_c2_shift, c2_shift); write_values(f_c2_act, fm3);
            write_values(f_pool2, fm4); write_values(f_gap_sum, gap_sum);
            write_values(f_gap_shift, gap_shift); write_values(f_logits, logits);
        }
    }

    std::ofstream summary(prefix + "_AUDIT.csv", std::ios::trunc);
    summary << "metric,value\n";
    summary << "images," << images << "\n";
    summary << "released_prediction_mismatches," << released_prediction_mismatches << "\n";
    summary << "conv1_elements," << images * 16U * 64U * 64U << "\n";
    summary << "conv1_acc_min," << audit.c1_acc_min << "\nconv1_acc_max," << audit.c1_acc_max << "\n";
    summary << "conv1_shift_min," << audit.c1_shift_min << "\nconv1_shift_max," << audit.c1_shift_max << "\n";
    summary << "conv1_shift_below_zero," << audit.c1_below_zero << "\nconv1_shift_equal_zero," << audit.c1_equal_zero << "\n";
    summary << "conv1_shift_above_127," << audit.c1_above_127 << "\nconv1_shift_equal_127," << audit.c1_equal_127 << "\n";
    summary << "conv1_output_zero," << audit.c1_output_zero << "\nconv1_output_127," << audit.c1_output_127 << "\n";
    summary << "conv2_elements," << images * 32U * 32U * 32U << "\n";
    summary << "conv2_acc_min," << audit.c2_acc_min << "\nconv2_acc_max," << audit.c2_acc_max << "\n";
    summary << "conv2_shift_min," << audit.c2_shift_min << "\nconv2_shift_max," << audit.c2_shift_max << "\n";
    summary << "conv2_shift_below_zero," << audit.c2_below_zero << "\nconv2_shift_equal_zero," << audit.c2_equal_zero << "\n";
    summary << "conv2_shift_above_127," << audit.c2_above_127 << "\nconv2_shift_equal_127," << audit.c2_equal_127 << "\n";
    summary << "conv2_output_zero," << audit.c2_output_zero << "\nconv2_output_127," << audit.c2_output_127 << "\n";
    summary << "gap_sum_min," << audit.gap_sum_min << "\ngap_sum_max," << audit.gap_sum_max << "\n";
    summary << "gap_shift_min," << audit.gap_shift_min << "\ngap_shift_max," << audit.gap_shift_max << "\n";
    summary << "logit_min," << audit.logit_min << "\nlogit_max," << audit.logit_max << "\n";
    if (released_prediction_mismatches != 0) return 8;
    std::cout << "processed_images=" << images
              << " released_prediction_mismatches=" << released_prediction_mismatches << "\n";
    return 0;
}
