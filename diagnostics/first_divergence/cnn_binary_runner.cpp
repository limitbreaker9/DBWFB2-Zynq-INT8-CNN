// Diagnostic host runner for the frozen DBWFB2 INT8 CNN.
//
// This file contains no CNN arithmetic. It reads 64x64 signed-INT8 tensors,
// packs them exactly like the PS firmware, and calls the released cnn_accel()
// implementation compiled from hardware/hls/src/cnn_accel.cpp.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include "cnn_accel.h"

static constexpr std::size_t PIXELS_PER_IMAGE = 64U * 64U;
static constexpr std::size_t WORDS_PER_IMAGE = PIXELS_PER_IMAGE / 4U;

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: cnn_binary_runner INPUT_INT8.bin OUTPUT_PREDICTIONS.bin\n";
        return 2;
    }

    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "cannot open input: " << argv[1] << "\n";
        return 3;
    }
    const std::streamsize byte_count = input.tellg();
    if (byte_count <= 0 || byte_count % static_cast<std::streamsize>(PIXELS_PER_IMAGE) != 0) {
        std::cerr << "input size is not a positive multiple of 4096 bytes\n";
        return 4;
    }
    input.seekg(0, std::ios::beg);

    const std::size_t image_count =
        static_cast<std::size_t>(byte_count) / PIXELS_PER_IMAGE;
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    if (!output) {
        std::cerr << "cannot open output: " << argv[2] << "\n";
        return 5;
    }

    std::vector<std::uint8_t> bytes(PIXELS_PER_IMAGE);
    std::vector<std::uint32_t> words(WORDS_PER_IMAGE);
    for (std::size_t image = 0; image < image_count; ++image) {
        input.read(reinterpret_cast<char *>(bytes.data()), PIXELS_PER_IMAGE);
        if (!input) {
            std::cerr << "short read at image " << image << "\n";
            return 6;
        }
        for (std::size_t word = 0; word < WORDS_PER_IMAGE; ++word) {
            const std::size_t base = 4U * word;
            words[word] =
                static_cast<std::uint32_t>(bytes[base + 0U]) |
                (static_cast<std::uint32_t>(bytes[base + 1U]) << 8U) |
                (static_cast<std::uint32_t>(bytes[base + 2U]) << 16U) |
                (static_cast<std::uint32_t>(bytes[base + 3U]) << 24U);
        }
        std::uint8_t prediction = 0xFFU;
        cnn_accel(words.data(), &prediction);
        output.write(reinterpret_cast<const char *>(&prediction), 1);
        if (!output) {
            std::cerr << "write failure at image " << image << "\n";
            return 7;
        }
    }

    std::cout << "processed_images=" << image_count << "\n";
    return 0;
}
