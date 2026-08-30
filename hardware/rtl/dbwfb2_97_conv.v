// =========================================================================================
// Module : dbwfb2_97_conv
// Purpose: Top-level discrete wavelet transform (DWT) convolution unit for DBWFB2 9/7 FB.
//          This module implements the low-pass (LL-subband / approximation) channel of the
//          1D filter bank described in:
//          "A Novel Design Approach and VLSI Architecture of Rationalized Bi-Orthogonal
//           Wavelet Filter Banks" - IEEE TVLSI, Vol. 32, No. 4, April 2024.
//
// Only the LPF / LL path is active because the CNN consumes the LL subband.
//
// Active Pipeline Stages:
//   1. delay_chain  - 8-stage shift register; exposes x0..x8 (9 taps).
//   2. lpf_shiftadd - Multiplierless 9-tap LPF shift-add tree → y_lpf_raw (32-bit).
//   3. Pipeline reg - Registers y_lpf_raw to break the long combinational path (Fmax).
//   4. output_norm  - Arithmetic right-shift >>>10; truncates to 16-bit signed y_lpf.
//
// Coefficients (LPF, scaled by 1024):
//   h(k) : 614, 273, -78, -17, 27   (symmetric taps, centre = x4 = x[n-4])
// =========================================================================================
module dbwfb2_97_conv (
    input  wire              clk,
    input  wire              rst_n,
    input  wire signed [15:0] xin,
    output wire signed [15:0] y_lpf   // LL-subband approximation coefficient
);

    // ---- 1. Delay chain ----
    // Exposes 9 consecutive taps x0..x8 for the LPF sliding window.
    wire signed [15:0] x0, x1, x2, x3, x4, x5, x6, x7, x8;

    delay_chain u_delay (
        .clk   (clk),
        .rst_n (rst_n),
        .xin   (xin),
        .x0    (x0), .x1 (x1), .x2 (x2), .x3 (x3),
        .x4    (x4),
        .x5    (x5), .x6 (x6), .x7 (x7), .x8 (x8)
    );

    // ---- 2. LPF shift-add tree ----
    // Multiplierless 9-tap symmetric convolution → 32-bit un-normalised result.
    wire signed [31:0] y_lpf_raw;

    lpf_shiftadd u_lpf (
        .clk       (clk),
        .rst_n     (rst_n),
        .x0        (x0), .x1 (x1), .x2 (x2), .x3 (x3),
        .x4        (x4),
        .x5        (x5), .x6 (x6), .x7 (x7), .x8 (x8),
        .y_lpf_raw (y_lpf_raw)
    );

    // ---- 3. Pipeline register ----
    // Breaks the combinational path of the LPF adder tree to maximise Fmax.
    reg signed [31:0] y_lpf_raw_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            y_lpf_raw_reg <= 32'sd0;
        else
            y_lpf_raw_reg <= y_lpf_raw;
    end

    // ---- 4. Output normalisation ----
    // Arithmetic right-shift >>>10 then truncate to 16-bit signed y_lpf.
    output_norm u_norm (
        .clk       (clk),
        .rst_n     (rst_n),
        .y_lpf_raw (y_lpf_raw_reg),
        .y_lpf     (y_lpf)
    );

endmodule
