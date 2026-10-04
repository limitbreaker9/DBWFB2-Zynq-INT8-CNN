// =========================================================================================
// Module : dbwfb2_97_conv
// Purpose: Top-level discrete wavelet transform (DWT) convolution unit for DBWFB2 9/7 FB.
//          This module implements the discrete-time 1D convolution equations depicted in 
//          Fig. 9 of the primary reference paper:
//          "A Novel Design Approach and VLSI Architecture of Rationalized Bi-Orthogonal 
//           Wavelet Filter Banks" - IEEE TVLSI, Vol. 32, No. 4, April 2024.
//
// Architecture and Pipeline Stages:
//   1. delay_chain  - An 8-stage deep shift register chain composed of cascading flip-flops.
//                     It buffers the streaming input sample 'xin' to concurrently expose a 
//                     moving window of 9 consecutive temporal samples: x[n] down to x[n-8].
//   2. lpf_shiftadd - A fully rationalized, multiplierless shift-and-add hardware tree 
//                     implementing the 9-tap Low-Pass Filter (LPF) transfer function h(k).
//   3. hpf_shiftadd - A multiplierless shift-and-add hardware tree implementing the 
//                     7-tap High-Pass Filter (HPF) transfer function g(k).
//   4. output_norm  - Dynamic scale normalization stages implementing arithmetic right shifts 
//                     (>>10 for LPF to division by 1024, and >>6 for HPF to division by 64)
//                     to restore the correct fixed-point precision and prevent bit growth.
//
// Coefficients (Extracted from Table VI, DBWFB2 Sub-class):
//   The filter bank eliminates costly floating-point multipliers by scaling coefficients 
//   into dyadic rationals, optimized for binary shifts and additions:
//   - LPF h(k) normalized by 1024 : 614, 273, -78, -17, 27  (Symmetric taps around center 0..±4)
//   - HPF g(k) normalized by 64   :  36, -19,  -2,   3      (Symmetric taps around center 0..±3)
//
// Hardware I/O Interfacing:
//   - clk     - Master system clock driving all internal pipeline and shift registers.
//   - rst_n   - Active-low asynchronous reset for initializing the filter state.
//   - xin     - 16-bit signed (2's complement) input sample direct from the memory interface.
//   - y_lpf   - 16-bit signed LPF output representing coarse/approximation coefficients (LL).
//               Note: Downsampling-by-2 (subsampling) is handled downstream or at the BRAM layer.
//   - y_hpf   - 16-bit signed HPF output representing detail/high-frequency coefficients (DD).
// =========================================================================================
module dbwfb2_97_conv (
    input  wire              clk,
    input  wire              rst_n,
    input  wire signed [15:0] xin,
    output wire signed [15:0] y_lpf,
    output wire signed [15:0] y_hpf
);

    // ---- 1. Delay chain instantiation ----
    // Exposes 9 consecutive spatial/temporal sample taps (x0 to x8) from the streaming input.
    // This forms the sliding window required for parallel convolution computation.
    wire signed [15:0] x0, x1, x2, x3, x4, x5, x6, x7, x8;

    delay_chain u_delay (
        .clk   (clk),
        .rst_n (rst_n),
        .xin   (xin),
        .x0    (x0), .x1 (x1), .x2 (x2), .x3 (x3),
        .x4    (x4), // Mathematical center tap of the 9-tap LPF
        .x5    (x5), .x6 (x6), .x7 (x7), .x8 (x8)
    );

    // ---- 2. LPF shift-add tree instantiation ----
    // Computes the low-pass approximation filter output combinationally using wire routing
    // and adder trees to resolve the 9-tap symmetric window coefficients.
    wire signed [31:0] y_lpf_raw;

    lpf_shiftadd u_lpf (
        .clk       (clk),
        .rst_n     (rst_n),
        .x0        (x0), .x1 (x1), .x2 (x2), .x3 (x3),
        .x4        (x4), // Perfectly aligned center weight
        .x5        (x5), .x6 (x6), .x7 (x7), .x8 (x8),
        .y_lpf_raw (y_lpf_raw)
    );

    // ---- 3. HPF shift-add tree instantiation ----
    // The High-Pass Filter contains a shorter 7-tap structural span. 
    // Mathematically, its internal center tap aligns with x[n-3] (represented by x3 here).
    // To maintain cross-channel structural alignment with the LPF window, it maps 
    // taps x0 through x6 directly from the shared shift-register backbone.
    wire signed [31:0] y_hpf_raw;

    hpf_shiftadd u_hpf (
        .clk       (clk),
        .rst_n     (rst_n),
        .x0        (x0), .x1 (x1), .x2 (x2),
        .x3        (x3), // Center tap for the 7-tap high-pass filter calculation
        .x4        (x4), .x5 (x5), .x6 (x6),
        .y_hpf_raw (y_hpf_raw)
    );

    // ---- Pipeline register for raw outputs ----
    // Registers the wide 32-bit un-normalized intermediate products. This isolates the long 
    // combinational path of the shift-add adder trees from the subsequent normalization logic.
    // This structural pipeline register maximizes the operational clock frequency (Fmax).
    reg signed [31:0] y_lpf_raw_reg, y_hpf_raw_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_lpf_raw_reg <= 0;
            y_hpf_raw_reg <= 0;
        end else begin
            y_lpf_raw_reg <= y_lpf_raw;
            y_hpf_raw_reg <= y_hpf_raw;
        end
    end

    // ---- 4. Output normalisation stage ----
    // Converts the wide intermediate internal words down to standard 16-bit signed outputs.
    // Performs dynamic bit-shifting (division by 1024 and 64) and rounding/clamping to maintain
    // unity-gain characteristics for the filter bank channels across 2D transform steps.
    output_norm u_norm (
        .clk       (clk),
        .rst_n     (rst_n),
        .y_lpf_raw (y_lpf_raw_reg),
        .y_hpf_raw (y_hpf_raw_reg),
        .y_lpf     (y_lpf),
        .y_hpf     (y_hpf)
    );

endmodule