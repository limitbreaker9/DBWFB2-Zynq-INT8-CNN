// ============================================================
// Module : output_norm
// Purpose: Final arithmetic right shift to produce scaled LPF output.
//          LPF: >>> 10  (denominator 1024) - matches >>10 block in Fig. 9
//
// NOTE [HPF REMOVED]: The high-pass output path (y_hpf_raw / y_hpf) has
//   been stripped. Only the LL-subband (approximation) coefficients are
//   produced. This is intentional - the CNN accelerator (conv1_pe) consumes
//   only the LPF / LL output. HPF can be re-introduced later if HH/LH/HL
//   subbands are ever needed.
//
// Using >>> (arithmetic right shift) explicitly so synthesis infers a
// shifter rather than a wired connection. Sign bit is preserved correctly.
// ============================================================
module output_norm (
    input  wire                clk,
    input  wire                rst_n,
    input  wire signed [31:0]  y_lpf_raw,
    output reg  signed [15:0]  y_lpf
);
    // Arithmetic right shift - sign-extending, matches Fig. 9 >>10 block
    wire signed [31:0] y_lpf_shifted;
    assign y_lpf_shifted = y_lpf_raw >>> 10;

    // Truncate to 16-bit output width after shifting
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            y_lpf <= 16'sd0;
        else
            y_lpf <= y_lpf_shifted[15:0];
    end
endmodule