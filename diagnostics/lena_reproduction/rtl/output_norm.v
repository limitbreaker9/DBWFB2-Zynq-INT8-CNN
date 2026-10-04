// ============================================================
// Module : output_norm
// Purpose: Final arithmetic right shift to produce scaled outputs
//          LPF: >>>10  (denominator 1024)  - matches >>10 block in Fig. 9
//          HPF: >>> 6  (denominator   64)  - matches >> 6 block in Fig. 9
//
// Using >>> (arithmetic right shift) explicitly, exactly as shown in
// the architecture diagram, so the synthesis tool infers a shifter
// rather than a wired connection. Sign bit is preserved correctly.
// ============================================================
module output_norm (
    input  wire                clk,
    input  wire                rst_n,
    input  wire signed [31:0] y_lpf_raw,
    input  wire signed [31:0] y_hpf_raw,
    output reg signed [15:0] y_lpf,
    output reg signed [15:0] y_hpf
);
    // Arithmetic right shift - sign-extending, matches Fig. 9 shift blocks
    wire signed [31:0] y_lpf_shifted;
    wire signed [31:0] y_hpf_shifted;

    assign y_lpf_shifted = y_lpf_raw >>> 10;
    assign y_hpf_shifted = y_hpf_raw >>> 6;

    // Truncate to output width after shifting
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        y_lpf <= 16'sd0;
        y_hpf <= 16'sd0;
    end else begin
        y_lpf <= y_lpf_shifted[15:0];
        y_hpf <= y_hpf_shifted[15:0];
    end
end
endmodule