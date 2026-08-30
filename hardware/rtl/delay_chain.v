// ============================================================
// Module : delay_chain
// Purpose: 8-stage shift register — produces x[n] to x[n-8]
//          (9 taps needed for a length-9 LPF)
// ============================================================
module delay_chain (
    input  wire              clk,
    input  wire              rst_n,
    input  wire signed [15:0] xin,   // x[n]
    output reg  signed [15:0] x0,    // x[n]
    output reg  signed [15:0] x1,    // x[n-1]
    output reg  signed [15:0] x2,    // x[n-2]
    output reg  signed [15:0] x3,    // x[n-3]
    output reg  signed [15:0] x4,    // x[n-4]
    output reg  signed [15:0] x5,    // x[n-5]
    output reg  signed [15:0] x6,    // x[n-6]
    output reg  signed [15:0] x7,    // x[n-7]
    output reg  signed [15:0] x8     // x[n-8]
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x0 <= 0; x1 <= 0; x2 <= 0; x3 <= 0;
            x4 <= 0; x5 <= 0; x6 <= 0; x7 <= 0; x8 <= 0;
        end else begin
            x0 <= xin;
            x1 <= x0;
            x2 <= x1;
            x3 <= x2;
            x4 <= x3;
            x5 <= x4;
            x6 <= x5;
            x7 <= x6;
            x8 <= x7;
        end
    end
endmodule
