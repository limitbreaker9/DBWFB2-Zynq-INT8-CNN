// ============================================================
// Module : lpf_shiftadd
// Purpose: Multiplier-free shift-add for all LPF h(k) terms
//          DBWFB2 9/7 — coefficients scaled by 1024 (>>10 final)
//
// Filter equation (exploiting symmetry):
//   Y_lpf[n] =  614*x4
//             + 273*(x3 + x5)
//             -  78*(x2 + x6)
//             -  17*(x1 + x7)
//             +  27*(x0 + x8)
//
// where x4 = centre tap = x[n-4]
//
// Shift-add derivations:
//   614 = 2^9 + 2^6 + 2^5 + 2^2 + 2^1
//   273 = 2^8 + 2^4 + 2^0
//    78 = 2^6 + 2^3 + 2^2 + 2^1
//    17 = 2^4 + 2^0
//    27 = 2^4 + 2^3 + 2^1 + 2^0
// ============================================================
module lpf_shiftadd (
    input  wire clk,
    input  wire rst_n,
    input  wire signed [15:0] x0, x1, x2, x3, x4,
                               x5, x6, x7, x8,
    output wire signed [31:0] y_lpf_raw   // before >>10
);

    // ---- symmetric sums ----
    wire signed [16:0] s04 = x0 + x8;   // h(±4) pair
    wire signed [16:0] s13 = x1 + x7;   // h(±3) pair — NOTE: negative coeff
    wire signed [16:0] s22 = x2 + x6;   // h(±2) pair — NOTE: negative coeff
    wire signed [16:0] s11 = x3 + x5;   // h(±1) pair

    // ---- 614 * x4 ----
    // 614 = 512+64+32+4+2
    wire signed [31:0] mult614;
    assign mult614 = ({{16{x4[15]}}, x4} <<< 9)
                   + ({{16{x4[15]}}, x4} <<< 6)
                   + ({{16{x4[15]}}, x4} <<< 5)
                   + ({{16{x4[15]}}, x4} <<< 2)
                   + ({{16{x4[15]}}, x4} <<< 1);

    // ---- 273 * (x3+x5) ----
    // 273 = 256+16+1
    wire signed [31:0] mult273;
    assign mult273 = ({{15{s11[16]}}, s11} <<< 8)
                   + ({{15{s11[16]}}, s11} <<< 4)
                   +  {{15{s11[16]}}, s11};

    // ---- 78 * (x2+x6) ----
    // 78 = 64+8+4+2
    wire signed [31:0] mult78;
    assign mult78  = ({{15{s22[16]}}, s22} <<< 6)
                   + ({{15{s22[16]}}, s22} <<< 3)
                   + ({{15{s22[16]}}, s22} <<< 2)
                   + ({{15{s22[16]}}, s22} <<< 1);

    // ---- 17 * (x1+x7) ----
    // 17 = 16+1
    wire signed [31:0] mult17;
    assign mult17  = ({{15{s13[16]}}, s13} <<< 4)
                   +  {{15{s13[16]}}, s13};

    // ---- 27 * (x0+x8) ----
    // 27 = 16+8+2+1
    wire signed [31:0] mult27;
    assign mult27  = ({{15{s04[16]}}, s04} <<< 4)
                   + ({{15{s04[16]}}, s04} <<< 3)
                   + ({{15{s04[16]}}, s04} <<< 1)
                   +  {{15{s04[16]}}, s04};
// ---- Pipeline registers for the multiplier outputs ----
    reg signed [31:0] mult614_r, mult273_r, mult78_r, mult17_r, mult27_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mult614_r <= 0; mult273_r <= 0; mult78_r <= 0;
            mult17_r  <= 0; mult27_r  <= 0;
        end else begin
            mult614_r <= mult614;
            mult273_r <= mult273;
            mult78_r  <= mult78;
            mult17_r  <= mult17;
            mult27_r  <= mult27;
        end
    end
    // ---- FIR sum ----
    // h(±2) and h(±3) are negative — subtract
    assign y_lpf_raw =  mult614_r
                      + mult273_r
                      - mult78_r
                      - mult17_r
                      + mult27_r;

endmodule
