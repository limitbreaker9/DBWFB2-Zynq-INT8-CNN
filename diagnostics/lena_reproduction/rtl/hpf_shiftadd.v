// ============================================================
// Module : hpf_shiftadd
// Purpose: Multiplier-free shift-add for all HPF g(k) terms
//          DBWFB2 9/7 — coefficients scaled by 64 (>>6 final)
//
// HPF taps (7-tap, length 7, centre = x[n-3]):
//   g(0)  =  36/64  → centre tap x3
//   g(±1) = -19/64  → x2, x4
//   g(±2) =  -2/64  → x1, x5
//   g(±3) =   3/64  → x0, x6
//
// Filter equation:
//   Y_hpf[n] =  36*x3
//             - 19*(x2 + x4)
//             -  2*(x1 + x5)
//             +  3*(x0 + x6)
//
// Shift-add derivations:
//    36 = 32+4  = 2^5 + 2^2
//    19 = 16+2+1 = 2^4 + 2^1 + 2^0
//     2 = 2^1
//     3 = 2+1   = 2^1 + 2^0
//
// NOTE: HPF uses only taps x0–x6 (7 taps).
//       x7, x8 are used by LPF only.
// ============================================================
module hpf_shiftadd (
    input  wire clk,
    input  wire rst_n,
    input  wire signed [15:0] x0, x1, x2, x3,
                               x4, x5, x6,
    output wire signed [31:0] y_hpf_raw   // before >>6
);

    // ---- symmetric sums ----
    wire signed [16:0] s03 = x0 + x6;   // g(±3) pair
    wire signed [16:0] s12 = x1 + x5;   // g(±2) pair — negative
    wire signed [16:0] s21 = x2 + x4;   // g(±1) pair — negative

    // ---- 36 * x3 ----
    // 36 = 32+4
    wire signed [31:0] mult36;
    assign mult36  = ({{16{x3[15]}}, x3} <<< 5)
                   + ({{16{x3[15]}}, x3} <<< 2);

    // ---- 19 * (x2+x4) ----
    // 19 = 16+2+1
    wire signed [31:0] mult19;
    assign mult19  = ({{15{s21[16]}}, s21} <<< 4)
                   + ({{15{s21[16]}}, s21} <<< 1)
                   +  {{15{s21[16]}}, s21};

    // ---- 2 * (x1+x5) ----
    // 2 = 2^1
    wire signed [31:0] mult2;
    assign mult2   =  {{15{s12[16]}}, s12} <<< 1;

    // ---- 3 * (x0+x6) ----
    // 3 = 2+1
    wire signed [31:0] mult3;
    assign mult3   = ({{15{s03[16]}}, s03} <<< 1)
                   +  {{15{s03[16]}}, s03};
    // registered multipliers
    reg signed [31:0] mult36_r, mult19_r, mult2_r, mult3_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mult36_r <= 0; mult19_r <= 0; mult2_r <= 0; mult3_r <= 0;
        end else begin
            mult36_r <= mult36;
            mult19_r <= mult19;
            mult2_r  <= mult2;
            mult3_r  <= mult3;
        end
    end
    // ---- FIR sum ----
    // g(±1) and g(±2) are negative — subtract
    assign y_hpf_raw =  mult36_r
                      - mult19_r
                      - mult2_r
                      + mult3_r;

endmodule
