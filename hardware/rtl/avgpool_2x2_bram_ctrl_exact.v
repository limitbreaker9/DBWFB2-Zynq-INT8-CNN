// =========================================================================================
// Module : avgpool_2x2_bram_ctrl
// Version: exact_v2
// Purpose: Same-flow 2x2 AvgPool preprocessing controller for Zynq BRAM.
//
// IMPORTANT:
//   This version explicitly accommodates synchronous BRAM read latency.
//   It replaces the earlier experimental AvgPool controller.
//
// NUMERICAL OPERATION
// -------------------
// Each invocation processes one 1-D stream of pixel_count values.
//
// Adjacent pairs are summed exactly:
//
//      y[k] = x[2k] + x[2k+1]
//
// No division or rounding occurs in RTL.
//
// When used twice by the PS:
//
//   row pass    -> horizontal pair sum: 0..510
//   column pass -> four-pixel sum:      0..1020
//
// The exact 2x2 average is therefore:
//
//      AvgPool = four_pixel_sum / 4
//
// Firmware performs the final deployment quantization using
// round-to-nearest-even:
//
//      q = RNE((four_pixel_sum - 508) / 4)
//
// This preserves the software ordering AvgPool -> subtract 127 -> rint.
//
// MEMORY LAYOUT
// -------------
// Input BRAM:  32-bit words, useful data in bits [15:0].
// Output BRAM: 32-bit words, zero-extended sum in bits [15:0].
//
// Pair outputs are intentionally written at output sample indices
// 0,2,4,... so the existing PS stride-2 reads can be reused unchanged:
//
//      pair 0 -> byte address   0
//      pair 1 -> byte address   8
//      pair 2 -> byte address  16
//      ...
//
// HANDSHAKE
// ---------
// start is asserted by AXI GPIO.
// done remains high until start is deasserted.
//
// BRAM LATENCY
// ------------
// The controller holds each input address for BRAM_READ_WAIT clock edges
// before capturing ib_dout.  Default BRAM_READ_WAIT=2 is deliberately
// conservative and works with the synchronous BRAM path used in the copied
// Vivado design; an extra wait cycle does not change arithmetic correctness.
//
// No multiplier or DSP is required.
// =========================================================================================

module avgpool_2x2_bram_ctrl #(
    parameter ADDR_W         = 16,
    parameter DATA_W         = 16,
    parameter BRAM_READ_WAIT = 2
)(
    input  wire                clk,
    input  wire                rst_n,

    // System control/status
    input  wire                start,
    input  wire [ADDR_W-1:0]   pixel_count,
    output reg                 done,

    // Input BRAM Port B
    output reg  [31:0]         ib_addr,
    output reg                 ib_en,
    input  wire [31:0]         ib_dout,

    // Output BRAM Port B
    output reg  [31:0]         ob_addr,
    output reg                 ob_en,
    output reg  [3:0]          ob_we,
    output wire [31:0]         ob_din
);

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    localparam S_IDLE     = 3'd0;
    localparam S_WAIT_A   = 3'd1;
    localparam S_CAP_A    = 3'd2;
    localparam S_WAIT_B   = 3'd3;
    localparam S_CAP_B    = 3'd4;
    localparam S_WRITE    = 3'd5;
    localparam S_DONE     = 3'd6;

    reg [2:0] state;

    // Index of the output pair being processed: 0 .. pixel_count/2 - 1.
    reg [ADDR_W-1:0] pair_idx;

    // Number of clock edges spent waiting for the current BRAM read.
    reg [7:0] wait_cnt;

    // First element of current adjacent pair.
    reg [DATA_W-1:0] sample_a;

    // Registered exact pair sum.
    reg [DATA_W-1:0] sum_reg;

    wire [DATA_W-1:0] sample_in = ib_dout[DATA_W-1:0];

    // One extra carry bit internally; legal experiment maximum is 1020.
    wire [DATA_W:0] add_ext =
        {1'b0, sample_a} + {1'b0, sample_in};

    assign ob_din = {{(32-DATA_W){1'b0}}, sum_reg};


    // -------------------------------------------------------------------------
    // Sequential controller
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            pair_idx  <= {ADDR_W{1'b0}};
            wait_cnt  <= 8'd0;
            sample_a  <= {DATA_W{1'b0}};
            sum_reg   <= {DATA_W{1'b0}};

            done      <= 1'b0;

            ib_addr   <= 32'd0;
            ib_en     <= 1'b0;

            ob_addr   <= 32'd0;
            ob_en     <= 1'b0;
            ob_we     <= 4'b0000;
        end
        else begin
            case (state)

                // -------------------------------------------------------------
                // Wait for software start.
                // pixel_count must be non-zero and even for this 2x reduction.
                // -------------------------------------------------------------
                S_IDLE: begin
                    done      <= 1'b0;
                    pair_idx  <= {ADDR_W{1'b0}};
                    wait_cnt  <= 8'd0;

                    ib_addr   <= 32'd0;
                    ib_en     <= 1'b0;

                    ob_addr   <= 32'd0;
                    ob_en     <= 1'b0;
                    ob_we     <= 4'b0000;

                    if (start &&
                        (pixel_count != 0) &&
                        (pixel_count[0] == 1'b0)) begin

                        // Issue x[0].
                        ib_addr  <= 32'd0;
                        ib_en    <= 1'b1;
                        wait_cnt <= 8'd0;
                        state    <= S_WAIT_A;
                    end
                end


                // -------------------------------------------------------------
                // Hold address of x[2*pair_idx] long enough for synchronous BRAM.
                // -------------------------------------------------------------
                S_WAIT_A: begin
                    ob_en <= 1'b0;
                    ob_we <= 4'b0000;

                    if (wait_cnt >= (BRAM_READ_WAIT - 1)) begin
                        wait_cnt <= 8'd0;
                        state    <= S_CAP_A;
                    end
                    else begin
                        wait_cnt <= wait_cnt + 1'b1;
                    end
                end


                // -------------------------------------------------------------
                // Capture A, then issue B = x[2*pair_idx + 1].
                // -------------------------------------------------------------
                S_CAP_A: begin
                    sample_a <= sample_in;

                    ib_addr <=
                        ((((pair_idx << 1) + 1'b1)) << 2);

                    ib_en    <= 1'b1;
                    wait_cnt <= 8'd0;
                    state    <= S_WAIT_B;
                end


                // -------------------------------------------------------------
                // Hold B address for synchronous BRAM.
                // -------------------------------------------------------------
                S_WAIT_B: begin
                    if (wait_cnt >= (BRAM_READ_WAIT - 1)) begin
                        wait_cnt <= 8'd0;
                        state    <= S_CAP_B;
                    end
                    else begin
                        wait_cnt <= wait_cnt + 1'b1;
                    end
                end


                // -------------------------------------------------------------
                // Capture B and register exact sum.
                //
                // Prepare output write at sample index 2*pair_idx:
                //     byte address = (2*pair_idx) * 4 = pair_idx * 8
                //
                // The BRAM commits this registered write on the following edge.
                // -------------------------------------------------------------
                S_CAP_B: begin
                    sum_reg <= add_ext[DATA_W-1:0];

                    ob_addr <= (pair_idx << 3);
                    ob_en   <= 1'b1;
                    ob_we   <= 4'b1111;

                    ib_en   <= 1'b0;

                    state   <= S_WRITE;
                end


                // -------------------------------------------------------------
                // At entry to this state, the output BRAM sees the previously
                // registered ob_en/ob_we/ob_addr/sum_reg and commits the write.
                //
                // Then either issue the next pair's A address or finish.
                // -------------------------------------------------------------
                S_WRITE: begin
                    ob_en <= 1'b0;
                    ob_we <= 4'b0000;

                    if ((pair_idx + 1'b1) < (pixel_count >> 1)) begin
                        pair_idx <= pair_idx + 1'b1;

                        // Issue A for the next pair:
                        // A index = 2*(pair_idx+1)
                        ib_addr <=
                            (((pair_idx + 1'b1) << 1) << 2);

                        ib_en    <= 1'b1;
                        wait_cnt <= 8'd0;
                        state    <= S_WAIT_A;
                    end
                    else begin
                        ib_en <= 1'b0;
                        state <= S_DONE;
                    end
                end


                // -------------------------------------------------------------
                // Hold done until software drops start.
                // -------------------------------------------------------------
                S_DONE: begin
                    done  <= 1'b1;

                    ib_en <= 1'b0;
                    ob_en <= 1'b0;
                    ob_we <= 4'b0000;

                    if (!start) begin
                        state <= S_IDLE;
                    end
                end


                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
