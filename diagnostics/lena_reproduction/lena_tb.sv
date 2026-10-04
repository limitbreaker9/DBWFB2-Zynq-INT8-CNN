`timescale 1ns/1ps
// Replay the firmware order using the original, unmodified RTL.
// Both historical BRAM XCIs specify synchronous READ_LATENCY_B = 1,
// no optional output registers, and byte addresses on their native ports.
module lena_tb;
  reg clk = 0;
  always #5 clk = ~clk;
  reg rst_n = 0;
  reg start = 0;
  wire done;
  wire [31:0] ib_addr, ob_addr, ob_din;
  wire ib_en, ob_en;
  wire [3:0] ob_we;
  reg [31:0] ib_dout = 0;
  reg [31:0] input_mem [0:255];
  reg [31:0] output_mem [0:255];
  reg [7:0] pixels [0:65535];
  reg signed [15:0] matrix_l [0:65535];
  reg signed [15:0] matrix_h [0:65535];
  reg signed [15:0] ll [0:16383];
  reg signed [15:0] lh [0:16383];
  reg signed [15:0] hl [0:16383];
  reg signed [15:0] hh [0:16383];
  integer r, c, i, j, fd, wait_cycles, transactions = 0;
  integer idle_cycles = 32;
  dbwfb2_97_bram_ctrl dut(
    .clk(clk), .rst_n(rst_n), .start(start), .pixel_count(16'd256),
    .done(done), .ib_addr(ib_addr), .ib_en(ib_en), .ib_dout(ib_dout),
    .ob_addr(ob_addr), .ob_en(ob_en), .ob_we(ob_we), .ob_din(ob_din)
  );
  always @(posedge clk) begin
    if (!rst_n) ib_dout <= 0;
    else if (ib_en) begin
      if ((ib_addr >> 2) > 255) $fatal(1, "Input address out of range");
      ib_dout <= input_mem[ib_addr >> 2];
    end
    if (rst_n && ob_en && |ob_we) begin
      if (ob_we !== 4'b1111) $fatal(1, "Unexpected partial output write");
      if ((ob_addr >> 2) > 255) $fatal(1, "Output address out of range");
      output_mem[ob_addr >> 2] <= ob_din;
    end
  end
  task run_line;
    begin
      @(negedge clk); start = 1;
      wait_cycles = 0;
      while (done !== 1'b1) begin
        @(posedge clk); #1;
        wait_cycles = wait_cycles + 1;
        if (wait_cycles > 400) $fatal(1, "Controller timeout");
      end
      transactions = transactions + 1;
      @(negedge clk); start = 0;
      repeat(idle_cycles) @(negedge clk);
    end
  endtask
  initial begin
    if ($value$plusargs("idle=%d", idle_cycles)) begin end
    $readmemh("lena_pixels.hex", pixels);
    for (i = 0; i < 256; i = i + 1) begin
      input_mem[i] = 0; output_mem[i] = 0;
    end
    repeat(4) @(negedge clk);
    rst_n = 1;
    repeat(32) @(negedge clk);
    // Firmware horizontal pass: store all 256 packed low/high outputs.
    for (r = 0; r < 256; r = r + 1) begin
      for (i = 0; i < 256; i = i + 1) begin
        @(negedge clk); input_mem[i] = {24'd0, pixels[r*256+i]};
      end
      run_line;
      for (i = 0; i < 256; i = i + 1) begin
        matrix_l[r*256+i] = $signed(output_mem[i][15:0]);
        matrix_h[r*256+i] = $signed(output_mem[i][31:16]);
      end
    end
    // Firmware vertical pass: only even columns, L then H per column;
    // read only even row addresses. Do not reset between transactions.
    for (c = 0; c < 256; c = c + 2) begin
      for (i = 0; i < 256; i = i + 1) begin
        @(negedge clk); input_mem[i] = {16'd0, matrix_l[i*256+c]};
      end
      run_line;
      for (r = 0; r < 256; r = r + 2) begin
        j = (r/2)*128+c/2;
        ll[j] = $signed(output_mem[r][15:0]);
        lh[j] = $signed(output_mem[r][31:16]);
      end
      for (i = 0; i < 256; i = i + 1) begin
        @(negedge clk); input_mem[i] = {16'd0, matrix_h[i*256+c]};
      end
      run_line;
      for (r = 0; r < 256; r = r + 2) begin
        j = (r/2)*128+c/2;
        hl[j] = $signed(output_mem[r][15:0]);
        hh[j] = $signed(output_mem[r][31:16]);
      end
    end
    fd = $fopen("subbands.txt", "w");
    if (!fd) $fatal(1, "Cannot open temporary output");
    for (i = 0; i < 16384; i = i + 1)
      $fwrite(fd, "%0d %0d %0d %0d\n", ll[i], lh[i], hl[i], hh[i]);
    $fclose(fd);
    $display("PASS: %0d original-RTL transactions; four 128x128 subbands; idle=%0d", transactions, idle_cycles);
    $finish;
  end
endmodule
