# Vivado implementation material

Both designs target `xc7z020clg484-1` at 100 MHz. Each arm provides:

- `block_design/PL.bd`: actual Vivado block-design source;
- `constraints/constraints.xdc`: supplied clock constraint;
- `documentation/`: architecture and report screenshots;
- `reports/`: current placed/routed utilization, timing, and power reports;
- `xsa/PL_wrapper.xsa`: exported hardware platform when available.

The Zynq PS `M_AXI_GP0` master reaches two AXI BRAM controllers and AXI GPIO through the AXI/SmartConnect control fabric. The preprocessing controller accesses the two BRAMs through Port B. The PS performs the row/column transpose and reorder between separable passes. The HLS CNN reads packed input from DDR through its `m_axi` interface and the PS high-performance port. There is no claimed direct output-BRAM-to-input-BRAM data path between preprocessing passes.

The repository supplies source and final evidence rather than complete generated projects. Vivado/Vitis caches, `.runs`, `.Xil`, and workspace intermediates are intentionally excluded. The corrected AvgPool packaged HLS IP is archived under `hardware/hls/avgpool/packaged_ip/`. To reconstruct a project, make the packaged RTL/HLS IP available, import the matching block design and XDC, regenerate output products, and implement for the stated device and clock.

The AvgPool XSA contains the reported October-3 bitstream. The programming event and UART capture are recorded in the provenance manifest. The older August-22 XSA and August-30 AvgPool reports are development artifacts. DBWFB2 report identities are recorded in `diagnostics/provenance/dbwfb2_hardware_provenance.json`.
