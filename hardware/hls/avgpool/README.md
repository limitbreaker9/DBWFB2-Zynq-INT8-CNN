# AvgPool HLS package

The public AvgPool deployment configuration and quantized weights use seed 42, gain 1, `C1_SHIFT=9`, and `C2_SHIFT=7`.

The October 3 `csynth.rpt` and `packaged_ip/` contain the selected AvgPool tensors and shifts 9/7. Packaged IP revision 2114812621 has the six-tensor fingerprint `6835b8a8984eb1d4454d8f95bf9ec87731bd1ae98a0975917ef86c4c5d7053e9`. All six generated tensors match AvgPool and differ from the selected DBWFB2 package. The generated Conv1 and Conv2 shift slices are `[18:9]` and `[23:7]`.

The reported Board500 UART capture is [`../../../results/logs/avgpool/uart/avgpool_board500_20261003.txt`](../../../results/logs/avgpool/uart/avgpool_board500_20261003.txt): 416/500 correct, 498/500 selected software-reference agreement and zero errors. The [provenance manifest](../../../diagnostics/provenance/avgpool_hardware_provenance.json) records source, generated-ROM, IP, bitstream, XSA, report and UART hashes. Earlier hardware-package evidence is retained in the historical provenance directory, not as the reported AvgPool implementation.
