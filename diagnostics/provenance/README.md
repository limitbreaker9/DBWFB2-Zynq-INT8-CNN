# Hardware implementation provenance

`avgpool_hardware_provenance.json` identifies the reported AvgPool seed-42, gain-1, shifts-9/7 source, generated tensors/IP, Vivado implementation, XSA, programmed bitstream and Board500 UART capture.

`dbwfb2_hardware_provenance.json` identifies the DBWFB2 reports that support 10,622 LUT, 12,136 FF and +1.591 ns WNS. Other numerical totals are not values from this report set.

`avgpool_historical_20260830/` preserves an earlier development implementation and XSA. Its nested documentation and reports describe that build only. It used AvgPool preprocessing with the DBWFB2 CNN package. The reported implementation uses the selected AvgPool tensors and shifts 9/7, with Board500 416/500 correct and 498/500 selected software-reference agreement.

Whole-system AvgPool values are 10,327 LUT, 11,817 FF, 67 BRAM tiles, 4 DSP, WNS +0.776 ns, TNS 0.000 ns and a 1.889 W Vivado total on-chip estimate. The [October-3 resource hierarchy](../../hardware/vivado/avgpool/documentation/utilization_hierarchy_20261003.png) records the controller and CNN utilization used in the resource comparison. A separate preprocessing-hierarchy dynamic-power attribution remains unavailable.

`implementation_manifest.json` lists repository file paths and sizes. The IP core revision is a hardware-package version identifier.
