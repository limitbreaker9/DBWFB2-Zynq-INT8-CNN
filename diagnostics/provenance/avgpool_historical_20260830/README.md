# HISTORICAL: superseded AvgPool hardware evidence

All files below this directory describe earlier development artifacts, not the reported October-3 AvgPool build. Nested documents apply only to those development artifacts. Numeric data and generated artifacts are preserved.

The August-30 hardware assembly combined AvgPool preprocessing with the selected DBWFB2 CNN weights and shifts 9/8. An older August-22 platform export and later intermediate package generations are also retained for provenance. The software experiment itself had already selected AvgPool seed 42, gain 1, and shifts 9/7; it was not changed by the hardware correction. Some repository source headers already matched the software package even though the assembled/generated hardware did not.

The reported October-3 implementation uses the selected AvgPool tensors and shifts 9/7, packaged IP revision 2114812621. Its reports and programmed bitstream supersede these hardware results. See `../avgpool_hardware_provenance.json` and the repository's `hardware/vivado/avgpool/` and `hardware/hls/avgpool/` directories.
