# Note: FCLK0 and RESET are automatically constrained by the Zynq Block Design.
# Do not add manual create_clock or set_false_path constraints for them.

# Bitstream Settings
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]