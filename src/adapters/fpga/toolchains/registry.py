"""FPGA toolchain backends — Vitis AI (DPUCZDX8G)."""

from adapters.fpga.toolchains.vitis_ai import VitisAiToolchain

FPGA_TOOLCHAINS = {
    "vitis_ai": VitisAiToolchain(),
}
