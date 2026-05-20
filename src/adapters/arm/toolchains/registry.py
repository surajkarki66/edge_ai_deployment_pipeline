"""ARM toolchain backends — NCNN, extensible for future targets."""

from adapters.arm.toolchains.ncnn import NcnnToolchain

ARM_TOOLCHAINS = {
    "ncnn": NcnnToolchain(),
}
