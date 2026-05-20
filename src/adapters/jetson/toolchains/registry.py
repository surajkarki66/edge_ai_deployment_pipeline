"""Jetson toolchain backends — TensorRT, extensible for future targets."""

from adapters.jetson.toolchains.tensorrt import TensorRTToolchain

JETSON_TOOLCHAINS = {
    "tensorrt": TensorRTToolchain(),
}
