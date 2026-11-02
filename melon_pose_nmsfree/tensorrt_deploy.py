"""Build the integrated 848x530 ONNX model on the deployment GPU."""

import argparse
import json
from pathlib import Path

import torch

if __package__:
    from .trt_runtime import INTEGRATED_CONTRACT, gpu_info, sha256
else:
    from trt_runtime import INTEGRATED_CONTRACT, gpu_info, sha256


def build_engine(onnx_path, engine_path, workspace_gib=2):
    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    metadata_path = engine_path.with_suffix(".json")
    for destination in (engine_path, metadata_path):
        if destination.exists():
            raise FileExistsError(destination)
    if workspace_gib < 1:
        raise ValueError("workspace_gib must be positive")
    manifest = json.loads(onnx_path.with_suffix(".json").read_text())
    if manifest.get("onnx_sha256") != sha256(onnx_path):
        raise ValueError("ONNX checksum differs from manifest")
    if manifest.get("contract") != INTEGRATED_CONTRACT:
        raise ValueError("Expected integrated 848x530 ONNX contract")

    import tensorrt as trt

    if not trt.__version__.startswith("10."):
        raise RuntimeError("This builder requires TensorRT 10")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; build on the deployment NVIDIA GPU")
    torch.cuda.set_device(0)
    print(f'Building on {gpu_info()["gpu"]}, TensorRT {trt.__version__}', flush=True)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path.resolve())):
        raise RuntimeError(
            "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        )
    if tuple(network.get_input(0).shape) != (1, 530, 848, 3):
        raise ValueError("Expected fixed input (1, 530, 848, 3)")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    # Preserve the original builder's FP32 preprocessing and pixel postprocessing.
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if layer.name.startswith("/model/") or layer.type == trt.LayerType.CONSTANT:
            continue
        float_outputs = [
            i
            for i in range(layer.num_outputs)
            if layer.get_output(i).dtype == trt.float32
        ]
        if float_outputs:
            layer.precision = trt.float32
            for i in float_outputs:
                layer.set_output_type(i, trt.float32)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    metadata = dict(
        manifest,
        **gpu_info(),
        tensorrt=trt.__version__,
        precision="FP16-enabled mixed precision",
        nms=False,
        engine_sha256=sha256(engine_path),
        torch=torch.__version__,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved: {engine_path}\nSaved: {metadata_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--workspace-gib", type=int, default=2)
    args = parser.parse_args()
    build_engine(args.onnx, args.engine, args.workspace_gib)


if __name__ == "__main__":
    main()
