"""Shared static TensorRT I/O buffers; independent of model/export/training code."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

INTEGRATED_CONTRACT = "bgr848_integrated_v1"


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def gpu_info():
    prop = torch.cuda.get_device_properties(0)
    return dict(gpu=prop.name, compute_capability=[prop.major, prop.minor])


def latency_summary(values):
    values = np.asarray(values, dtype=float)
    return dict(
        mean_ms=float(values.mean()),
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        max_ms=float(values.max()),
        over_16_67ms_fraction=float((values >= 1000 / 60).mean()),
        sequential_fps=1000 / float(values.mean()),
    )


class EngineBuffers:
    """Subclasses declare INPUT_SHAPE, OUTPUTS and CONTRACT; sequential GPU0 use."""

    def __init__(self, engine_path):
        import tensorrt as trt

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; TensorRT requires a supported NVIDIA GPU"
            )
        torch.cuda.set_device(0)
        engine_path = Path(engine_path)
        meta = json.loads(engine_path.with_suffix(".json").read_text())
        if meta.get("tensorrt") != trt.__version__ or any(
            meta.get(k) != v for k, v in gpu_info().items()
        ):
            raise ValueError(
                "Engine GPU/TensorRT mismatch: rebuild on this deployment GPU"
            )
        if (
            meta.get("engine_sha256") != sha256(engine_path)
            or meta.get("shape") != list(self.INPUT_SHAPE)
            or meta.get("contract", "normalized_v1") != self.CONTRACT
        ):
            raise ValueError("Engine metadata/checksum mismatch")
        self.metadata = meta
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("Cannot load engine; rebuild on this deployment GPU")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create TensorRT execution context")
        self.stream = torch.cuda.Stream()
        self.buffers = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            dtype = torch.from_numpy(
                np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            ).dtype
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d <= 0 for d in shape):
                raise ValueError("Only static tensors are supported")
            self.buffers[name] = torch.empty(shape, dtype=dtype, device="cuda")
            if not self.context.set_tensor_address(name, self.buffers[name].data_ptr()):
                raise RuntimeError(f"Cannot bind tensor {name}")
        if (
            set(self.buffers) != {"images", *self.OUTPUTS}
            or tuple(self.buffers["images"].shape) != self.INPUT_SHAPE
        ):
            raise ValueError("Unexpected engine I/O names or input shape")
        self.host_input = torch.empty(
            self.INPUT_SHAPE, dtype=self.buffers["images"].dtype, pin_memory=True
        )
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
