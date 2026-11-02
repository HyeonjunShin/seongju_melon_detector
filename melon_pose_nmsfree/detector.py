"""One-call 848x530 uint8 BGR camera inference using an integrated TensorRT engine."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__:
    from .trt_runtime import EngineBuffers, INTEGRATED_CONTRACT, gpu_info, latency_summary
else:
    from trt_runtime import EngineBuffers, INTEGRATED_CONTRACT, gpu_info, latency_summary


def check_frame(frame):
    if not isinstance(frame,np.ndarray) or frame.dtype != np.uint8:
        raise ValueError('Expected a uint8 NumPy BGR frame')
    if frame.shape != (530,848,3):
        raise ValueError('Expected frame shape (530,848,3): height530, width848, BGR')


def format_objects(detections,flags):
    objects = []
    for row,flag in zip(detections,flags):
        if not flag[0]:
            continue
        if not np.isfinite(row).all():
            raise RuntimeError('Non-finite integrated TensorRT detection')
        objects.append(dict(score=float(row[0]),bbox=row[1:5].tolist(),
                            body=row[5:7].tolist(),knot=row[7:9].tolist(),
                            visibility_probability=row[9:11].tolist(),
                            visible=[bool(flag[1]),bool(flag[2])],
                            angle_deg=float(row[11]) if flag[3] else None))
    return objects


class Detector(EngineBuffers):
    """Load once, then detector(frame_bgr). Sequential calls; do not share across threads.

    All image preprocessing and numeric postprocessing run inside TensorRT. Python
    handles input validation, memory copies, invalid-slot removal and dict formatting.
    Thresholds are fixed in the engine: confidence .3, visibility .5, max5, noNMS.
    """
    INPUT_SHAPE = (1,530,848,3)
    OUTPUTS = {'detections','flags'}
    CONTRACT = INTEGRATED_CONTRACT

    def __init__(self,engine_path):
        super().__init__(engine_path)
        if (self.metadata.get('input_dtype') != 'uint8'
            or self.metadata.get('input_format') != 'BGR'
            or self.buffers['images'].dtype != torch.uint8
            or tuple(self.buffers['detections'].shape) != (5,12)
            or tuple(self.buffers['flags'].shape) != (5,4)):
            raise ValueError('Incompatible integrated engine I/O contract')

    @torch.inference_mode()
    def __call__(self,frame,*,timing=False):
        check_frame(frame)
        # Handles SDK views with non-contiguous/negative strides as well.
        self.host_input.copy_(torch.from_numpy(np.ascontiguousarray(frame))[None])
        with torch.cuda.stream(self.stream):
            self.buffers['images'].copy_(self.host_input,non_blocking=True)
            if timing:
                self.start_event.record(self.stream)
            if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
                raise RuntimeError('TensorRT execution failed')
            if timing:
                self.end_event.record(self.stream)
            detections = self.buffers['detections'].float().cpu().numpy()
            flags = self.buffers['flags'].cpu().numpy()
        objects = format_objects(detections,flags)
        return (objects,self.start_event.elapsed_time(self.end_event)) if timing else objects

    def predict(self,frame,*,timing=False):
        return self(frame,timing=timing)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--warmup',type=int,default=100)
    parser.add_argument('--iterations',type=int,default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.warmup < 1 or args.iterations < 1:
        parser.error('Warmup and iterations must be positive')
    torch.set_num_threads(4)
    detector = Detector(args.engine)
    paths = sorted(p for p in args.source.iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg')) if args.source.is_dir() else [args.source]
    if not paths:
        parser.error('No images found')
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frame = np.asarray(image.convert('RGB'))[:,:,::-1].copy()
        check_frame(frame)
        frames.append(frame)
    report = dict(**gpu_info(),contract=INTEGRATED_CONTRACT,nms=False,
                  engine_sha256=detector.metadata['engine_sha256'])
    if args.benchmark:
        for i in range(args.warmup):
            detector(frames[i%len(frames)])
        torch.cuda.synchronize()
        gpu_times,total_times = [],[]
        for i in range(args.iterations):
            start = time.perf_counter()
            objects,gpu_ms = detector(frames[i%len(frames)],timing=True)
            json.dumps(objects)
            total_times.append((time.perf_counter()-start)*1000)
            gpu_times.append(gpu_ms)
        report.update(iterations=args.iterations,warmup=args.warmup,frames=len(frames),
                      input_shape=list(detector.INPUT_SHAPE),input_format='BGR',
                      input_dtype='uint8',model_input_size=[416,256],batch=1,
                      integrated_engine=latency_summary(gpu_times),
                      bgr_to_json=latency_summary(total_times),
                      target_gpu_verified='5060' in report['gpu'],
                      excludes=['camera acquisition','image decoding','RGB/BGR file conversion',
                                'robot communication','disk/network output'])
    else:
        report['frames'] = [dict(file=str(path),objects=detector(frame)) for path,frame in zip(paths,frames)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report if args.benchmark else {'frames':len(frames),'output':str(args.output)},indent=2))


if __name__ == '__main__':
    main()
