from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import ncnn


def load_export_metadata(path: str | Path) -> dict:
    """Parse Ultralytics NCNN ``metadata.yaml``"""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    imgsz = None
    names: dict[int, str] = {}
    end2end = None
    in_names = False
    in_imgsz = False

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        if line.startswith("end2end:"):
            end2end = line.split(":", 1)[1].strip().lower() in {"true", "1", "yes"}
            in_names = False
            in_imgsz = False
            continue

        if line.startswith("imgsz:"):
            in_imgsz = True
            in_names = False
            rest = line.split(":", 1)[1].strip()
            if rest and not rest.startswith("-"):
                imgsz = int(rest.strip("[] "))
            continue

        if in_imgsz and line.lstrip().startswith("-"):
            imgsz = int(line.split("-", 1)[1].strip())
            in_imgsz = False
            continue

        if line.startswith("names:"):
            in_names = True
            in_imgsz = False
            continue

        if in_names:
            stripped = line.lstrip()
            if not stripped[0].isdigit():
                in_names = False
                continue
            key, _, value = stripped.partition(":")
            names[int(key.strip())] = value.strip()
            continue

        in_names = False
        in_imgsz = False

    class_names = [names[i] for i in sorted(names)] if names else []
    return {
        "imgsz": imgsz,
        "class_names": class_names,
        "num_classes": len(class_names),
        "end2end": end2end,
    }


class YOLOv26NCNN:
    """NCNN runner for Ultralytics YOLO26 detect exports (one-to-many)."""

    def __init__(
        self,
        param_path,
        bin_path,
        input_size=640,
        num_classes=80,
        class_names=None,
        num_threads=4,
        use_vulkan=False,
        box_format="cxcywh",
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.num_threads = num_threads
        self.use_vulkan = use_vulkan
        self.box_format = box_format

        if len(self.class_names) != num_classes:
            raise ValueError(
                f"class_names length {len(self.class_names)} != num_classes {num_classes}"
            )

        self.net = ncnn.Net()

        # FP16 (matches pipeline ARM ncnn export with half=true)
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_arithmetic = True

        self.net.opt.num_threads = num_threads

        if use_vulkan:
            self.net.opt.use_vulkan_compute = True

        ret = self.net.load_param(param_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load param file: {ret}")

        ret = self.net.load_model(bin_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load model file: {ret}")

        print("YOLOv26 NCNN model loaded.")
        print("Note: NCNN disables YOLO26 end2end; decoding one-to-many + NMS.")
        print(f"Input size    : {input_size}x{input_size}")
        print(f"Classes       : {num_classes}")
        print(f"Threads       : {num_threads}")
        print(f"Vulkan        : {use_vulkan}")
        print(f"Box format    : {box_format}")
        print("FP16          : enabled")

    def letterbox(self, image):
        original_h, original_w = image.shape[:2]
        scale = min(
            self.input_size / original_w,
            self.input_size / original_h,
        )
        new_w = int(round(original_w * scale))
        new_h = int(round(original_h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full(
            (self.input_size, self.input_size, 3), 114, dtype=np.uint8
        )
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def iou(box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        intersection = inter_w * inter_h
        area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        area2 = (
            np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
            * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        )
        union = area1 + area2 - intersection
        return intersection / (union + 1e-7)

    def nms(self, boxes, scores, iou_threshold=0.45):
        if len(boxes) == 0:
            return []
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            current = order[0]
            keep.append(current)
            if len(order) == 1:
                break
            ious = self.iou(boxes[current], boxes[order[1:]])
            order = order[1:][ious < iou_threshold]
        return keep

    def decode_output(self, output):
        output = np.asarray(output, dtype=np.float32)
        print("Raw output shape:", output.shape)
        output = np.squeeze(output)
        print("Squeezed output shape:", output.shape)

        # Guard against confusing end2end ONNX/TRT layouts with NCNN output.
        if output.ndim == 2 and output.shape[-1] == 6 and output.shape[0] in {100, 200, 300}:
            raise RuntimeError(
                f"Output looks like end2end detections {output.shape} "
                "(x1,y1,x2,y2,conf,cls). NCNN YOLO26 exports do not produce this; "
                "use the one-to-many path: (4+nc, N) + host NMS."
            )

        if output.ndim != 2:
            raise RuntimeError(f"Unexpected output dimensions: {output.shape}")

        n_attr = 4 + self.num_classes
        if output.shape[0] == n_attr:
            output = output.T
        elif output.shape[1] == n_attr:
            pass
        else:
            # Helpful hint if --classes is wrong (e.g. COCO 80 vs custom 1).
            guess = None
            for axis in (0, 1):
                if output.shape[axis] > 4:
                    guess = output.shape[axis] - 4
                    break
            hint = f" Hint: tensor suggests num_classes≈{guess}." if guess else ""
            raise RuntimeError(
                f"Unexpected YOLOv26 NCNN output shape: {output.shape} "
                f"(expected {n_attr} attributes).{hint}"
            )

        print("Prediction tensor:", output.shape)
        boxes = output[:, :4].copy()
        cls_scores = output[:, 4:].copy()

        if cls_scores.min() < 0.0 or cls_scores.max() > 1.0:
            print("Scores outside [0,1] -> applying sigmoid")
            cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))

        class_ids = cls_scores.argmax(axis=1).astype(np.int32)
        scores = cls_scores.max(axis=1)

        print(f"Confidence min : {scores.min():.6f}")
        print(f"Confidence max : {scores.max():.6f}")
        print(f"Box raw range  : {boxes.min():.3f} .. {boxes.max():.3f}")

        if boxes.max() <= 2.0:
            print("Boxes look normalized -> scaling by input size")
            boxes *= self.input_size

        if self.box_format == "cxcywh":
            cx = boxes[:, 0].copy()
            cy = boxes[:, 1].copy()
            w = boxes[:, 2].copy()
            h = boxes[:, 3].copy()
            boxes[:, 0] = cx - w / 2.0
            boxes[:, 1] = cy - h / 2.0
            boxes[:, 2] = cx + w / 2.0
            boxes[:, 3] = cy + h / 2.0

        return boxes, scores, class_ids

    def __call__(self, image, conf_threshold=0.25, iou_threshold=0.45):
        original_h, original_w = image.shape[:2]
        input_image, scale, pad_x, pad_y = self.letterbox(image)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        mat = ncnn.Mat.from_pixels(
            input_image,
            ncnn.Mat.PixelType.PIXEL_RGB,
            self.input_size,
            self.input_size,
        )
        mat.substract_mean_normalize([], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])

        start = time.perf_counter()
        with self.net.create_extractor() as ex:
            ret = ex.input("in0", mat)
            if ret != 0:
                raise RuntimeError(f"Failed to set input: {ret}")
            ret, output = ex.extract("out0")
            if ret != 0:
                raise RuntimeError(f"Failed to extract out0: {ret}")
            output = np.array(output)
        inference_time = time.perf_counter() - start

        boxes, scores, class_ids = self.decode_output(output)

        mask = scores >= conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        print(f"Candidates after confidence filter: {len(boxes)}")

        if len(boxes) == 0:
            return [], inference_time

        final_indices = []
        for cls in np.unique(class_ids):
            cls_indices = np.where(class_ids == cls)[0]
            keep = self.nms(boxes[cls_indices], scores[cls_indices], iou_threshold)
            final_indices.extend(cls_indices[keep])
        final_indices = sorted(final_indices, key=lambda i: scores[i], reverse=True)

        detections = []
        for i in final_indices:
            x1, y1, x2, y2 = boxes[i]
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            x1 = max(0.0, min(x1, original_w - 1))
            y1 = max(0.0, min(y1, original_h - 1))
            x2 = max(0.0, min(x2, original_w - 1))
            y2 = max(0.0, min(y2, original_h - 1))
            class_id = int(class_ids[i])
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names[class_id]
                    if 0 <= class_id < len(self.class_names)
                    else str(class_id),
                    "confidence": float(scores[i]),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
            )
        return detections, inference_time


def draw_detections(image, detections):
    result = image.copy()
    img_h, img_w = result.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        name = det.get("class_name", f"class {det['class_id']}")
        label = f"{name} {det['confidence']:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = max(0, min(x1, img_w - tw - 1))
        text_y = y1 - 5 if y1 - th - 8 >= 0 else y1 + th + 5
        text_y = min(text_y, img_h - baseline - 1)
        cv2.rectangle(
            result,
            (text_x, text_y - th - baseline),
            (text_x + tw, text_y + baseline),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            result,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "YOLOv26 NCNN inference (Raspberry Pi). "
            "NCNN uses one-to-many + host NMS."
        )
    )
    parser.add_argument("--param", required=True, help="NCNN .param file")
    parser.add_argument("--bin", required=True, help="NCNN .bin file")
    parser.add_argument("--image", required=True, help="Input image")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional Ultralytics metadata.yaml next to the ncnn model",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Input size (must match export imgsz). Default: from --metadata or 640",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=None,
        help="Num classes. Default: from --metadata or 80 (COCO)",
    )
    parser.add_argument(
        "--names",
        default=None,
        help="Comma-separated class names (overrides --metadata names)",
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    parser.add_argument(
        "--box-format",
        choices=["cxcywh", "xyxy"],
        default="cxcywh",
        help="Ultralytics NCNN one-to-many box format (default: cxcywh)",
    )
    parser.add_argument("--vulkan", action="store_true", help="Enable Vulkan")
    parser.add_argument("--output", default="result.jpg", help="Output path")
    args = parser.parse_args()

    meta = {}
    if args.metadata:
        meta = load_export_metadata(args.metadata)
        print(f"Loaded metadata: imgsz={meta.get('imgsz')} "
              f"classes={meta.get('num_classes')} end2end={meta.get('end2end')}")
        if meta.get("end2end"):
            print(
                "WARNING: metadata says end2end=true, but NCNN export normally "
                "forces end2end=false. Decoding as one-to-many + NMS."
            )

    input_size = args.size or meta.get("imgsz") or 640
    if args.names:
        class_names = [n.strip() for n in args.names.split(",") if n.strip()]
    else:
        class_names = meta.get("class_names") or [f"class_{i}" for i in range(80)]
    num_classes = args.classes or meta.get("num_classes") or len(class_names)
    if len(class_names) != num_classes:
        if args.names is None and meta.get("class_names"):
            class_names = meta["class_names"]
            num_classes = len(class_names)
        else:
            raise ValueError(
                f"class name count {len(class_names)} != --classes/{num_classes}"
            )

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")
    print(f"Image size    : {image.shape[1]}x{image.shape[0]}")

    model = YOLOv26NCNN(
        param_path=args.param,
        bin_path=args.bin,
        input_size=input_size,
        num_classes=num_classes,
        class_names=class_names,
        num_threads=args.threads,
        use_vulkan=args.vulkan,
        box_format=args.box_format,
    )

    total_start = time.perf_counter()
    detections, inference_time = model(
        image, conf_threshold=args.conf, iou_threshold=args.iou
    )
    total_time = time.perf_counter() - total_start

    print()
    print("=" * 60)
    print("TIMING")
    print("=" * 60)
    print(f"NCNN inference : {inference_time * 1000:.2f} ms")
    print(f"Total          : {total_time * 1000:.2f} ms")
    print(f"NCNN FPS       : {1.0 / inference_time:.2f}")
    print(f"Total FPS      : {1.0 / total_time:.2f}")
    print("=" * 60)

    print()
    print("=" * 60)
    print("DETECTIONS")
    print("=" * 60)
    if not detections:
        print("No detections.")
    else:
        for i, det in enumerate(detections):
            print(
                f"[{i}] class={det['class_name']} "
                f"confidence={det['confidence']:.4f} "
                f"bbox={det['bbox']}"
            )
    print("=" * 60)

    result_image = draw_detections(image, detections)
    if not cv2.imwrite(args.output, result_image):
        raise RuntimeError(f"Failed to save {args.output}")
    print()
    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
