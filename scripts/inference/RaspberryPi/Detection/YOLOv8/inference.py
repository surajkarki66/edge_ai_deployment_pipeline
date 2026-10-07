import argparse
import time

import cv2
import numpy as np
import ncnn


class YOLOv8NCNN:
    def __init__(
        self,
        param_path,
        bin_path,
        input_size=416,
        num_classes=1,
        num_threads=4,
        use_vulkan=False,
        box_format="cxcywh",
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_threads = num_threads
        self.use_vulkan = use_vulkan
        self.box_format = box_format  # "cxcywh" (default YOLOv8) or "xyxy"

        self.net = ncnn.Net()

        # FP16
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

        print("YOLOv8n NCNN model loaded.")
        print(f"Input size    : {input_size}x{input_size}")
        print(f"Classes       : {num_classes}")
        print(f"Threads       : {num_threads}")
        print(f"Vulkan        : {use_vulkan}")
        print(f"Box format    : {box_format}")
        print("FP16          : enabled")

    # =========================================================
    # Letterbox
    # =========================================================

    def letterbox(self, image):
        original_h, original_w = image.shape[:2]

        scale = min(
            self.input_size / original_w,
            self.input_size / original_h,
        )

        new_w = int(round(original_w * scale))
        new_h = int(round(original_h * scale))

        resized = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        canvas = np.full(
            (self.input_size, self.input_size, 3), 114, dtype=np.uint8
        )

        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2

        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        return canvas, scale, pad_x, pad_y

    # =========================================================
    # IoU
    # =========================================================

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

    # =========================================================
    # NMS
    # =========================================================

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

    # =========================================================
    # Decode NCNN output
    # =========================================================

    def decode_output(self, output):
        output = np.asarray(output, dtype=np.float32)
        print("Raw output shape:", output.shape)

        output = np.squeeze(output)
        print("Squeezed output shape:", output.shape)

        if output.ndim != 2:
            raise RuntimeError(f"Unexpected output dimensions: {output.shape}")

        # Expected (4 + num_classes, N) -> transpose to (N, 4 + num_classes)
        n_attr = 4 + self.num_classes

        if output.shape[0] == n_attr:
            output = output.T
        elif output.shape[1] == n_attr:
            pass
        else:
            raise RuntimeError(
                f"Unexpected YOLOv8 output shape: {output.shape}"
            )

        print("Prediction tensor:", output.shape)

        boxes = output[:, :4].copy()
        cls_scores = output[:, 4:].copy()

        # -----------------------------------------------------
        # Scores: apply sigmoid only if they are not already
        # probabilities in [0, 1]
        # -----------------------------------------------------
        if cls_scores.min() < 0.0 or cls_scores.max() > 1.0:
            print("Scores outside [0,1] -> applying sigmoid")
            cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))

        class_ids = cls_scores.argmax(axis=1).astype(np.int32)
        scores = cls_scores.max(axis=1)

        print(f"Confidence min : {scores.min():.6f}")
        print(f"Confidence max : {scores.max():.6f}")

        # -----------------------------------------------------
        # Boxes: normalized -> pixels (letterbox space)
        # -----------------------------------------------------
        print(
            f"Box raw range  : {boxes.min():.3f} .. {boxes.max():.3f}"
        )
        if boxes.max() <= 2.0:
            print("Boxes look normalized -> scaling by input size")
            boxes *= self.input_size

        # -----------------------------------------------------
        # Boxes: center format -> corner format
        # -----------------------------------------------------
        if self.box_format == "cxcywh":
            cx = boxes[:, 0].copy()
            cy = boxes[:, 1].copy()
            w = boxes[:, 2].copy()
            h = boxes[:, 3].copy()

            boxes[:, 0] = cx - w / 2.0  # x1
            boxes[:, 1] = cy - h / 2.0  # y1
            boxes[:, 2] = cx + w / 2.0  # x2
            boxes[:, 3] = cy + h / 2.0  # y2

        return boxes, scores, class_ids

    # =========================================================
    # Inference
    # =========================================================

    def __call__(self, image, conf_threshold=0.25, iou_threshold=0.45):
        original_h, original_w = image.shape[:2]

        input_image, scale, pad_x, pad_y = self.letterbox(image)

        # BGR -> RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        mat = ncnn.Mat.from_pixels(
            input_image,
            ncnn.Mat.PixelType.PIXEL_RGB,
            self.input_size,
            self.input_size,
        )

        # RGB / 255
        mat.substract_mean_normalize(
            [], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
        )

        start = time.perf_counter()

        with self.net.create_extractor() as ex:
            ret = ex.input("in0", mat)
            if ret != 0:
                raise RuntimeError(f"Failed to set input: {ret}")

            ret, output = ex.extract("out0")
            if ret != 0:
                raise RuntimeError(f"Failed to extract out0: {ret}")

            # Convert inside the extractor scope, before it is released
            output = np.array(output)

        inference_time = time.perf_counter() - start

        boxes, scores, class_ids = self.decode_output(output)

        # Confidence filtering
        mask = scores >= conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        print(f"Candidates after confidence filter: {len(boxes)}")

        if len(boxes) == 0:
            return [], inference_time

        # Per-class NMS
        final_indices = []

        for cls in np.unique(class_ids):
            cls_indices = np.where(class_ids == cls)[0]
            keep = self.nms(
                boxes[cls_indices], scores[cls_indices], iou_threshold
            )
            final_indices.extend(cls_indices[keep])

        final_indices = sorted(
            final_indices, key=lambda i: scores[i], reverse=True
        )

        # Convert boxes back to original image
        detections = []

        for i in final_indices:
            x1, y1, x2, y2 = boxes[i]

            # Remove letterbox padding and undo scaling
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # Clip to original image
            x1 = max(0.0, min(x1, original_w - 1))
            y1 = max(0.0, min(y1, original_h - 1))
            x2 = max(0.0, min(x2, original_w - 1))
            y2 = max(0.0, min(y2, original_h - 1))

            detections.append(
                {
                    "class_id": int(class_ids[i]),
                    "confidence": float(scores[i]),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

        return detections, inference_time


# =============================================================
# Draw detections
# =============================================================

def draw_detections(image, detections):
    result = image.copy()
    img_h, img_w = result.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])

        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"class {det['class_id']} {det['confidence']:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Keep the label inside the image
        text_x = max(0, min(x1, img_w - tw - 1))
        text_y = y1 - 5 if y1 - th - 8 >= 0 else y1 + th + 5
        text_y = min(text_y, img_h - baseline - 1)

        # Background for readability
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


# =============================================================
# Main
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="YOLOv8n NCNN inference")

    parser.add_argument("--param", required=True, help="NCNN .param file")
    parser.add_argument("--bin", required=True, help="NCNN .bin file")
    parser.add_argument("--image", required=True, help="Input image")
    parser.add_argument("--size", type=int, default=416, help="Input size")
    parser.add_argument("--classes", type=int, default=1, help="Num classes")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    parser.add_argument(
        "--box-format",
        choices=["cxcywh", "xyxy"],
        default="cxcywh",
        help="Box format produced by the model output",
    )
    parser.add_argument("--vulkan", action="store_true", help="Enable Vulkan")
    parser.add_argument("--output", default="result.jpg", help="Output path")

    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    print(f"Image size    : {image.shape[1]}x{image.shape[0]}")

    model = YOLOv8NCNN(
        param_path=args.param,
        bin_path=args.bin,
        input_size=args.size,
        num_classes=args.classes,
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

    if len(detections) == 0:
        print("No detections.")
    else:
        for i, det in enumerate(detections):
            print(
                f"[{i}] class={det['class_id']} "
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
