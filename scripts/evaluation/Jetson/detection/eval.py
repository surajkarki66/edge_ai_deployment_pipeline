from ultralytics import YOLO

model = YOLO("best.engine")

metrics = model.val(
    data="./hpd2/data.yaml",   # path to your dataset config
    split="test",       # use the 'test' split defined in data.yaml
    imgsz=416,          # must match engine's export size
    save_json=True,     # saves detailed results to a JSON file
)

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")
