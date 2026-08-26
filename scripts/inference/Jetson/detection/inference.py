from ultralytics import YOLO

model = YOLO("best.engine")

results = model.predict(source="test.jpg", show=False, save=True)
