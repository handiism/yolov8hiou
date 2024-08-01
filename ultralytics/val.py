from ultralytics import YOLO

# Load a model

model = YOLO('F:/yolov8/ultralytics/best.pt')  # load a custom model

# Predict with the model
results = model('F:/yolov8/ultralytics/0.jpg')  # predict on an image

