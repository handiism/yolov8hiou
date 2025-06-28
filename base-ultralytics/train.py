from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolov8.yaml')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='coco.yaml', epochs=300, imgsz=640, batch=8,)

    # # Load a model
    # model = YOLO('F:/yolov8/ultralytics/runs/detect/train32/weights/last.pt')
    # #
    # # # Resume training
    # results = model.train(resume=True)

