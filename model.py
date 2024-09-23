from ultralytics import YOLO

# List of YOLOv8 models to convert
yolo_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]

for model_name in yolo_models:
    # Load the YOLOv8 model
    model = YOLO(model_name)

    # Export the model to ONNX format
    onnx_model_name = model_name.replace(".pt", ".onnx")
    model.export(format="onnx")  # creates corresponding .onnx file

    # Load the exported ONNX model
    onnx_model = YOLO(onnx_model_name)

    # Run inference
    results = onnx_model("https://ultralytics.com/images/bus.jpg")
