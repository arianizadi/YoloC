from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8x.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolov8n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("yolov8x.onnx")

# Run inference
results = onnx_model("https://ultralytics.com/images/bus.jpg")
