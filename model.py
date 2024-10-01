from ultralytics import YOLO
import glob

# Find all .pt files in the current directory
yolo_models = glob.glob("*.pt")

for model_name in yolo_models:
    # Load the YOLOv8 model
    model = YOLO(model_name)

    # Export the model to ONNX format
    onnx_model_name = model_name.replace(".pt", ".onnx")
    model.export(format="onnx")  # creates corresponding .onnx file
