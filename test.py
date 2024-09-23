import io

import cv2

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x.pt")

# Run inference on an image using GPU
results = model.predict(source="bus.jpg", device=0)  # device=0 specifies the first GPU

# Plot inference results
plot = results[0].plot()  

# Convert the plot to an image
image = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)

# Save the image to a PNG file
cv2.imwrite("output.png", image)
