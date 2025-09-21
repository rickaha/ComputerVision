from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/path/yolo11n.pt")  # path to your pretrained or custome YOLO11 model

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("/path/yolo11n.onnx", task='detect')

# Run inference
results = onnx_model("https://ultralytics.com/images/bus.jpg")
