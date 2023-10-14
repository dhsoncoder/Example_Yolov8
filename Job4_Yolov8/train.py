from ultralytics import YOLO

import os

#train với dữ liệu nhà phát hành
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format

# train với dữ liệu custom
# Training classification
# Load a model
model1 = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results1 = model1.train(data='./dataset/cc-1', epochs=100, imgsz=640)

# Training detection
# Load a model
model2 = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results2 = model2.train(data='./dataset/Detect-chicken-3', epochs=100, imgsz=640)

# Training segmentation
# Load a model
model3 = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
results3 = model2.train(data='./dataset/Detect-chicken-3', epochs=100, imgsz=640)

