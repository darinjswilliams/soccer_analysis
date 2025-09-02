from ultralytics import YOLO

model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

#inference on an video
results = model.predict("input_videos/D35bd9041_1.mp4", save=True, stream=True)  # predict on an image
print('=' * 40)
print(results[0])
for box in results[0].boxes:
    print(box.xyxy, box.conf, box.cls)  #s xyxy (tensor), confidence (tensor), class (tensor)
    



