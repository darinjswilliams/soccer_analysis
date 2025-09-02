from ultralytics import YOLO

model = YOLO("models/best.pt")  # load a pretrained model (recommended for training)

#inference on an video
results = model.predict("input_videos/D35bd9041_1.mp4", save=True)  # predict on an image
print(results[0])
print('=' * 50)
for box in results[0].boxes:
    print(box.xyxy, box.conf, box.cls)  #s xyxy (tensor), confidence (tensor), class (tensor)
    



