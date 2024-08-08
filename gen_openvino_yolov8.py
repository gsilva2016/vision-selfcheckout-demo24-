from ultralytics import YOLO

model_name = "yolov8n.pt"
det_imgsz = (384,640)
model = YOLO(model_name)
model_names = model.names

path = model.export(format='openvino', imgsz=det_imgsz, half=True)
print("OpenVINO ", model_name, " created: ", path)

path = model.export(format='openvino', imgsz=det_imgsz, int8=True)
print("OpenVINO int8 ", model_name, " created: ", path)

with open("yolo-names.txt", "w") as text_file:
    print(list(model_names.values()), file=text_file)




