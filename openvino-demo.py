import os
import sys
import ast
import tkinter as tkinter
import cv2
import time
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter.scrolledtext as tks

from collections import OrderedDict
import argparse
import numpy as np
from queue import Queue

try:
    pyrs = 1
    import pyrealsense2 as rs
except:
    print("pyrealsense: no")
    pyrs = 0


# ov direct from yolov8-object-det notebook
#from typing import Tuple, Dict


# multi-proc for multi-cameras
import subprocess


import time
from paho.mqtt import client as mqtt_client
broker = 'broker.hivemq.com'
broker = '198.175.88.142'
broker = ''
port = 1883
topic = "/roi/bottle-1/describe_the_item_in_the_image"
client_id = 'test-client/`'
ready = 0

shopping_cart_ui = None
gen_ai_ui = None

parser = argparse.ArgumentParser()
#parser.add_argument("--source", nargs="?", default="sample.mp4")
parser.add_argument("--source", "-s", action='append', help='USB or RTSP or FILE sources', required=True)
parser.add_argument("--enable_cls_preprocessing", default=False, action="store_true")
parser.add_argument("--reclassify_interval", nargs="?", type=int, default=1)
parser.add_argument("--max_tracked_objects", nargs="?", type=int, default=20)
parser.add_argument("--show", default=False, action="store_true")
parser.add_argument("--cls_model")
parser.add_argument("--enable_int8", default=False, action="store_true")
parser.add_argument("--device_name")
# 384, 640
parser.add_argument("--det_imgsz", type=int, nargs='+', default=[384,640])
parser.add_argument("--cls_imgsz", type=int, nargs='+', default=[224,224])

args = parser.parse_args()
source = args.source
cls_model_name = args.cls_model
enable_cls_preprocessing = args.enable_cls_preprocessing
show_gui = args.show
reclassify_interval = args.reclassify_interval
max_tracked_objects = args.max_tracked_objects
det_imgsz = args.det_imgsz
cls_imgsz = args.cls_imgsz
en_int8 = args.enable_int8
device_name = args.device_name

if device_name is None or device_name == "":
    device_name = "GPU"

if cls_model_name is None or cls_model_name == "" or "efficient" in cls_model_name:
    cls_model_name = "efficientnet-b0.xml"
elif "resnet" in cls_model_name:
    cls_model_name = "resnet-50-tf_i8.xml"
else:
    cls_model_name = "efficientnet-b0.xml"
print("classification_model: ", cls_model_name)
print("enable_int8: ", en_int8)
print("device_name: ", device_name)
print("show_gui: ", show_gui)

if reclassify_interval < 1:
    print("reclassify_interval < 1 is not allowed. reclassify_interval reset to 1.")
    reclassify_interval = 1

class LRUCache:
    def __init__(self, max_tracked_objects):
        self.data = OrderedDict()
        self.capacity = max_tracked_objects

    def empty(self):
        return not self.data

    def pop(self) -> int:
        key = list(self.data.keys())[0]
        ret = self.data[key]
        del self.data[key]
        return key,ret

    def get(self, key: int) -> int:
        if key not in self.data:
            return None
        else:
            self.data.move_to_end(key)
            return self.data[key]

    def put(self, key: int, value) -> None:
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
    def clear(self):
        self.data = OrderedDict()

class timed_tracked_object:
    def __init__(self, cur_time, trackid, label):
        self.cur_time = cur_time
        self.trackid = trackid
        self.label = label

class Box:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.id = None

def on_connect(client, userdata, flags, rc):
    global ready
    if rc != 0:
        print("Failed to connect, return code %d", rc)
        ready = 0
    else:
        ready = 1

def connect_mqtt():
    #client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1, client_id)
    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port, 60)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        global tracked_mqtt_results

        #print("On msg: ", msg.topic)
        #print(msg.payload)
        topicArr = msg.topic.split("/")
        item_id = topicArr[2]
        item_res = msg.payload.decode("utf-8")
        #item_prompt = topicArr[3]
        #print("GenAI: ", item_id, item_res)
        tracked_mqtt_results.put(item_id, item_res) # + "|" + item_prompt)
        
    client.subscribe("/roi-res/#")
    client.on_message = on_message


def publish(client, img_bytes, item_id, prompt):
    #print("Sending bytes: ", img_bytes)
    #print(isinstance(img_bytes, bytes))
    #res = client.publish(topic, payload=img_bytes, qos=2, retain=False)

    prompt = prompt.replace(" ", "_")
    topic = "/roi/" + item_id + "/" + prompt
    print("Topic: '" + topic + "', Prompt: '" + prompt + "' was sent")

    res = client.publish(topic, img_bytes, 2, retain=False)
    sts = res[0]
    if sts == 0:
        pass
        print("JPG sent successfully")
    else:
        print("JPG send failed")

def do_update_annotated_frame_detection(annotated_frame, det_imgsz, xyxy, label) -> None:
    #print("----------->: ", label, " ",  xyxy[0], " " , xyxy[1], " ", xyxy[2], " " , xyxy[3], " ", annotated_frame.shape)

    scaleX = annotated_frame.shape[1] / det_imgsz[1]
    scaleY = annotated_frame.shape[0] / det_imgsz[0]
    scaleX = 1
    scaleY = 1
    #scaleX =1
    #scaleY = 1
    x = int(xyxy[0] * scaleX)
    y = int(xyxy[1] * scaleY)
    w = int(xyxy[2] * scaleX)
    h = int(xyxy[3] * scaleY)

    #print("----->:  " , x, " ", y, " ", w, " ", h, annotated_frame.shape, " " , det_imgsz)

    cv2.putText(
        annotated_frame,
        label,
        (x, y),
        0,
        1, # SF
        (0, 0, 255),  # Text Color
        thickness=3,      # Text thickness
        lineType=cv2.LINE_AA
    )

    cv2.rectangle(annotated_frame, 
            (x,y),
            (w,h), 
            (0,0,255), 
            thickness = 3
    ) 
    #return np.asarray(img)


# MQTT
#client = connect_mqtt()
#subscribe(client)
#client.loop_start()
#while ready != 1:
#    time.sleep(1)
#print("Connecting to broker ")


# Tracked objects ... need to refactor into a single cache
tracked_objects = LRUCache(max_tracked_objects)
tracked_objects_time = LRUCache(max_tracked_objects)
tracked_objects_state = LRUCache(max_tracked_objects)
tracked_mqtt_results = LRUCache(max_tracked_objects) # Queue(max_size = max_tracked_objects)


def close_window(event):
    sys.exit()


def promptTextFocused(event):
    entry = event.widget
    entry.tag_add(tkinter.SEL, "1.0", tkinter.END)
    entry.mark_set(tkinter.INSERT, "1.0")
    entry.see(tkinter.INSERT)



def promptTextClicked(text, item_id, jpgImg, entry, promptLabel, lblres):
    #print("prompt: ", text, item_id)
    entry.delete('1.0', tkinter.END) #, 'end-1c')
    entry.insert('1.0', 'Enter text and press <ENTER>')
    entry.tag_add(tkinter.SEL, "1.0", tkinter.END)
    entry.mark_set(tkinter.INSERT, "1.0")
    entry.see(tkinter.INSERT)

    publish(client, jpgImg.tobytes(), item_id, text)

    promptLabel.config(text=text)
    promptLabel.update_idletasks()

    lblres.delete('1.0', tkinter.END)
    lblres.insert(1.0, "...")

    return "break"


genAiItems = 0
rowGen = 0
colGen = 0
x = 0
y = 0
def addGenAIResult(root, item_img, item_id, item_prompt, img_bytes):
    global genAiItems
    global rowGen
    global colGen
    global x
    global y
    frameHeight = 400
    frameWidth = 300
    barWidth = 10
    frameColor = "gray64"
    frameColor = "white"
    #print("Adding genAIView widget: ", item_id)
    f = tkinter.Frame(root, name=str(item_id), width=frameWidth, height=frameHeight, bg=frameColor)
    #fblank = tkinter.Frame(root, width=barWidth,height=frameHeight)

    itemWidth = 50
    itemHeight = 85
    #x = colGen * frameWidth
    #y = rowGen * frameHeight

    #if colGen > 0:
    #    x = x + barWidth

    #if x + frameWidth >= windowWidth:
    #    colGen = 0
    #    rowGen = rowGen + 1
    #    #x = colGen * frameWidth
    #    #y = rowGen * frameHeight
    #    #colGen = colGen + 1
    #    print("Start new row!!")
    ##else:
    ##    colGen = colGen + 1

    #print("Placing pic: ", x, y, x+frameWidth, y+frameHeight) 
    f.pack()
    f.pack_propagate(0)
    #fblank.pack()

    x = colGen * frameWidth
    y = rowGen * frameHeight

    if colGen > 0:
        x = x + (colGen*barWidth)
    if rowGen > 0:
        y = y + (rowGen*barWidth)

    if x + frameWidth >= windowWidth:
        colGen = 0
        rowGen = rowGen + 1
        x = 0
        y = (rowGen * frameHeight) + (rowGen * barWidth) 

    #print("Placing pic: ", x, y, colGen, rowGen) #, x+frameWidth, y+frameHeight)

    f.place(x=x, y=y)

    colGen = colGen +1
    genAiItems = genAiItems + 1

    matImg = cv2.resize(item_img, (itemWidth,itemHeight), interpolation=cv2.INTER_LINEAR)
    b,g,r = cv2.split(matImg)
    img = cv2.merge((r,g,b))
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    
    promptLabel = tkinter.Label(f, name="lblprompt"+str(item_id), text=item_prompt+"\n", wraplength=200, anchor=tkinter.W, justify="left", bg=frameColor)
    promptLabel.pack()

    itemPicture = tkinter.Label(f, text=" ", image=imgtk)
    itemPicture.pack() #.grid(row=startingRowIdxShoppingCart,column=0, sticky=tkinter.W)
    itemPicture.image = imgtk

    blnkLabel = tkinter.Label(f, text=" ", bg=frameColor, justify=tkinter.LEFT, anchor=tkinter.W, width=frameWidth-2)
    blnkLabel.pack()

    #itemRes.pack()

    itemRes = tks.ScrolledText(f, name="lblres"+str(item_id), width=frameWidth, height=10, wrap='word', bd=0)
    itemRes.insert(1.0, "...")
    itemRes.pack()

    blnkLabel2 = tkinter.Label(f, text=" ", bg=frameColor, justify=tkinter.LEFT, anchor=tkinter.W, width=frameWidth-2)
    blnkLabel2.pack()

    promptEntry = tkinter.Text(f, name="entry"+str(item_id), width=frameWidth, height=2)
    promptEntry.bind('<Return>', (lambda event: promptTextClicked(promptEntry.get("1.0", 'end-1c'), str(item_id), img_bytes, promptEntry,promptLabel, itemRes)))

    promptEntry.bind('<FocusIn>', promptTextFocused)
    promptEntry.insert(1.0, 'Enter text and press <ENTER>')

    promptEntry.pack()
    


def updateGenAIResult(ui_frame, item_id, item_result, is_first):
    try:
        found_ui = ui_frame.nametowidget(str(item_id)) if is_first else ui_frame
    except:
        return
    #print(found_ui, found_ui._name)

    if found_ui._name == "lblres"+str(item_id):
        #print("found lbl toupdate")
        found_ui.delete(1.0, 'end')
        found_ui.insert(1.0, item_result)
        found_ui.update_idletasks()
    #if found_ui._name == "lblprompt"+str(item_id):
    #    found_ui.config(text = item_prompt)
    #    found_ui.update_idletasks()

    #print(found_ui.winfo_children())
    for child in found_ui.winfo_children():
        updateGenAIResult(child, item_id, item_result, False)

    #lbl = found_ui.nametowidget("lblres"+str(item_id))
    #lbl.config(text = item_result)
    #lbl.update_idletasks()

    #print(lbl)

def addShoppingCartItem(cartFrame, startingRowIdxShoppingCart, itemText, itemImg):
    # Add detected item in shopping cart view
    itemWidth = 50
    itemHeight = 85
    #print(itemImg.shape)
    matImg = cv2.resize(itemImg, (itemWidth,itemHeight), interpolation=cv2.INTER_LINEAR) 
    b,g,r = cv2.split(matImg)
    img = cv2.merge((r,g,b))
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    itemPictureFrame = tkinter.Frame(cartFrame, height=itemHeight, width=itemWidth)
    #itemPictureFrame.grid_propagate(0)
    itemPictureFrame.grid(row=startingRowIdxShoppingCart, column=0, sticky=tkinter.W, pady=2)
    itemPicture = tkinter.Label(itemPictureFrame, text=itemImg, image=imgtk)
    itemPicture.image = imgtk
    itemPicture.grid(row=startingRowIdxShoppingCart,column=0, sticky=tkinter.W)
    itemLabel = tkinter.Label(cartFrame, text=itemText)
    itemLabel.grid(row=startingRowIdxShoppingCart,column=1, sticky=tkinter.W, padx=5)
    startingRowIdxShoppingCart = startingRowIdxShoppingCart + 1
    return startingRowIdxShoppingCart

def do_bit_on_roi(cls_model, roi, cls_imgsz, output_layer):
    resized_roi = cv2.resize(roi, (64,64), interpolation=cv2.INTER_LINEAR) 
    resized_roi = tf.reshape(resized_roi, [1,64,64,3])
    result_infer = cls_model([resized_roi])[output_layer]
    #print(result_infer)
    #print("Done bit infer!")
    result_index = np.argmax(result_infer)
    return result_index

def do_efficientnet_on_roi(cls_model, roi, cls_imgsz, output_layer):
    #print(roi.shape, " ", roi.size)

    # https://docs.openvino.ai/2022.3/omz_models_model_resnet_50_tf.html
    resized_roi = roi
    mean = [0,0,0]
    resized_roi = cv2.dnn.blobFromImage(resized_roi, 1, (cls_imgsz[1],cls_imgsz[0]),mean,1, crop=False)
    return cls_model_names[np.argmax(cls_model([resized_roi])[output_layer])]


def do_yolo_on_frame(det_model, frame, det_imgsz, output_layer):    
    #resized_roi = cv2.resize(frame, (det_imgsz[0], det_imgsz[1]), interpolation=cv2.INTER_LINEAR)
    resized_roi = frame
    resized_roi = cv2.dnn.blobFromImage(resized_roi, 1/255, (det_imgsz[1],det_imgsz[0]),[0,0,0],1, crop=False)
    res = det_model(resized_roi)[output_layer]
    res = cv2.transpose(res[0])
    
    nmsThreshold = 0.6
    confThreshold = 0.65

    boxes = []
    scores = []
    classIds = []
    for x in res:
        confs = x[4:]
        (min_conf, max_conf, min_loc, (x1, classIdIdx)) = cv2.minMaxLoc(confs)
        if max_conf >= .65:
            box = [x[0] - (0.5 * x[2]), x[1] - (0.5 * x[3]), x[2], x[3]]
            boxes.append(box)
            scores.append(max_conf)
            classIds.append(classIdIdx)

    scaleX = frame.shape[1] / det_imgsz[1] 
    scaleY = frame.shape[0] / det_imgsz[0]


    boxIds = cv2.dnn.NMSBoxes(boxes, scores, confThreshold, nmsThreshold) #, eta) 

    detections = []
    results = []
    for boxId in boxIds:
        xyxy = [ round(boxes[boxId][0]*scaleX), round(boxes[boxId][1]*scaleY), round((boxes[boxId][2]+boxes[boxId][0])*scaleX), round((boxes[boxId][3]+boxes[boxId][1])*scaleY) ]
        box = Box(xyxy, scores[boxId], classIds[boxId])
        results.append({"boxes": [box]})

    return results

def clear_shopping_cart():
    global genAiItems, rowGen, colGen,x,y

    # clean genai table info
    genAiItems = 0
    rowGen = 0
    colGen = 0
    x = 0
    y = 0

    # clear shopping cart
    isFirst = True
    for w in shopping_cart_ui.winfo_children():
        if not isFirst:
            w.destroy()
        else:
            isFirst = False

    # clear gen ai view
    for w in gen_ai_ui.winfo_children():
        #print("Destroy: ", w._name)
        w.destroy()
    tracked_objects.clear()
    tracked_objects_time.clear()
    tracked_objects_state.clear()

# Main Window Rendering
windowHeight=480
windowWidth=640
startingRowIdxShoppingCart=1
# Main View
cartFrameWidth = 200
cartFrameHeight = windowHeight - 100
videoFrameWidth = windowWidth - cartFrameWidth
videoFrameHeight = cartFrameHeight

if show_gui:
    window = tkinter.Tk()
    window.title("Vision Self-Checkout Demo")
    window.bind('<Escape>', close_window)
    window.geometry(str(windowWidth) + 'x' + str(windowHeight))
    #window.maxsize(width=windowWidth, height=windowHeight)

    # Tab Widget
    tabWidget = ttk.Notebook(window)
    tabCameraView = tkinter.Frame(tabWidget)

    tabWidget.add(tabCameraView, text='Main')
    tabWidget.pack(expand=1, fill="both")


    videoFrame = tkinter.Frame(tabCameraView, bg="black", width=videoFrameWidth, height=videoFrameHeight)
    cartFrame = tkinter.Frame(tabCameraView, width=cartFrameWidth, height=cartFrameHeight)
    shoppingClearButton = tkinter.Button(tabCameraView, text="Clear Cart", command=clear_shopping_cart)

    videoFrame.grid_propagate(False)
    cartFrame.grid_propagate(0)

    shopping_cart_ui = cartFrame


    videoFrame.grid(         row=0, column=0, sticky=tkinter.NW, pady=10, rowspan=2)
    cartFrame.grid(          row=0, column=1, sticky=tkinter.NE, pady=10, padx=10, rowspan=1)
    shoppingClearButton.grid(row=1, column=1, sticky=tkinter.NW, padx=10, pady=5, rowspan=1)

    shoppingCartLabel = tkinter.Label(cartFrame, text="               Shopping Cart")
    shoppingCartLabel.grid(row=0, column=0, sticky=tkinter.NW, columnspan=2, pady=10, padx=0)

# Video playback / inference
caps = []

for s in source:
    if "/dev/video" in s:
        cap = cv2.VideoCapture(s)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        try:
            print("Loading video stream.")
            cap = cv2.VideoCapture(s, cv2.CAP_GSTREAMER)
            print("Video stream loaded.")
        except:
            print("Retrying loading video stream.")
            cap = cv2.VideoCapture(s, cv2.CAP_GSTREAMER)
            print("Video stream loaded.")
    caps.append(cap)

## Source model defaults to 384x640
det_model_names = ""
with open("yolo-names.txt", "r") as text_file:
    det_model_names = ast.literal_eval(text_file.read())

if en_int8:
    model_name = "yolov8n-int8.xml"
else:
    model_name = "yolov8n.xml"

print("Loading detection and classification models.")
from openvino.runtime import Core
ie = Core()
ov_model = ie.read_model(model_name)
ov_config = {}
ov_model.reshape({0: [1,3,det_imgsz[0], det_imgsz[1]]})

# Use if GPU or AUTO AND GPU AVAILABLE
if "GPU" in device_name or "AUTO" == device_name:
    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
else:
    ov_config = {}
model = ie.compile_model(ov_model, config=ov_config, device_name=device_name)
output_layer = model.output(0)

# Classification model
cls_model_names = ""
ov_cls_model = ie.read_model(model=cls_model_name)

if cls_model_name == "efficientnet-b0.xml":
    ## Efficientnet-B0
    with open("efficientnet.labels", "r") as text_file:
        cls_model_names = ast.literal_eval(text_file.read())
elif "bit" in cls_model_name:
    ## BiT: BiT_M_R50x1_10C_50e_IR/1/FP32/64_64_3/model_64_64_3.xml
    ov_cls_model.reshape({0:[1,64,64,3]})
else:
    ## Resnet-50
    with open("resnet.labels", "r") as text_file:
        cls_model_names = ast.literal_eval(text_file.read())

cls_model = ie.compile_model(model=ov_cls_model, device_name=device_name, config=ov_config)
cls_output_layer = cls_model.output(0)
print("Detection and classification models loaded.")

frame_count = 0
skip_frame_reclassify = False

print("capturing video(s) from: ", source)
print("Running ", model_name, " OpenVINO")
#sfilter = ["banana", "apple", "orange", "broccoli", "carrot", "bottle"]
#sfilter = ["bottle"]
sfilter = ["person", "bottle"]
vidPicture = None
numOfCaps = len(caps)
capIdx = 0
while True:
    cap = caps[capIdx]
    success, frame = cap.read()
    if not cap.isOpened() or not success:
        print('video feed done...')
        break


    annotated_frame = None

    if capIdx == 0:
        start_time = time.time()


    # Add any GenAI results and render them in GenAI widget(s)
    while not tracked_mqtt_results.empty():
        trackid, res = tracked_mqtt_results.pop()
        #resArr = res.split("|")
        #print(trackid, res[0], res[1])
        #updateGenAIResult(tabGenAIView, trackid, res, True)


    results = do_yolo_on_frame(model, frame.copy(), det_imgsz, output_layer)    
    send_lvlm_processing = True

    #print("process results......", videoFrameWidth, " ", videoFrameHeight)
    if show_gui:
        annotated_frame = frame.copy()
    for result in results:
        boxes = result["boxes"]
        result_label = ""
        result_label_cls = ""

        for box in boxes:
            result_label = det_model_names[int(box.cls)]
            if not any(s in result_label for s in sfilter):
                send_lvlm_processing = False
                break
        
        if not send_lvlm_processing:
            continue

        #if show_gui:
        #    do_update_annotated_frame_detection(annotated_frame, det_imgsz, box.xyxy, result_label)
            #annotated_frame = result.plot()


        #print("No hand in pic-->", result_label, "person" in result_label)

        # Add any GenAI results and render them in GenAI widget(s)
        #while not tracked_mqtt_results.empty():
        #    trackid, res = tracked_mqtt_results.pop()
        #    #resArr = res.split("|")
        #    print(trackid, res[0], res[1])
        #    updateGenAIResult(tabGenAIView, trackid, res, True)


        #for box, track_id in zip(boxes, track_ids):        
        for box in boxes:
            c, conf, id = int(box.cls), float(box.conf), None if box.id is None else int(box.id.item())

            #origFrame = result.orig_img.copy()
            origFrame = frame.copy()


            roi = None #save_one_box(torch.tensor(box.xyxy), origFrame,BGR=True, save=False)
            x = box.xyxy[0]
            y = box.xyxy[1]
            w = box.xyxy[2] 
            h = box.xyxy[3]
            if x < 0:
                x = 0
            if y < 0:
                y = 0

            roi = origFrame[int(y) : int(h), int(x) : int(w), :: (1 if "eff" in cls_model_name else -1)]
            #cv2.imwrite("roi.jpg", roi)
            result_label = det_model_names[c]
            cls_label = do_efficientnet_on_roi(cls_model, roi, cls_imgsz, cls_output_layer)
            if show_gui:
                do_update_annotated_frame_detection(annotated_frame, det_imgsz, box.xyxy, result_label + ": " + cls_label)

            print("Detected: ", result_label, " Classification: ", cls_label)
            result_label = cls_label


    if capIdx == numOfCaps -1:
        capIdx = 0        
        frame_count = frame_count + 1
    else:
        capIdx = capIdx + 1

    # Skip reclassification based on tracked objects and interval specified
    skip_frame_reclassify = frame_count % reclassify_interval != 0

    elapsed_time = time.time() - start_time
    frame_driver = 5
    if frame_count % frame_driver == 0:
        print("Seconds taken for last ", frame_driver, " frames in pipeline: ", elapsed_time)

    # Display the annotated frame
    if show_gui and capIdx == 0:
        if annotated_frame is None:
            annotated_frame = frame

        #cv2.imshow("YOLOv8 " + tracker + " Tracking" + " with OpenVINO" if use_openvino else "", annotated_frame)
        #print("Frame info: " , annotated_frame.shape, " ", videoFrameWidth, " ", videoFrameHeight)

        matImg = cv2.resize(annotated_frame, (videoFrameWidth,videoFrameHeight), interpolation=cv2.INTER_LINEAR)
        b,g,r = cv2.split(matImg)
        imgt = cv2.merge((r,g,b))
        imgt = Image.fromarray(imgt)
        imgtkk = ImageTk.PhotoImage(image=imgt)
        if not vidPicture is None:
            vidPicture.configure(image = imgtkk)
            vidPicture.image = imgtkk
        else:
            vidPicture = tkinter.Label(videoFrame, text=" ") #, image=imgtkk)
        #vidPicture.image = imgtkk
        vidPicture.grid(row=0,column=0, sticky=tkinter.W)
        vidPicture.update_idletasks()
        vidPicture.update()
        window.update_idletasks()
        window.update()


################
if show_gui:
    window.mainloop()
