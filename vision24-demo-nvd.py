import os
import sys
import tkinter as tkinter
import cv2
import time
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter.scrolledtext as tks

from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
from collections import OrderedDict
import argparse
import numpy as np
from queue import Queue


import tensorflow.compat.v2 as tf

# ov direct from yolov8-object-det notebook
import torch
from ultralytics.utils import ops
from typing import Tuple, Dict

import weaviate
import base64

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
parser.add_argument("--tracker", nargs="?", default="botsort.yaml", help="botsort.yaml (Default) or bytetrack.yaml")
#parser.add_argument("--source", nargs="?", default="sample.mp4")
parser.add_argument("--source", "-s", action='append', help='USB or RTSP or FILE sources', required=True)
parser.add_argument("--model", nargs="?", default="yolov8n.pt")
parser.add_argument("--cls_model", nargs="?", default="yolov8n-cls.pt")
parser.add_argument("--enable_cls_preprocessing", default=False, action="store_true")
parser.add_argument("--use_openvino", default=False, action="store_true")
parser.add_argument("--reclassify_interval", nargs="?", type=int, default=1)
parser.add_argument("--max_tracked_objects", nargs="?", type=int, default=20)
parser.add_argument("--show", default=True, action="store_true")
# 384, 640
parser.add_argument("--det_imgsz", type=int, nargs='+', default=[384,640])
parser.add_argument("--cls_imgsz", type=int, nargs='+', default=[224,224])

args = parser.parse_args()
model_name = args.model
cls_model_name = args.cls_model
source = args.source
tracker = args.tracker
enable_cls_preprocessing = args.enable_cls_preprocessing
show_gui = args.show
use_openvino = args.use_openvino
reclassify_interval = args.reclassify_interval
max_tracked_objects = args.max_tracked_objects
det_imgsz = args.det_imgsz
cls_imgsz = args.cls_imgsz


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


class VerticalScrolledFrame(ttk.Frame):
    def __init__(self, parent, *args, **kw):
        ttk.Frame.__init__(self, parent, *args, **kw)

        # Create a canvas object and a vertical scrollbar for scrolling it.
        vscrollbar = ttk.Scrollbar(self, orient=tkinter.VERTICAL)
        vscrollbar.pack(fill=tkinter.Y, side=tkinter.RIGHT, expand=False)
        canvas = tkinter.Canvas(self, bd=0, highlightthickness=0,
                           yscrollcommand=vscrollbar.set)
        canvas.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)
        vscrollbar.config(command=canvas.yview)

        # Reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # Create a frame inside the canvas which will be scrolled with it.
        self.interior = interior = ttk.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=tkinter.NW)

        # Track changes to the canvas and frame width and sync them,
        # also updating the scrollbar.
        def _configure_interior(event):
            # Update the scrollbars to match the size of the inner frame.
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # Update the canvas's width to fit the inner frame.
                canvas.config(width=interior.winfo_reqwidth())
        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # Update the inner frame's width to fill the canvas.
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)

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


    #if rowGen > 0:
    #    f.place(x=x, y=y+barWidth)
    #else:
    #    f.place(x=x, y=y) # expand=0, fill=tkinter.NONE) 
    
    #if colGen-1 > 0:
    #    fblank.place(x=x+frameWidth-barWidth,y=y)
    #    print("Placing bar: ", x+frameWidth-barWidth, y)
    #else:
    #    fblank.place(x=x+frameWidth,y=y+barWidth)
    #    print("Placing bar: ", x+frameWidth, y)

    # fill=tkinter.BOTH, expand=1) # .grid(column=genAiItems, sticky=tkinter.NW, padx=5, pady=5)
    genAiItems = genAiItems + 1

    matImg = cv2.resize(item_img, (itemWidth,itemHeight), interpolation=cv2.INTER_LINEAR)
    b,g,r = cv2.split(matImg)
    img = cv2.merge((r,g,b))
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    #itemPictureFrame = tkinter.Frame(f, height=itemHeight, width=itemWidth)
    #itemPictureFrame.grid_propagate(0)
    #itemPictureFrame.grid(row=0, column=0, sticky=tkinter.E, pady=2, columnspan=2)

    #itemPicture = tkinter.Label(itemPictureFrame, text=" ", image=imgtk)
    
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
    # "The image features a large, illuminated scoreboard with a clock on it. The clock is positioned towards the top right corner of the scoreboard, and it is displaying the time as 11:45. The scoreboard is located in a room, possibly a gym or a sports facility, as there are several sports balls scattered around the area. Some of the balls are placed close to the scoreboard, while others are located further away. The presence of the sports balls suggests that the room is used for various sports activities and events.")
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

def do_weaviate_resnet50_on_roi(roi):
    ret, jpgImg = cv2.imencode('.jpg', roi)
    img_str = base64.b64encode(jpgImg).decode()
    sourceImage = { "image": img_str }
    weaviate_results = client.query.get(
            "Bottle", ["filepath", "brand"]
            ).with_near_image(
                    sourceImage, encode=False
            ).with_limit(1).do()
    #print("weaviate_result is: ")
    #print(weaviate_results)
    for res in weaviate_results:
        if len(weaviate_results["data"]["Get"]["Bottle"]) > 0:
            return weaviate_results["data"]["Get"]["Bottle"][0]["brand"]
    return "unknown"

def postprocess(
    pred_boxes: np.ndarray,
    input_hw: Tuple[int, int],
    orig_img: np.ndarray,
    min_conf_threshold: float = 0.5,
    nms_iou_threshold: float = 0.7,
    agnosting_nms: bool = False,
    max_detections: int = 30,
):
    """
    YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
    Parameters:
        pred_boxes (np.ndarray): model output prediction boxes
        input_hw (np.ndarray): preprocessed image
        orig_image (np.ndarray): image before preprocessing
        min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
        nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
        max_detections (int, *optional*, 300):  maximum detections after NMS
    Returns:
       pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label]
    """
    nms_kwargs = {"agnostic": agnosting_nms, "max_det": max_detections, "max_nms": 100 }
    preds = ops.non_max_suppression(torch.from_numpy(pred_boxes), min_conf_threshold, nms_iou_threshold, nc=80, **nms_kwargs)

    boxes = []
    results = []
    results.append({"boxes": boxes})
#    return results

    for i, pred in enumerate(preds):
        shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
        if not len(pred):
            continue
        pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
        #print("pred: ", pred)
        xyxy = [ pred[0][1], pred[0][1], pred[0][0]+pred[0][2], pred[0][1]+pred[0][3] ]
        box = Box(xyxy, pred[0][4], pred[0][5])
        #print("------>Box: ", box.xyxy, " ", box.cls, " " , box.conf)
        boxes.append(box)
    #print("Num of boxes: ", len(boxes))
    return results

def do_bit_on_roi(cls_model, roi, cls_imgsz, output_layer):
    resized_roi = cv2.resize(roi, (64,64), interpolation=cv2.INTER_LINEAR) 
    resized_roi = tf.reshape(resized_roi, [1,64,64,3])
    result_infer = cls_model([resized_roi])[output_layer]
    #print(result_infer)
    #print("Done bit infer!")
    #result_index = np.argmax(result_infer)

def do_yolo_on_roi2(det_model, roi, det_imgsz, output_layer):
    resized_roi = cv2.resize(roi, det_imgsz, interpolation=cv2.INTER_LINEAR)
    resized_roi = resized_roi.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    resized_roi = resized_roi.reshape((1,3, det_imgsz[0], det_imgsz[1]))

    res = det_model(resized_roi)[output_layer]
    print(roi.shape, " " , resized_roi.shape[:2])
    quit()
    detections = postprocess(res, resized_roi.shape[:2], roi)
    return detections


def do_yolo_on_frame(det_model, frame, det_imgsz, output_layer):    
    # Input N C H W
    #print(det_model.input(0))
    # Resize with W H
    resized_roi = cv2.resize(frame, (det_imgsz[0], det_imgsz[1]), interpolation=cv2.INTER_LINEAR)
    resized_roi = cv2.dnn.blobFromImage(resized_roi, 1/255, (det_imgsz[1],det_imgsz[0]),[0,0,0],1, crop=False)
    #resized_roi = tf.reshape(resized_roi, [1,3, det_imgsz[0], det_imgsz[1]])
    #print(resized_roi.shape)
    #input_tensor = np.expand_dims(resized_roi, 0)
    #print(input_tensor) 

    #resized_roi = tf.reshape(resized_roi, [1,3,det_imgsz[0], det_imgsz[1]])
    #print(output_layer)
    #print(det_imgsz)
    #print(resized_roi.shape)
    res = det_model(resized_roi)[output_layer]
    res = cv2.transpose(res[0])
    # output layer is 1,84,5400 so needs NMS.
    #print(output_layer)    
    #print(res)
    
    #eta = 0.5
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
            #print(box, " " , classIdIdx)
            #quit()

    # TODO: ScaleX and scaleY if H,W are not equal
    #print(frame.shape, " " , det_imgsz)
    scaleX = frame.shape[1] / det_imgsz[1] 
    scaleY = frame.shape[0] / det_imgsz[0]

    #print(frame.shape[1], " ", det_imgsz[0], " " , frame.shape[0], " ", det_imgsz[1])
    #print(scaleX, " " , scaleY)

    #print(len(boxes))
    boxIds = cv2.dnn.NMSBoxes(boxes, scores, confThreshold, nmsThreshold) #, eta) 
    #print("Post NMS")
    #print(len(boxIds))

    detections = []
    results = []
    for boxId in boxIds:
        xyxy = [ round(boxes[boxId][0]*scaleX), round(boxes[boxId][1]*scaleY), round((boxes[boxId][2]+boxes[boxId][0])*scaleX), round((boxes[boxId][3]+boxes[boxId][1])*scaleY) ]
        box = Box(xyxy, scores[boxId], classIds[boxId])
        #print("------>Box: ", box.xyxy, " ", box.cls, " " , box.conf)
        #detections.append(box)
        results.append({"boxes": [box]})

    #print(roi.shape, " " , det_imgsz)
    #detections = postprocess(res, det_imgsz, roi)
    #detections = [{"boxes": []}]
    #print(res)
    #print("---")
    #print(torch.from_numpy(res[0]))
    #print("***", detections)
    #results.append({"boxes": detections})
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


WEAVIATE_URL = os.getenv('WEAVIATE_URL')
if not WEAVIATE_URL:
    WEAVIATE_URL = 'http://127.0.0.1:8000'

client = weaviate.Client(WEAVIATE_URL)

window = tkinter.Tk()

# Main Window Rendering
windowHeight=480
windowWidth=640
startingRowIdxShoppingCart=1

window.title("Vision Self-Checkout Demo")
window.bind('<Escape>', close_window)
window.geometry(str(windowWidth) + 'x' + str(windowHeight))
#window.maxsize(width=windowWidth, height=windowHeight)

# Tab Widget
tabWidget = ttk.Notebook(window)
tabCameraView = tkinter.Frame(tabWidget)
#tabGenAIView = tkinter.Frame(tabWidget, width=windowWidth, height=windowHeight)
#tabGenAIView.pack()
#tabGenAIView.pack_propagate(0)

tabWidget.add(tabCameraView, text='Main')
#tabWidget.add(tabGenAIView, text='GenAI-Details')
tabWidget.pack(expand=1, fill="both")

# Main View
cartFrameWidth = 200
cartFrameHeight = windowHeight - 100
videoFrameWidth = windowWidth - cartFrameWidth
videoFrameHeight = cartFrameHeight


videoFrame = tkinter.Frame(tabCameraView, bg="black", width=videoFrameWidth, height=videoFrameHeight)
cartFrame = tkinter.Frame(tabCameraView, width=cartFrameWidth, height=cartFrameHeight)
shoppingClearButton = tkinter.Button(tabCameraView, text="Clear Cart", command=clear_shopping_cart)

videoFrame.grid_propagate(False)
cartFrame.grid_propagate(0)

# Add later if time permits. Need to refactor with a canvas for this...
#cartScrollbar = ttk.Scrollbar(tabCameraView, orient="vertical")
#cartScrollbar.pack(fill=tkinter.Y, side=tkinter.RIGHT, expand=False)
shopping_cart_ui = cartFrame
#gen_ai_ui = tabGenAIView


videoFrame.grid(         row=0, column=0, sticky=tkinter.NW, pady=10, rowspan=2)
cartFrame.grid(          row=0, column=1, sticky=tkinter.NE, pady=10, padx=10, rowspan=1)
shoppingClearButton.grid(row=1, column=1, sticky=tkinter.NW, padx=10, pady=5, rowspan=1)

shoppingCartLabel = tkinter.Label(cartFrame, text="               Shopping Cart")
shoppingCartLabel.grid(row=0, column=0, sticky=tkinter.NW, columnspan=2, pady=10, padx=0)

# Video playback / inference
## Source model defaults to 384x640
model = YOLO(model_name)
model_names = model.names
#cls_model = YOLO(cls_model_name)

# Set OpenVINO model to default to 384x640 as well
#from openvino.runtime import Core
#ie = Core()
if use_openvino:
    pass
    # ultralytics CPU usage is over 1000% for this YOLO model!! Hihgly inefficient
    #path = model.export(format='openvino', imgsz=det_imgsz, half=True)
    #model = YOLO(path)
    #path = cls_model.export(format='openvino', imgsz=cls_imgsz)
    #cls_model = YOLO(path)

    #ov_model = ie.read_model("/savedir/yolov8n-int8.xml")
    #ov_config = {}
    #ov_model.reshape({0: [1,3,det_imgsz[0], det_imgsz[1]]})
    # Use if GPU or AUTO AND GPU AVAILABLE
    #ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    #model = ie.compile_model(ov_model, "NPU", ov_config)
    #output_layer = model.output(0)

# use Bit Model instead
#cls_model = ie.read_model(model="BiT_M_R50x1_10C_50e_IR/1/FP32/64_64_3/model_64_64_3.xml")
#cls_model.reshape({0:[1,64,64,3]})

#compiled_model = ie.compile_model(model=cls_model, device_name="GPU")
#output_layer = compiled_model.output(0)

frame_count = 0
skip_frame_reclassify = False
caps = []

for s in source:
    cap = cv2.VideoCapture(s) #, cv2.CAP_GSTREAMER)
    caps.append(cap)


#if "/dev/video" in source:
#    print("Requesting 1280x720 camera resolution")
#    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#ctx = rs.context()
#devices = ctx.query_devices()

#pipe = rs.pipeline()
#cfg = rs.config()
#serial_num2 = '130322272768' # d405
#serial_num3 = '130322273236' # 
#serial_num = '130322272045'

#cfg2 = rs.config()
#cfg3 = rs.config()
#pipe2 = rs.pipeline()
#pipe3 = rs.pipeline()
#cfg2.enable_device(serial_num2)
#cfg2.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 5)
#cfg2.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 5)
#profile2 = pipe2.start(cfg2)

#cfg3.enable_device(serial_num3)
#cfg3.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 5)
#cfg3.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 5)
#profile3 = pipe3.start(cfg3)

#cfg.enable_device(serial_num)
#cfg.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 5)
#cfg.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 5)
#profile = pipe.start(cfg)


# Skip 5 first frames to give auto-exposure time to adjust
#for t in range(5):
#    pipe.wait_for_frames()
#    pipe2.wait_for_frames()
#    pipe3.wait_for_frames()

print("capturing video(s) from: ", source)

print("Running ", model_name, " with OpenVINO" if use_openvino else "")
#sfilter = ["banana", "apple", "orange", "broccoli", "carrot", "bottle"]
sfilter = ["bottle"]
vidPicture = None
#while cap.isOpened():
#frameset = pipe.wait_for_frames()
#frameset2 = pipe2.wait_for_frames()
#frameset3 = pipe3.wait_for_frames()
#color_frame = frameset.get_color_frame()
#depth_frame = frameset.get_depth_frame()
#color_frame2 = frameset2.get_color_frame()
#color_frame3 = frameset3.get_color_frame()

numOfCaps = len(caps)
capIdx = 0
#for cap in caps:
while True:
    cap = caps[capIdx]
    success, frame = cap.read()
    if not cap.isOpened() or not success:
        print('video failed...')
        break


    annotated_frame = None
    #window.update_idletasks()
    #window.update()

    if capIdx == 0:
        start_time = time.time()

    #if not success:
    #    print("Could not read any frames. Quitting.")
    #    break

    # Convert images to numpy arrays
    #depth_image = np.asanyarray(depth_frame.get_data())
    #frame = np.asanyarray(color_frame.get_data())
    #frame2 = np.asanyarray(color_frame2.get_data())
    #frame3 = np.asanyarray(color_frame3.get_data())


    # Add any GenAI results and render them in GenAI widget(s)
    while not tracked_mqtt_results.empty():
        trackid, res = tracked_mqtt_results.pop()
        #resArr = res.split("|")
        #print(trackid, res[0], res[1])
        #updateGenAIResult(tabGenAIView, trackid, res, True)


    results = model(frame, imgsz=det_imgsz, verbose=False)  #model.track(frame, tracker=tracker, imgsz=det_imgsz, persist=True, verbose=True)
    #result2 = model(frame2, imgsz=det_imgsz, verbose=False)
    #result3 = model(frame3, imgsz=det_imgsz, verbose=False)
    #results = do_yolo_on_frame(model, frame.copy(), det_imgsz, output_layer)    
    send_lvlm_processing = True

    #print("process results......", videoFrameWidth, " ", videoFrameHeight)
    if show_gui:
        annotated_frame = frame.copy()
    for result in results:
        boxes = result.boxes.cpu() # ultralytics
        #boxes = result["boxes"]
        track_ids = result.boxes.id.int().cpu().tolist() if not result.boxes.id is None else []
        result_label = ""
        result_label_cls = ""

        for box in boxes:
            result_label = model_names[int(box.cls)]
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

            roi = save_one_box(torch.tensor(box.xyxy), origFrame,BGR=True, save=False)
            result_label = model_names[c]
            #print("Detected: ", result_label)
            cls_label = do_weaviate_resnet50_on_roi(roi)
            if show_gui:
                do_update_annotated_frame_detection(annotated_frame, det_imgsz, box.xyxy[0].tolist(), result_label + ": " + cls_label)
                #print(box.xyxy, " ", result_label, " ", cls_label)
                #annotated_frame = result.plot()
            result_label = cls_label

            #print("Feature classification: ", result_label)
           
            #result.save_crop("/savedir", str(track_id) + ".jpg")

            #tracked_object = tracked_objects.get(track_id)
            #tracked_object_time = tracked_objects_time.get(track_id)
            #tracked_object_state = tracked_objects_state.get(track_id)

            #if not skip_frame_reclassify or not tracked_object:
            #if not tracked_object:
            #    result_label = model_names[c] 
            #else:
            #    result_label = tracked_object

            #if not tracked_object:
            #    # for cls
            #    # comment out below and 
            #    #do_weaviate_resnet50_on_roi(roi, cls_imgsz)
            #    # result_label = do_classification_on_roi(cls_model, roi, cls_imgsz)
            #    #result_label = do_bit_on_roi(compiled_model, roi, cls_imgsz,output_layer)
            #    result_label = do_weaviate_resnet50_on_roi(roi)
            #    #result_label2 = do_bit_on_roi(compiled_model, roi, cls_imgsz, output_layer)
            #    #result_label3 = do_bit_on_roi(compiled_model, roi, cls_imgsz, output_layer)

            #    objTime = time.time()
            #    tracked_objects.put(track_id, result_label)
            #    tracked_objects_time.put(track_id, objTime)
            #    tracked_objects_state.put(track_id, 0)

            #    #print("Caching: ", tracked_objects.get(track_id))

            #    #print("New item not tracked needs LVLM trackid: ", track_id, ", item: ", result_label, " at ", objTime)
            #elif not tracked_object is None:
            #    # this is a debounce routine
            #    # items is tracked but has it been tracking long enough for hand to be out of picture
            #    if ((time.time() - tracked_object_time)*1000) > 100 and tracked_object_state == 0:
            #        item_label = result_label + ": #" + str(track_id)
            #        ret, jpgImg = cv2.imencode('.jpg', roi)
            #        startingRowIdxShoppingCart = addShoppingCartItem(cartFrame, startingRowIdxShoppingCart,
            #                                                         item_label, roi)
            #        #publish(client, jpgImg.tobytes(), str(track_id), "describe the item in the image")
            #        #addGenAIResult(tabGenAIView, roi, track_id, "describe the item in the image", jpgImg)
            #        tracked_objects_state.put(track_id, 1)

            ## only need if using cls
            ##if show_gui:
            ##    do_update_annotated_frame(annotated_frame, box.xywh, result_label)A

    if capIdx == numOfCaps -1:
        capIdx = 0        
        frame_count = frame_count + 1
    else:
        capIdx = capIdx + 1

#    print(capIdx, " " , numOfCaps)
    # Skip reclassification based on tracked objects and interval specified
    skip_frame_reclassify = frame_count % reclassify_interval != 0

    elapsed_time = time.time() - start_time
    if frame_count % 5 == 0:
        print("Seconds taken for pipeline: ", elapsed_time)

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

    # Break the loop if 'q' is pressed
    #if cv2.waitKey(1) & 0xFF == ord("q"):
    #    break
    try:
        pass
        #success, frame = cap.read()

        #frameset = pipe.wait_for_frames()
        #frameset2 = pipe2.wait_for_frames()
        #frameset3 = pipe3.wait_for_frames()
        #color_frame = frameset.get_color_frame()
        #depth_frame = frameset.get_depth_frame()

        #color_frame2 = frameset2.get_color_frame()
        #color_frame3 = frameset3.get_color_frame()
    except:
        print("camera error...restart needed.")
        #for dev in devices:
        #    dev.hardware_reset()
        #    time.sleep(1)


################

window.mainloop()
