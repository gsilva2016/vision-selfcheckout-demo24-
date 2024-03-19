import os
import sys
import tkinter as tkinter
import cv2
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter.scrolledtext as tks

from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
from collections import OrderedDict
import argparse
import numpy as np
from queue import Queue

import time
from paho.mqtt import client as mqtt_client
broker = 'broker.hivemq.com'
broker = '198.175.88.142'
port = 1883
topic = "/roi/bottle-1/describe_the_item_in_the_image"
client_id = 'test-client/`'
ready = 0

shopping_cart_ui = None
gen_ai_ui = None

parser = argparse.ArgumentParser()
parser.add_argument("--tracker", nargs="?", default="botsort.yaml", help="botsort.yaml (Default) or bytetrack.yaml")
parser.add_argument("--source", nargs="?", default="sample.mp4")
parser.add_argument("--model", nargs="?", default="yolov8n.pt")
parser.add_argument("--cls_model", nargs="?", default="yolov8n-cls.pt")
parser.add_argument("--enable_cls_preprocessing", default=False, action="store_true")
parser.add_argument("--use_openvino", default=True, action="store_true")
parser.add_argument("--reclassify_interval", nargs="?", type=int, default=1)
parser.add_argument("--max_tracked_objects", nargs="?", type=int, default=20)
parser.add_argument("--show", default=True, action="store_true")
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

def do_update_annotated_frame(annotated_frame, xywh, label) -> None:
    x = int(xywh[0, 0])
    y = int(xywh[0, 1])
    w = int(xywh[0, 2])
    h = int(xywh[0, 3])

    cv2.putText(
        annotated_frame,
        label,
        (x+5, y),
        0,
        1, # SF
        (255, 255, 255),  # Text Color
        thickness=3,      # Text thickness
        lineType=cv2.LINE_AA
    )
    #return np.asarray(img)


# MQTT
client = connect_mqtt()
subscribe(client)
client.loop_start()
while ready != 1:
    time.sleep(1)
print("Connecting to broker ")


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
    found_ui = ui_frame.nametowidget(str(item_id)) if is_first else ui_frame
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



window = tkinter.Tk()

# Main Window Rendering
windowHeight=1000
windowWidth=1800
startingRowIdxShoppingCart=1

window.title("Vision Self-Checkout Demo")
window.bind('<Escape>', close_window)
window.geometry(str(windowWidth) + 'x' + str(windowHeight))
#window.maxsize(width=windowWidth, height=windowHeight)

# Tab Widget
tabWidget = ttk.Notebook(window)
tabCameraView = tkinter.Frame(tabWidget)
tabGenAIView = tkinter.Frame(tabWidget, width=windowWidth, height=windowHeight)
tabGenAIView.pack()
tabGenAIView.pack_propagate(0)

tabWidget.add(tabCameraView, text='Main')
tabWidget.add(tabGenAIView, text='GenAI-Details')
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
gen_ai_ui = tabGenAIView


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
if use_openvino:
    path = model.export(format='openvino', imgsz=det_imgsz, half=True)
    model = YOLO(path)
    #path = cls_model.export(format='openvino', imgsz=cls_imgsz)
    #cls_model = YOLO(path)


frame_count = 0
skip_frame_reclassify = False
cap = cv2.VideoCapture(source)
if "/dev/video" in source:
    print("Requesting 1280x720 camera resolution")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("capturing video from: ", source)

print("Running ", model_name, " with OpenVINO" if use_openvino else "")
sfilter = ["banana", "apple", "orange", "broccoli", "carrot", "bottle"]
vidPicture = None
while cap.isOpened():
    annotated_frame = None
    #window.update_idletasks()
    #window.update()

    success, frame = cap.read()

    if not success:
        print("Could not read any frames. Quitting.")
        break

    # Add any GenAI results and render them in GenAI widget(s)
    while not tracked_mqtt_results.empty():
        trackid, res = tracked_mqtt_results.pop()
        #resArr = res.split("|")
        #print(trackid, res[0], res[1])
        updateGenAIResult(tabGenAIView, trackid, res, True)


    results = model.track(frame, tracker=tracker, imgsz=det_imgsz, persist=True, verbose=False)
    send_lvlm_processing = True

    for result in results:
        boxes = result.boxes.cpu()
        track_ids = result.boxes.id.int().cpu().tolist() if not result.boxes.id is None else []
        result_label = ""

        for box in boxes:
            result_label = model_names[int(box.cls)]
            if not any(s in result_label for s in sfilter):
                send_lvlm_processing = False
                break
        
        if not send_lvlm_processing:
            continue

        if show_gui:
            annotated_frame = result.plot()

        #print("No hand in pic-->", result_label, "person" in result_label)



        # Add any GenAI results and render them in GenAI widget(s)
        #while not tracked_mqtt_results.empty():
        #    trackid, res = tracked_mqtt_results.pop()
        #    #resArr = res.split("|")
        #    print(trackid, res[0], res[1])
        #    updateGenAIResult(tabGenAIView, trackid, res, True)


        for box, track_id in zip(boxes, track_ids):
            c, conf, id = int(box.cls), float(box.conf), None if box.id is None else int(box.id.item())

            roi = save_one_box(box.xyxy, result.orig_img.copy(),BGR=True, save=False)
            #result.save_crop("/savedir", str(track_id) + ".jpg")

            tracked_object = tracked_objects.get(track_id)
            tracked_object_time = tracked_objects_time.get(track_id)
            tracked_object_state = tracked_objects_state.get(track_id)

            #if not skip_frame_reclassify or not tracked_object:

            result_label = model_names[c]

            if not tracked_object:
                # for cls
                # result_label = do_classification_on_roi(cls_model, roi, cls_imgsz)

                objTime = time.time()
                tracked_objects.put(track_id, result_label)
                tracked_objects_time.put(track_id, objTime)
                tracked_objects_state.put(track_id, 0)

                #print("New item not tracked needs LVLM trackid: ", track_id, ", item: ", result_label, " at ", objTime)
            elif not tracked_object is None:
                # this is a debounce routine
                # items is tracked but has it been tracking long enough for hand to be out of picture
                if ((time.time() - tracked_object_time)*1000) > 100 and tracked_object_state == 0:
                    item_label = result_label + ": #" + str(track_id)
                    ret, jpgImg = cv2.imencode('.jpg', roi)
                    startingRowIdxShoppingCart = addShoppingCartItem(cartFrame, startingRowIdxShoppingCart,
                                                                     item_label, roi)
                    publish(client, jpgImg.tobytes(), str(track_id), "describe the item in the image")
                    addGenAIResult(tabGenAIView, roi, track_id, "describe the item in the image", jpgImg)
                    tracked_objects_state.put(track_id, 1)

            # only need if using cls
            #if show_gui:
            #    do_update_annotated_frame(annotated_frame, box.xywh, result_label)

    frame_count = frame_count + 1

    # Skip reclassification based on tracked objects and interval specified
    skip_frame_reclassify = frame_count % reclassify_interval != 0

    # Display the annotated frame
    if show_gui:
        if annotated_frame is None:
            annotated_frame = frame

        #cv2.imshow("YOLOv8 " + tracker + " Tracking" + " with OpenVINO" if use_openvino else "", annotated_frame)

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


################

window.mainloop()
