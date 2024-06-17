import sys
import subprocess
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--tracker", nargs="?", default="botsort.yaml", help="botsort.yaml (Default) or bytetrack.yaml")
#parser.add_argument("--source", nargs="?", default="sample.mp4")
parser.add_argument("--source", "-s", action='append', help='USB or RTSP or FILE sources', required=True)
parser.add_argument("--model", nargs="?", default="yolov8n.pt")
parser.add_argument("--cls_model", nargs="?", default="yolov8n-cls.pt")
parser.add_argument("--enable_cls_preprocessing", default=False, action="store_true")
parser.add_argument("--use_openvino", default=True, action="store_true")
parser.add_argument("--reclassify_interval", nargs="?", type=int, default=1)
parser.add_argument("--max_tracked_objects", nargs="?", type=int, default=20)
parser.add_argument("--show", default=False, action="store_true")
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

workers = []
for s in source:
    argslist = [ "vision24-demo.py",  "--source",  str(s) , "--show"]
    worker = subprocess.Popen(["python3"] + argslist, stdout=None)
    workers.append(worker)

rc = 0
try:
    q_len = len(workers)
    while q_len > 0:
        for w in workers:
            try:
                w_rt = w.wait(1)
            except subprocess.TimeoutExpired:
                continue
            q_len -= 1
            if w_rt != 0:
                if rc != 1:
                    for w in workers:
                        w.terminate()
                rc = 1
except KeyboardInterrupt:
    print("quitting...")
    for w in workers:
        w.terminate()
    for w in workers:
        w.wait()
    raise

sys.exit(rc)
