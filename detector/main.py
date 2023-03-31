from jina import Executor
import torch
import time
import random
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.datasets import letterbox
class Tracker(Executor):
    def __init__(
        self, device: str = "cpu", weights="models/yolov7-w6.pt", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = torch.load(weights)

model = attempt_load("./models/yolov7-w6.pt")
device = select_device("0")
half = device.type != 'cpu'
stride = int(model.stride.max()) 
imgsz = check_img_size(1280, s=stride)  # check img_size
if half:
        model.half()

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
print(stride,names,colors)

old_img_w = old_img_h = imgsz
old_img_b = 1


img0 = cv2.imread('horses.jpeg')
img = letterbox(img0, (1280,1280), stride = stride)[0]
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
img = np.ascontiguousarray(img)
t0 = time.time()
img = torch.from_numpy(img).to(device)
img = img.half() if half else img.float()  # uint8 to fp16/32
img /= 255.0  # 0 - 255 to 0.0 - 1.0
if img.ndimension() == 3:
    img = img.unsqueeze(0)
if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=True)[0]
t1 = time_synchronized()
pred = model(img, augment=True)[0]
t2 = time_synchronized()
pred = non_max_suppression(pred, 0.25, 0.45, classes=[0,17], agnostic=True)
t3 = time_synchronized()
for i, det in enumerate(pred):
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)
                    
cv2.imshow("1", img0)
if cv2.waitKey(0) == ord('q'):  # q to quit
    cv2.destroyAllWindows()
    raise StopIteration