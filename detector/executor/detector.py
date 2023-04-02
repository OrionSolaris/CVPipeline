from jina import Executor, requests, DocumentArray
from rich import print

from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes
from ultralytics.yolo.utils.ops import non_max_suppression
import cv2

frame = cv2.imread("../horses.jpeg")
model = YOLO("./weights/yolov8n")
results = model.predict(frame, classes=[0, 1])

# print(type(results[0].boxes))
# print(dir(results[0].boxes))

# for x in results[0].boxes:
#     print(x)
# print(results[0].boxes.boxes)
# boxes = non_max_suppression(results[0].boxes.boxes)
# print(model.overrides)
res_plotted = results[0].plot(line_width=1)
cv2.imshow("1", res_plotted)
if cv2.waitKey(0) == ord("q"):
    raise StopIteration
# class Detector(Executor):
#     def __init__(
#         self,
#         weights: str,
#         *args,
#         **kwargs,
#     ):
#         Executor.__init__(self, *args, **kwargs)

#         self.model  = YOLO(weights)

#     @requests
#     def detect(self, docs: DocumentArray, **kwargs):
#         docs_to_return = DocumentArray()
#         for doc in docs:
#             results = self.model(doc.tensor)
#             print(type(results[0].boxes))
#             for x in results[0].boxes:
#                 print(x)

#             res_plotted = results[0].plot(line_width=1)
