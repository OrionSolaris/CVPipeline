from jina import Flow
from detector.main import Detector

f = Flow(port=8080).add(
    name="DetectorYoloV7",
    uses=Detector,
    uses_with={
        "weights": "./models/yolov7.pt",
        "device": "0",
        "img_size": 640,
        "trace": True,
    },
)

with f:
    f.block()
