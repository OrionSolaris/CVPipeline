from jina import Flow

f = Flow(port=8080, extra_search_paths=["executors/detector"]).add(
    name="DetectorYoloV7",
    uses="detector.yml",
    uses_with={
        "weights": "executors/detector/executor/weights/yolov7.pt",
        "device": "0",
        "img_size": 640,
        "trace": True,
    },
)

with f:
    f.block()
