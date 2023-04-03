from jina import Flow

f = Flow(port=8080, extra_search_paths=["detector"]).add(
    name="DetectorYoloV8",
    uses="detector.yml",
    uses_with={
        "weights": "./detector/executor/weights/yolov8s.onnx",
    },
)

with f:
    f.block()
