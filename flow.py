from jina import Flow

f = (
    Flow(port=8080, extra_search_paths=["detector"])
    .config_gateway(replicas=3)
    .add(
        name="DetectorYoloV8",
        uses="detector.yml",
        uses_with={
            "weights": "./detector/executor/weights/yolov8s.onnx",
            "input_size": (640, 640, 3),
        },
        replicas=1,
    )
)

with f:
    f.block()
