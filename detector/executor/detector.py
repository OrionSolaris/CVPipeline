from jina import Executor, requests, DocumentArray
from rich import print

from ultralytics import YOLO
from ultralytics.yolo.utils import ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_yaml
import torch


class Detector(Executor):
    def __init__(
        self,
        weights: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = YOLO(weights, task="detect")
        self.classes = yaml_load(check_yaml("coco128.yaml"))["names"]
        self.device = "0" if torch.cuda.is_available() else "cpu"

    @requests
    def detect(self, docs: DocumentArray, **kwargs):
        docs_to_return = DocumentArray()
        for doc in docs:
            results = self.model.predict(
                doc.tensor, classes=doc.tags.get("classes"), device=self.device
            )
            doc.tensor = results[0].plot(line_width=2)
            docs_to_return.append(doc)
        return docs_to_return
