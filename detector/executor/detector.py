from jina import Executor, requests, DocumentArray
from rich import print

from ultralytics import YOLO
from ultralytics.yolo.utils import ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_yaml
import torch
import numpy as np

from .helper.torch_sync import time_synchronized


class Detector(Executor):
    def __init__(
        self,
        weights: str,
        input_size: tuple,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = YOLO(weights, task="detect")
        self.classes = yaml_load(check_yaml("coco128.yaml"))["names"]
        self.device = "0" if torch.cuda.is_available() else "cpu"
        warmup = np.ones(input_size, dtype=np.float64)
        self.model.predict(
            warmup,
            classes=[0],
            device=self.device,
        )

    @requests
    def detect(self, docs: DocumentArray, **kwargs):
        docs_to_return = DocumentArray()
        for doc in docs:
            doc.tags["det_recieve"] = time_synchronized()
            results = self.model.predict(
                doc.tensor, classes=doc.tags.get("classes"), device=self.device
            )
            doc.tensor = results[0].plot(line_width=2)
            doc.tags["det_preproc"] = results[0].speed["preprocess"]
            doc.tags["det_inference"] = results[0].speed["inference"]
            doc.tags["det_postproc"] = results[0].speed["postprocess"]
            doc.tags["det_send"] = time_synchronized()
            docs_to_return.append(doc)
        return docs_to_return
