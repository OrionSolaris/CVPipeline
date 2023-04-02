from jina import Executor, requests, DocumentArray, Flow
from rich import print
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
from models.experimental import attempt_load
from utils.get_classes import get_classes
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel
from utils.datasets import letterbox


class Detector(Executor):
    def __init__(
        self,
        img_size: int = 640,
        device: str = "cpu",
        weights: str = "./models/yolov7.pt",
        trace: bool = False,
        colors: list = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.device = select_device(str(device))
        self.half = self.device.type != "cpu"
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)
        if trace:
            self.model = TracedModel(self.model, self.device, img_size)
        if self.half:
            self.model = self.model.half()
        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, self.img_size, self.img_size)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )
        self.old_img_w = self.old_img_h = self.img_size
        self.old_img_b = 1

        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )
        print(
            f"\n[green]INFO[/green]: {len(self.names)} total classes for detector: {self.names}"
        )
        self.colors = (
            colors
            if colors
            else [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        )
        cudnn.benchmark = True

    @requests
    def detect(self, docs: DocumentArray, **kwargs):
        with torch.no_grad():
            # docs_to_return = DocumentArray()
            # docs.map()
            processed_img = letterbox(
                docs[0].embedding, self.img_size, stride=self.stride
            )[0]
            processed_img = processed_img[:, :, ::-1].transpose(
                2, 0, 1
            )  # BGR to RGB, to 3x416x416
            processed_img = np.ascontiguousarray(processed_img)
            processed_img = torch.from_numpy(processed_img).to(self.device)
            processed_img = processed_img.half() if self.half else processed_img.float()
            processed_img /= 255.0
            if processed_img.ndimension() == 3:
                processed_img = processed_img.unsqueeze(0)
            if self.device.type != "cpu" and (
                self.old_img_b != processed_img.shape[0]
                or self.old_img_h != processed_img.shape[2]
                or self.old_img_w != processed_img.shape[3]
            ):
                self.old_img_b = processed_img.shape[0]
                self.old_img_h = processed_img.shape[2]
                self.old_img_w = processed_img.shape[3]
                for i in range(3):
                    self.model(
                        processed_img,
                        augment=docs[0].tags["augment"]
                        if "augment" in docs[0].tags
                        else True,
                    )[0]
            t1 = time_synchronized()
            with torch.no_grad():
                pred = self.model(
                    processed_img,
                    augment=docs[0].tags["augment"]
                    if "augment" in docs[0].tags
                    else True,
                )[0]
            t2 = time_synchronized()
            pred = non_max_suppression(
                pred,
                conf_thres=docs[0].tags["confidence_threshold"]
                if "confidence_threshold" in docs[0].tags
                else 0.25,
                iou_thres=docs[0].tags["iou_threshold"]
                if "iou_threshold" in docs[0].tags
                else 0.45,
                classes=get_classes(docs[0].tags.get("classes"), self.names),
                agnostic=docs[0].tags["agnostic_nms"]
                if "agnostic_nms" in docs[0].tags
                else True,
            )
            t3 = time_synchronized()
            for i, det in enumerate(pred):
                gn = torch.tensor(docs[0].embedding.shape)[
                    [1, 0, 1, 0]
                ]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        processed_img.shape[2:], det[:, :4], docs[0].embedding.shape
                    ).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    # Write results
                    for *xyxy, conf, cls in reversed(det):

                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        plot_one_box(
                            xyxy,
                            docs[0].embedding,
                            label=label,
                            color=self.colors[int(cls)],
                            line_thickness=1,
                        )
            print(
                f"[green]INFO[/green]: Done! ([blue]{round(1E3 * (t2 - t1),1)}ms[/blue]) Inference, ([blue]{round(1E3 * (t3 - t2),1)}ms[/blue]) NMS"
            )
            #     docs_to_return.append(doc)
            # return docs_to_return