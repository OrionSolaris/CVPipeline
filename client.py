from jina import Client, Document, DocumentArray
from rich import print
import cv2
import numpy as np

from utils.torch_sync import time_synchronized
from utils.metrics_timer import Timer

c = Client(host="grpc://0.0.0.0:8080")

cap = cv2.VideoCapture("./test_examples/nyc_walking1080.mp4")
cap_fps = cap.get(cv2.CAP_PROP_FPS)
cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

print(
    f"""[green]INFO[/green]: Video of [blue][bold]{int(cap_width)}[/bold][/blue][grey78][not bold]x[/not bold][/grey78][blue][bold]{int(cap_height)}[/bold][/blue] @ [blue]{round(cap_fps,1)}[/blue] FPS"""
)

section_metrics = Timer()

while True:
    # Capture frame-by-frame
    t0 = time_synchronized()
    ret, frame = cap.read()
    t1 = time_synchronized()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # Display the resulting frame
    doc = DocumentArray(
        [
            Document(
                tensor=frame,
                tags={
                    "identity": "cam1",
                    "classes": [0],
                },  # pick the classes you want here @ classes.yml
            )
        ]
    )
    before_post = time_synchronized()

    resp = c.post("/", doc, request_size=1)

    t2 = time_synchronized()

    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time_synchronized()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(
        resp[0].tensor,
        str(round(fps, 1)),
        (0, 30),
        font,
        1,
        (100, 255, 0),
        1,
        cv2.LINE_AA,
        False,
    )
    cv2.imshow("pred", resp[0].tensor)

    t3 = time_synchronized()
    print(
        f"[green]INFO[/green]: Done! ([blue][bold]{round(1E3 * (t1 - t0),1)}ms[/bold][/blue]) Read Frame, ([blue][bold]{round(1E3 * (t2 - t1),1)}ms[/bold][/blue]) Send/Receive, ([blue][bold]{round(1E3 * (t3 - t2),1)}ms[/bold][/blue]) Show Frame, ([blue]{fps:.1f}[/blue]) Est. FPS"
    )

    section_metrics.read_frames["values"] = np.append(
        section_metrics.read_frames["values"], [1e3 * (t1 - t0)]
    )
    section_metrics.send_recieve["values"] = np.append(
        section_metrics.send_recieve["values"], [1e3 * (t2 - t1)]
    )
    section_metrics.show_frames["values"] = np.append(
        section_metrics.show_frames["values"], [1e3 * (t3 - t2)]
    )

    section_metrics.det_preproc["values"] = np.append(
        section_metrics.det_preproc["values"], [resp[0].tags.get("det_preproc")]
    )
    section_metrics.det_infer["values"] = np.append(
        section_metrics.det_infer["values"], [resp[0].tags.get("det_inference")]
    )
    section_metrics.det_postproc["values"] = np.append(
        section_metrics.det_postproc["values"], [resp[0].tags.get("det_postproc")]
    )

    avg_serialisation_cost = round(
        np.average(
            [
                1e3 * (resp[0].tags.get("det_recieve") - before_post),
                1e3 * (t2 - resp[0].tags.get("det_send")),
            ]
        ),
        1,
    )
    total_serialisation_cost = round(
        np.sum(
            [
                1e3 * (resp[0].tags.get("det_recieve") - before_post),
                1e3 * (t2 - resp[0].tags.get("det_send")),
            ]
        ),
        1,
    )

    section_metrics.serialise["values"] = np.append(
        section_metrics.serialise["values"], [avg_serialisation_cost]
    )
    section_metrics.total_serialise["values"] = np.append(
        section_metrics.total_serialise["values"], [total_serialisation_cost]
    )
    if cv2.waitKey(1) == ord("q"):
        break
section_metrics.print_avgs()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
