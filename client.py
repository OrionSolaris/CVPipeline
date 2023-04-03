from jina import Client, Document, DocumentArray
from rich import print
import cv2

from utils.torch_sync import time_synchronized

c = Client(host="grpc://0.0.0.0:8080")

cap = cv2.VideoCapture("nyc_walking2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
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
                tags={"identity": "cam1", "classes": [0]},
            )
        ]
    )
    resp = c.post("/", doc, request_size=1)
    t2 = time_synchronized()
    cv2.imshow("pred", resp[0].tensor)
    t3 = time_synchronized()
    print(
        f"[green]INFO[/green]: Done! ([blue][bold]{round(1E3 * (t1 - t0),1)}ms[/bold][/blue]) Read Frame, ([blue][bold]{round(1E3 * (t2 - t1),1)}ms[/bold][/blue]) Send/Recieve, ([blue][bold]{round(1E3 * (t3 - t2),1)}ms[/bold][/blue]) Show Frame, ([blue]{round(1000/round(sum([round(1E3 * (t1 - t0),1),round(1E3 * (t2 - t1),1),round(1E3 * (t3 - t2),1)])),2)}[/blue]) Est. FPS"
    )
    if cv2.waitKey(1) == ord("q"):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
