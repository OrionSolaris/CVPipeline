from jina import Client, Document, DocumentArray
import cv2
from detector.utils.torch_utils import time_synchronized
from rich import print

c = Client(host="grpc://0.0.0.0:8080")

# img = cv2.imread("horses.jpeg")
# doc = DocumentArray(
#     [
#         Document(
#             tensor=img, tags={"identity": "cam1", "augment": True, "classes": [0, 17]}
#         )
#     ]
# )
# resp = c.post("/", doc)
# for x in resp:
#     print(x.tags)

# cv2.imshow("pred", resp[0].tensor)
# if cv2.waitKey(0) == ord("q"):  # q to quit
#     cv2.destroyAllWindows()
#     raise StopIteration

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
                embedding=frame,
                tags={"identity": "cam1", "augment": True, "classes": [0]},
            )
        ]
    )
    resp = c.post("/", doc, request_size=1)
    t2 = time_synchronized()
    cv2.imshow("pred", resp[0].embedding)
    t3 = time_synchronized()
    print(
        f"[green]INFO[/green]: Done! ([blue]{round(1E3 * (t1 - t0),1)}ms[/blue]) Read Frame, ([blue]{round(1E3 * (t2 - t1),1)}ms[/blue]) Send/Recieve , ([blue]{round(1E3 * (t3 - t2),1)}ms[/blue]) Show Frame"
    )
    if cv2.waitKey(1) == ord("q"):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
