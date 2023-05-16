import cv2
from ultralytics import YOLO
import cvzone
import math

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("./videos/license plate.mp4")
# cap.set(3, 1280)
# cap.set(4, 720)

model = YOLO("./Yolo-Weights/best.pt")

classNames = ['licence']

while True:
    success, img = cap.read()

    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), thickness=3)
            # x1,y1,width,height= box.xywh
            w, h = x2-x1, y2-y1

            # confident
            conf = math.ceil((box.conf[0]*100))/100
            # print(conf)

            # class name
            cls = int(box.cls[0])
            # print(cls)

            if conf >= 0.50:
                cvzone.cornerRect(img, (x1, y1, w, h), l=5)
                cvzone.putTextRect(
                    img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), thickness=1, scale=1)
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
