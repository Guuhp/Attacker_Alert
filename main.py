import cv2
from ultralytics import YOLO

video = cv2.VideoCapture('asset/video.mp4')
model = YOLO('yolov8n.pt')

while True:
    check, frame = video.read()
    frame = cv2.resize(frame,(1270,720))
    image = model(frame)
    
    for objects in image:
        obj = objects.boxes
        for data in obj:
            x,y,w,h = data.xyxy[0]
            x,y,w,h = int(x),int(y),int(w),int(h)
            if int(data.cls[0]) == 0:
                cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 5)


    cv2.imshow('video', frame)
    cv2.waitKey(1)