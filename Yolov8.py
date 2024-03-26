
from ultralytics import YOLO
import cv2
import time
import math



model=YOLO("yolov8n_openvino_model/",task="detect",verbose=False)

cap=cv2.VideoCapture(0)

n_frames = 0
fps_cum = 0.0
fps_avg = 0.0

while True:
    ret,frame=cap.read()
    if not ret:
        print("Error")
        break
    start_time = time.perf_counter()
    n_frames+=1

    res=model(frame,device="cpu",classes=[0],conf=0.75,verbose=False,agnostic_nms=True,iou=0.5)
    boxes=res[0].boxes
    lpc=0
    for xyxy in boxes.xyxy:
        lpc+=1
        cv2.rectangle(frame,(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),(0,255,0))

    end_time = time.perf_counter()
    fps = 1.0 / (end_time - start_time)
    fps_cum += fps
    fps_avg = fps_cum / n_frames
    cv2.putText(frame,f"Live Person Count : {lpc}",(10,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,color=(0,255,0),fontScale=1)
    cv2.putText(frame,'FPS: {}'.format(math.ceil(fps_avg)),(10,90),cv2.FONT_HERSHEY_COMPLEX_SMALL,color=(0,255,0),fontScale=1)

    cv2.imshow("Application",frame)

    key=cv2.waitKey(1)
    if key==ord('q'):
        break
cv2.destroyAllWindows()
    

# res=model.predict("bus.jpg",device="cpu",classes=[0],conf=0.75)
# boxes=res[0].boxes
# print(boxes.xyxy)
        