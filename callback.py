import cv2
import time
import math
import av


n_frames = 0
fps_cum = 0.0
fps_avg = 0.0

def callback(frame,model):
    global n_frames,fps_cum,fps_avg
    start_time = time.perf_counter()
    n_frames+=1
    frame=frame.to_ndarray(format="bgr24")
    res=model.predict(frame,device="cpu",classes=[0],conf=0.70,verbose=False,iou=0.7,imgsz=(240,320))
    boxes=res[0].boxes
    lpc=0
    for xyxy in boxes.xyxy:
        lpc+=1
        cv2.rectangle(frame,(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),(0,255,0))
    end_time = time.perf_counter()
    fps = 1.0 / (end_time - start_time)
    fps_cum += fps
    fps_avg = fps_cum / n_frames
    cv2.putText(frame,f"Live Person Count : {lpc}",(10,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,color=(0,0,0),fontScale=1)
    cv2.putText(frame,'FPS: {}'.format(math.ceil(fps_avg)),(10,90),cv2.FONT_HERSHEY_COMPLEX_SMALL,color=(0,0,0),fontScale=1)
    return av.VideoFrame.from_ndarray(frame,format="bgr24")