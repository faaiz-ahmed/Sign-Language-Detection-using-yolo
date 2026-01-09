from flask import Flask, Response
import cv2
import time
from ultralytics import YOLO

app=Flask(__name__)

model=YOLO('runs/detect/train/weights/best.pt')

cap=cv2.VideoCapture('http://172.18.9.163:8080/video')
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def generate_frames():
    while True:
        cap.grab()
        success,frame=cap.read()
        if not success:
            print("Camera read failed")
            continue

        frame=cv2.resize(frame,(416,320))

        results=model(frame,conf=0.3,imgsz=320)
        annotated_frame=results[0].plot()

        ret,buffer=cv2.imencode('.jpg',annotated_frame)
        if not ret:
            continue

        frame=buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

        time.sleep(0.03)

@app.route('/')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0',port=5000,threaded=True)