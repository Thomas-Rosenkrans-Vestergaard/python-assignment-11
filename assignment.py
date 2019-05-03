import cv2
import time
import numpy as np

def get_video_input():
    cap = cv2.VideoCapture(0)
    time.sleep(0.25)

    w, h = (640, 400)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    return cap

def run(operation=None):
    cap = get_video_input()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if operation is not None:
            frame = operation(frame)

        cv2.imshow("Face detection", frame)
        
        if cv2.waitKey(5) == 27:
            break

    cv2.destroyAllWindows()

def detection(frame):
    faces = detect_faces(frame)
    mark_rect(frame, faces)
    return frame   

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier('facedetection.xml')
    faces_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(faces_gray)

def mark_rect(frame, rects):
    for (x,y,w,h) in rects:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

if __name__ == "__main__":
    run(detection)