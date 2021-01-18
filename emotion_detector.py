from cv2 import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace


cap = cv2.VideoCapture(0)

face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:

    ret, frame = cap.read()
    res = DeepFace.analyze(frame, actions=["emotion"])

    face_rect = face_haar.detectMultiScale(frame)
    for x, y, w, h in face_rect:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    
    cv2.putText(frame, res["dominant_emotion"], (x, y-20), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()