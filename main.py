import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from deepface import DeepFace

face_classifier = cv2.CascadeClassifier()
face_classifier.load(cv2.samples.findFile("haarcascade_frontalface_default.xml"))

cap = cv2.VideoCapture("testvideo.mp4")

while True:
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame_gray)
    response = DeepFace.analyze(frame,actions=("emotion"),enforce_detection=False)
    print(response)
    for face in faces:
        x,y,w,h = face
        cv2.putText(frame,text=response[0]["dominant_emotion"],org=(x,y),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(255,0,0),thickness=2)
        new_frame = cv2.rectangle(frame,(x,y),(x+w,y+h),color=(255,0,0),thickness=2)
        cv2.imshow(" ",new_frame)
    if(cv2.waitKey(30)==1):
        break
cap.release()
cv2.destroyAllWindows()