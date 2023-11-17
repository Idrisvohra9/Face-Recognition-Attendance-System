import os
from datetime import datetime
import imutils

import cv2
import face_recognition
import numpy as np

path = 'Faces'
b = True
images = []
classNames = []
myList = os.listdir(path)

print("Encoding Images...")
for cl in myList:
    curImg = cv2.imread(os.path.join(path, cl))
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    """
    This function takes a list of images and returns a list of face encodings for each image.

    Args:
    - images: A list of images

    Returns:
    - encodeList: A list of face encodings for each image
    """
    encodeList = []
    for img in images:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(img)
            if len(face_encodings) > 0:
                encode = face_encodings[0]
                encodeList.append(encode)
        except Exception as e:
            print(f"Error converting image color: {e}")
    return encodeList


def markAttendance(name):
    global b
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = set()
        for line in myDataList:
            entry = line.split(',')
            nameList.add(entry[0])
        if name not in nameList:
            now = datetime.now()
            if b:
                print(name + " Detected. ")
                b = False
                dtString = now.strftime('%H:%M:%S')
                f.write(f'\n{name},{dtString}')
        else:
            if b:
                print(name + " Detected. ")
                b = False


encodeListKnown = findEncodings(images)

print('Encoding Complete')

cap = cv2.VideoCapture(0)

ip = "192.168.88.150"

address = f"https://{ip}:8080/video"
cap.open(address)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

while True:
    success, img = cap.read()
    img = imutils.resize(img, width=500)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = [coord * 4 for coord in faceLoc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, classNames[matchIndex], (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(classNames[matchIndex])
        else:
            name = "Unknown"
            y1, x2, y2, x1 = [coord * 4 for coord in faceLoc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Face Recognition', img)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
