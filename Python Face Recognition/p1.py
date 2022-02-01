'''
We can do face recognition more easily by making use of additional libraries in Python.
'''

#IMPORT UTILITIES
import cv2
import numpy as np
import face_recognition
import os

path = 'Image Database'
images = []
classNames = []
myList=os.listdir(path)
print(myList)
for cl in myList:       #   This for loop is here to get rid of the file extension like .jpg.
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):      # This function encodes all the sample files in the Image Database.
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)       # Starts Webcam

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)           # Captures the face in webcam
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)       # Encodes the face in webcam

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)        #Compares encoded Image Database and the face in webcam
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)        #Differentiates encoded Image Database and the face in webcam
                                                                                    #The less the distance means more similarity
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1= y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)




    cv2.imshow('Webcam',img)
    cv2.waitKey(1)