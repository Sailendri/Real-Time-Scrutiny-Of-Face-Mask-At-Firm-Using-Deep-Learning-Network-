# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 18:51:52 2021

@author: saile
"""
# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
#from pygame import mixer
import numpy as np
import imutils
import time
import cv2
import os
import math

#system libraries
import os
import sys
from threading import Timer
import shutil
import time


import face_recognition
from datetime import datetime


#..........................................

def excel_func():
    path = os.getcwd()+"//dataset//att_without_mask"
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
        print(classNames)
 
    print("classNames completed") 
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')
 
    cap = cv2.VideoCapture(0)
    matches=[]

    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
    
    if matches[matchIndex]:
        name = classNames[matchIndex].upper()
        print(name+"hi")
        #else:
        #   name = "unknown"
    y1,x2,y2,x1 = faceLoc
    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    pname = markAttendance(name)
    print(pname+"hello")

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

    # importing the client from the twilio
    from twilio.rest import Client

    detailsdict = {"SAILENDRI": ["ACad9e820f038d6a4dc7aaed3446a9ffe7", "defe618bc71028525b92e2bfc54a3ed9", "+13213514012","+917868898423"],"VIKASHINI": ["AC679013caffa4d1ebdac437dcd3664339", "d339ada8faa030abe1b121b5078e7723", "4407036522","+919150098698"],"NANDHINI":  ["AC874288b8bc33c7fbfb9ef3596aedf02c", "76757066c3bc7ec48fe950b46130cdcc", "+13852066234","+917598470275"]}
    pname=pname.rstrip("\n")
    #name=name.rstrip("")
    x,y,z,a = detailsdict.get(name)
    print(x)
    print(y)
    print(z)
    print(a)
    # Your Account Sid and Auth Token from twilio account
    account_sid = x
    auth_token = y
    # instantiating the Client
    client = Client(account_sid, auth_token)
    # sending message
    message = client.messages.create(body='!!!Hey there please wear mask!!!', from_=z
                                     , to=a)
    # printing the sid after success
    print(message.sid)
    
    
  
def findEncodings(images):
  encodeList = []
  for img in images:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      encode = face_recognition.face_encodings(img)[0]
      encodeList.append(encode)
  return encodeList
 
def markAttendance(name):
  with open('peoplelist.csv','r+') as f:
    myDataList = f.readlines()
    nameList = []
    for line in myDataList:
       entry = line.split(',')
       final_name = entry[0]
       nameList.append(entry[0])
    #if name not in nameList:
       nowtime = datetime.now()
    dtString = nowtime.strftime('%H:%M:%S')
    f.writelines(f'\n{name},{dtString}')
    print(final_name)
  #f = open("peoplelist.csv", "r+")
  return final_name
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
 


while(True):
    detections = None 
    def detect_and_predict_mask(frame, faceNet, maskNet,threshold):
	# grab the dimensions of the frame and then construct a blob
   	# from it
       global detections 
       (h, w) = frame.shape[:2]
       blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
       faceNet.setInput(blob)
       detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
       faces = []
       locs = []
       preds = []
	# loop over the detections
       for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
            confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
            if confidence >threshold:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
            
			# add the face and bounding boxes to their respective
			# lists
                locs.append((startX, startY, endX, endY))
			#print(maskNet.predict(face)[0].tolist())
                preds.append(maskNet.predict(face)[0].tolist())
       return (locs, preds)

# SETTINGS
    MASK_MODEL_PATH=os.getcwd()+"\\model\\mask_model.h5"
    FACE_MODEL_PATH=os.getcwd()+"\\face_detector" 
    SOUND_PATH=os.getcwd()+"\\sounds\\alarm.wav" 
    THRESHOLD = 0.5

# Load Sounds
#mixer.init()
#sound = mixer.Sound(SOUND_PATH)

# load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([FACE_MODEL_PATH, "deploy.prototxt"])
    weightsPath = os.path.sep.join([FACE_MODEL_PATH,"res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(MASK_MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(0).start()
    time.sleep(2.0)



# loop over the frames from the video stream
    while True:
#        rab the frame from the threaded video stream and resize it
#      o have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        original_frame = frame.copy()
    #detect faces in the frame and determine if they are wearing a
	# face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet,THRESHOLD)

	# loop over the detected face locations and their corresponding
    	# locations
        for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

		      # determine the class label and color we'll use to draw
              #bounding box and text
              
            label = "Mask" if mask > withoutMask else "No Mask" 
        
		#if(label=="No Mask") and (mixer.get_busy()==False):
		#    sound.play()
        
        
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
            cv2.putText(original_frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), color, 2)
            cv2.rectangle(frame, (startX, startY+math.floor((endY-startY)/1.6)), (endX, endY), color, -1)
    
        
        cv2.addWeighted(frame, 0.5, original_frame, 0.5 , 0,frame)

	# show the output frame
        frame= cv2.resize(frame,(860,490))
        cv2.imshow("Masks Detection by Sailendri", frame)
        key = cv2.waitKey(1) & 0xFF
    
    
	# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        if (mask < withoutMask):
            excel_func()
        
        
# do a bit of cleanup
    cv2.destroyAllWindows()
  #  cap.release()
    vs.stop()
    import mergecode
    mergecode.py


