import cv2

import numpy as np

#Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays

import sys 
#This module provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.

facePath = "haarcascade_frontalface_default.xml"
smilePath = "haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(facePath)

#here we are creating an object named facecascade which will detect if that is particular feature or not
#cascadeclassifier is a class in cv2

smileCascade = cv2.CascadeClassifier(smilePath)
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
#webcam is getting on


cap.set(4,480)  #window height and width
#cap.set(3,480)

sF = 1.15

while True:
        
	ret, frame = cap.read()
	  #we are reading frames and converting the frames into grayscale i.e; we are converting from three color channel to one color channel
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor= sF,
		minNeighbors=9,
		minSize=(30,30)
		
        
	)
    

	for (x, y, w, h) in faces:
        # If faces are found, it returns the positions of detected faces as Rect(x,y,w,h)
        
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            #The rectangle will be drawn on roi_color
            #Two opposite vertices of the rectangle are defined by x,y and x+w, y+h
            #The color of the rectangle is given by Scalar(0,0,255) which is the BGR value for blue
            
            
		roi_gray = gray[y:y+h, x:x+w]  
        
        #roi is region of interest
		roi_color = frame[y:y+h, x:x+w]

		smile = smileCascade.detectMultiScale(
			gray,
			scaleFactor= 1.7,
			minNeighbors=22,
			minSize=(25, 25)
			)
			
		eyes = eye_cascade.detectMultiScale(
			gray,
			scaleFactor= 1.7,
			minNeighbors=22,
			minSize=(25, 25)
			)

       
		for (x, y, w, h) in smile:
			print ("Found", "smiles!")
			cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
           
           
			for (ex, ey, ew, eh) in eyes:
				print("Found", "eyes!")
				cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
  
	cv2.imshow('Smile Detector', frame)
	K = cv2.waitKey(7)
	if K== 27:
		break

cap.release()
cv2.destroyAllWindows()
