import streamlit as st
import cv2
import numpy as np
from PIL import  Image
import os

header=st.beta_container()
cont1=st.beta_container()
cont2=st.beta_container()
cont3=st.beta_container()

with cont1:
	st.header('FLO-IN FACE RECOGNITION EXAMPLE')
	my_placeholder = st.empty() 
	my_placeholder2 = st.empty()
	gendat=st.button('generate dataset')
	result=st.button('stop')
	res2=st.button('stop recog')
	if gendat:
			# Method to generate dataset to recognize a person
		def generate_dataset(img, id, img_id):
		    # write image in data dir
		    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg", img)
		
		# Method to draw boundary around the detected feature
		def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
		    # Converting image to gray-scale
		    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		    # detecting features in gray-scale image, returns coordinates, width and height of features
		    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
		    coords = []
		    # drawing rectangle around the feature and labeling it
		    for (x, y, w, h) in features:
		        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
		        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
		        coords = [x, y, w, h]
		    return coords
		
		# Method to detect the features
		def detect(img, faceCascade, img_id):
		    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
		    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
		    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
		    if len(coords)==4:
		        # Updating region of interest by cropping image
		        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
		        # Assign unique id to each user
		        user_id = 1
		        # img_id to make the name of each image unique
		        generate_dataset(roi_img, user_id, img_id)
		
		    return img
		
		
		# Loading classifiers
		faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		
		
		# Capturing real time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
		video_capture = cv2.VideoCapture(0)
		
		# Initialize img_id with 0
		img_id = 0
		
		while True:
		    if img_id % 50 == 0:
		        print("Collected ", img_id," images")
		        st.success('DATASST CREATED')
		        #result=True		
		    # Reading image from video stream
		    _, img = video_capture.read()
		    # Call method we defined above
		    img = detect(img, faceCascade, img_id)
		    # Writing processed image in a new window
		    #cv2.imshow("face detection", img)
		    my_placeholder.image(img,use_column_width=True)
		    img_id += 1
		    if (cv2.waitKey(1) & 0xFF == ord('q')or result):
		        break
		        exit()
		
		# releasing web-cam
		video_capture.release()
		# Destroying output window
		cv2.destroyAllWindows()
	
		
	
	classif=st.button('create XML')
	if classif:
		
		# Method to train custom classifier to recognize face
		def train_classifer(data_dir):
		    # Read all the images in custom data-set
		    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
		    faces = []
		    ids = []
		
		    # Store images in a numpy format and ids of the user on the same index in imageNp and id lists
		    for image in path:
		        img = Image.open(image).convert('L')
		        imageNp = np.array(img, 'uint8')
		        id = int(os.path.split(image)[1].split(".")[1])
		
		        faces.append(imageNp)
		        ids.append(id)
		
		    ids = np.array(ids)
		
		    # Train and save classifier
		    clf = cv2.face.LBPHFaceRecognizer_create()
		    clf.train(faces, ids)
		    clf.write("classifier.xml")
		    st.success('xml file created')
		
		
		train_classifer("data")
		
	
	recog=st.button('check rcognize')
	
	
	#my_placeholder = st.empty()
	if recog:
		def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
		    # Converting image to gray-scale
		    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		    # detecting features in gray-scale image, returns coordinates, width and height of features
		    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
		    coords = []
		    # drawing rectangle around the feature and labeling it
		    for (x, y, w, h) in features:
		        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
		        # Predicting the id of the user
		        id, _ = clf.predict(gray_img[y:y+h, x:x+w])
		        # Check for id of user and label the rectangle accordingly
		        if id==1:
		            cv2.putText(img, "vineet", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
		        coords = [x, y, w, h]
		
		    return coords
		
		# Method to recognize the person
		def recognize(img, clf, faceCascade):
		    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
		    coords = draw_boundary(img, faceCascade, 1.1, 10, color["white"], "Face", clf)
		    return img
		
		
		# Loading classifier
		faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		
		# Loading custom classifier to recognize
		clf = cv2.face.LBPHFaceRecognizer_create()
		clf.read("classifier.xml")
		
		# Capturing real time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
		video_capture = cv2.VideoCapture(0)
		
		while True:
		    # Reading image from video stream
		    _, img = video_capture.read()
		    # Call method we defined above
		    img = recognize(img, clf, faceCascade)
		    # Writing processed image in a new window
		    #cv2.imshow("face detection", img)
		    my_placeholder2.image(img,use_column_width=True)
		    if ((cv2.waitKey(1) & 0xFF == ord('q')) or res2) :
		        break
		        exit()
	# releasing web-cam
	#video_capture.release()
	# Destroying output window
	cv2.destroyAllWindows()	
	


	
	
		


		
	
