import cv2

# Method to generate dataset to recognize a person
def generate_dataset(img, id, img_id):
    # write image in data dir
    cv2.imwrite("data/"+str(con)+"/."+str(id)+"."+str(img_id)+".jpg", img)

# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("hi",gray_img)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    print(features)
    
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
   
    print(coords)
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
video_capture = cv2.VideoCapture('video.mp4')

# Initialize img_id with 0
img_id = 0
con=1
con2=1
while True:
    video_capture = cv2.VideoCapture('video'+str(con2)+'.mp4')
    #print("Collected ", img_id," images")
    # Reading image from video stream
    _, img = video_capture.read()
    # Call method we defined above
    img = detect(img, faceCascade, img_id)
    # Writing processed image in a new window
    #cv2.imshow("face detection", img)
    print(img)
    print(img.shape)
    img_id += 1
    if img_id % 10 == 0:
      con+=1
      con2+=1
      if con2==4:
        break
    
    elif img_id % 30 == 0:
      break        

# releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()
