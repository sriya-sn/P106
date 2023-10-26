import cv2


# Create our body classifier
body_cascade = cv2.CascadeClassifier("/Users/poornimaponnuswamy/Downloads/PRO-106-ProjectTemplate-main/haarcascade_fullbody.xml")

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:

    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    faces = body_cascade.detectMultiScale(grey,1.1,5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        crop = frame[y:y+h,x+w]
        cv2.imwrite("body.jpg",crop)
    cv2.imshow("img",frame)
    
    # Extract bounding boxes for any bodies identified
    

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
