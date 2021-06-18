import cv2 
cap=cv2.VideoCapture(0)
face_cascade =cv2.CascadeClassifier("C:/Users/HP/Downloads/haarcascade_frontalface_alt.xml")
while True:
    
    ret, frame= cap.read() ## capture camera frame, and ret store true and false
    gray_frame =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if ret==False:
        continue
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    ### 
    """ The first argument is the image, the second is the 
    scalefactor (how much the image size will be reduced at each image scale), 
    and the third is the minNeighbors (how many neighbors each rectangle should have)"""
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,125,0),2)## color and width 
        
    cv2.imshow("video frame",frame)
    key_pressed =cv2.waitKey(1) & 0xFF
    if key_pressed ==ord('n'):
        break
        
cap.release()
cv2.destroyAllWindows()