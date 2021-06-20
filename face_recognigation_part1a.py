# importing library 
import cv2
import numpy as np
###########
cap=cv2.VideoCapture(0)
face_cascade =cv2.CascadeClassifier("C:/Users/HP/Downloads/haarcascade_frontalface_alt.xml") 
skip=0
face_data=[] #list

## storing the path of folder 
dataset_path=r'C:/Users/HP/Desktop/data/'  
## making data set
file_name=input("enter the name of person: ") 

### infinite loop
while True: 
    ret, frame= cap.read()
    gray_frame =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)## converting into gray image 
    
    if ret==False:  ## checking success 
        continue
    
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5) ## detecting the cordinates of rectangle
    
    if len(faces)==0:
        continue
    
    ## Will sort them according to  there area (width * height)
     # key = labda f:f[2]*f[3] is a function  
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    
    ## will give last element 
    for face in faces[-1:]:
        x,y,w,h=face ## storing coordinate of last frame/faces
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        offset=10 
        ### making frame around face(taking gray scale to min matrix size )
        face_section=gray_frame[y-offset:y+h+offset,x-offset:x+w+offset] 
        face_section=cv2.resize(face_section,(100,100))
       
        ### save/ append at every 10th frame
        skip +=1
        if(skip%10==0):
            face_data.append(face_section)
            print(len(face_data))
            
      ### showing both frames on screen   
    cv2.imshow("video frame",frame)
    cv2.imshow("face section ",face_section)
    

    key_pressed =cv2.waitKey(1) & 0xFF
    if key_pressed ==ord('q'):
        break
    
##storing image data (reshaping into 1-D array )
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

##saving in npy form(The .npy format is the standard binary file)
## The format stores all of the shape and dtype 
np.save(dataset_path +file_name +'.npy',face_data) 
cap.release()
cv2.destroyAllWindows()
