#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


import numpy as np


# In[3]:


from PIL import Image


# In[4]:


import os


# In[5]:


import pickle

BASE_DIR = os.path.dirname(os.path.abspath('faces_train.ipynd'))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('F:/dd/Library/etc/haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ","-").lower()
            #print(label, path)
            
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
                
            id_ = label_ids[label]
            #print(label_ids)
            
            pil_image = Image.open(path).convert("L")
            #size = (550,550)
            #final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                



with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("face-trainner.yml")


# In[6]:


import os
BASE_DIR = os.path.dirname(os.path.abspath('faces_train.ipynd'))
image_dir = os.path.join(BASE_DIR, "images")
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            print(path)


# In[7]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle


# In[ ]:





# In[8]:


face_cascade = cv2.CascadeClassifier('F:/dd/Library/etc/haarcascades/haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")
labels = {"person_name": 1}

with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


# In[ ]:


cap = cv2.VideoCapture(0)

while (True):
    
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf <=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            
        
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)
        
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x, y), (end_cord_x, end_cord_y), color, stroke)
        
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
   


# In[ ]:




