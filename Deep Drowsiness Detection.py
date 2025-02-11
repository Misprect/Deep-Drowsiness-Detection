#!/usr/bin/env python
# coding: utf-8

# # Install and Import Dependencies

# In[18]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5')


# In[19]:


get_ipython().system('cd yolov5')


# In[1]:


pip install -r requirements.txt


# In[1]:


import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


# # Load Model

# In[2]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# In[3]:


model


# # Make Detections

# In[4]:


img = 'https://ultralytics.com/images/zidane.jpg'


# In[5]:


results = model(img)
results.print()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))
plt.show()


# In[7]:


results.render()


# # Real Time Detections

# In[8]:


cap = cv2.VideoCapture('Untitled video - Made with Clipchamp (7).mp4')
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# # Train from Scratch

# In[9]:


import uuid   # Unique identifier
import os
import time


# In[10]:


IMAGES_PATH = os.path.join('data', 'images') #/data/images
labels = ['awake', 'drowsy', 'happy', 'sad']
number_imgs = 50


# In[20]:


cap = cv2.VideoCapture(0)
# Loop through labels
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    
    # Loop through image range
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        
        # Webcam feed
        ret, frame = cap.read()
        
        # Naming out image path
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        
        # Writes out image to file 
        cv2.imwrite(imgname, frame)
        
        # Render to the screen
        cv2.imshow('Image Collection', frame)
        
        # 2 second delay between captures
        time.sleep(2)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


# In[10]:


cap.release()
cv2.destroyAllWindows()


# In[11]:


print(os.path.join(IMAGES_PATH, labels[0]+'.'+str(uuid.uuid1())+'.jpg'))


# In[12]:


for label in labels:
    print('Collecting images for {}'.format(label))
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        print(imgname)   


# In[12]:


get_ipython().system('git clone https://github.com/tzutalin/labelImg')


# In[13]:


get_ipython().system('pip install pyqt5 lxml --upgrade')
get_ipython().system('cd labelImg && pyrcc5 -o libs/resources.py resources.qrc')


# In[ ]:


get_ipython().system('cd yolov5 && python train.py --img 320 --batch 16 --epochs 100 --data dataset.yml --weights yolov5s.pt --workers 2')


# # Load Custom Model

# In[ ]:


model = torch.hub.load('ultralytics/yolov5', 'custom', path=' yolov5/runs/train/exp/weights/last.pt',force_reload=True,_verbose=False)


# In[ ]:


img = os.path.join('data', 'images', 'awake.8cabb7a6-b47a-11ed-8ea5-6018953feb89.jpg')


# In[ ]:


results = model(img)


# In[ ]:


results.print()


# In[ ]:


results.render()  # Ensure bounding boxes are drawn on the image
image = np.squeeze(results.render())  # Get the image 

# Make a copy to remove read-only flag
image = image.copy()

plt.imshow(image)
plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))
plt.show()


# In[ ]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




