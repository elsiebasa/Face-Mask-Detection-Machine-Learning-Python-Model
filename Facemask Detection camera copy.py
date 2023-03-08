#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import threading
import time
import asyncio
import torch
import torchvision
from torchvision import transforms

from PIL import Image


# In[3]:


import torch
import torchvision
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from time import time
import torchvision.transforms as tt
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


# In[4]:


faceDetectModel = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print(faceDetectModel)


# In[5]:


class ConvNet(nn.Module):
    def __init__(self, weight_decay=0.01):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5) 
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        
        # Add L2 regularization to the fully connected layers
        self.fc1.weight_decay = weight_decay
        self.fc2.weight_decay = weight_decay
        self.fc3.weight_decay = weight_decay

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[6]:


face_mod = "Face_Mask_ML.pt"



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ConvNet() # a torch.nn.Module object
model.load_state_dict(torch.load(face_mod ))
model = model.to(device)




# In[7]:


model.eval()


# In[8]:


# Define the transformation to be applied to each frame
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Define the function to perform predictions on each frame
#def predict(frame):
#    img_tensor = transform(frame)
#    img_tensor = img_tensor.unsqueeze(0)
#    with torch.no_grad():
#        output = model(img_tensor)
#        _, predicted = torch.max(output, 1)
#    return predicted.item()


# In[12]:


#Define the transformation to be applied to each frame
transform = transforms.Compose([
   transforms.ToPILImage(),
   transforms.Resize((32, 32)),
   transforms.ToTensor(),
])

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

# Define the function to perform predictions on each face
def predict(face):
   img_tensor = transform(face)
   img_tensor = img_tensor.unsqueeze(0)
   with torch.no_grad():
       output = model(img_tensor)
       _, predicted = torch.max(output, 1)
   return predicted.item()

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
   # Capture a frame from the webcam
   ret, frame = cap.read()

   # Convert the frame to grayscale
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
   gray = cv2.equalizeHist(gray)



   # Detect faces in the frame
   faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)

   # Loop over each face detected in the frame
   for (x, y, w, h) in faces:
       # Extract the face from the frame
       face = frame[y:y+h, x:x+w]

       # Apply the prediction function to the face
       prediction = predict(face)

       # Display the prediction on the face
       if prediction == 0:
           text = "No mask"
           color = (0, 0, 255)
       else:
           text = "Mask"
           color = (0, 255, 0)
   
       cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
       cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

   # Display the frame
   cv2.imshow('frame', frame)

   # Quit if the user presses 'q'
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()


# In[10]:


# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()


# In[11]:


cap.release()


# In[14]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




