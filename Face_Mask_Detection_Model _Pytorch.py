#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:





# In[2]:


#image resize
resize_transform = transforms.Resize((32, 32))


# Define the transforms for the training and test sets
train_transform = transforms.Compose([
    resize_transform,
    transforms.RandomHorizontalFlip(),  # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),  # convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
])

test_transform = transforms.Compose([
    resize_transform,
    transforms.ToTensor(),  # convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
])

# Load the datasets
train_dataset = torchvision.datasets.ImageFolder(root='/Users/elsiebasa/Documents/project_stats:ml/face_mask_detection/Train', 
                                                 transform=train_transform)
test_dataset = torchvision.datasets.ImageFolder(root='/Users/elsiebasa/Documents/project_stats:ml/face_mask_detection/Test', 
                                                transform=test_transform)

# Define the data loaders
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters 
num_epochs = 6
learning_rate = 0.001


# In[4]:


from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (8,6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break
        
show_batch(train_data_loader)


# In[5]:


iter(train_data_loader)


# # def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# 
# 
# # get some random training images
# dataiter = iter(train_data_loader)
# images, labels = dataiter.next()
# 

# In[6]:


#
classes = ['WithMask','WithoutMask']


    
import torch.nn as nn
import torch.optim as optim

# Define the model architecture with L2 regularization
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

# Create an instance of the model with L2 regularization
model = ConvNet(weight_decay=0.01)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)



# In[7]:


# Train the model on a dataset
for epoch in range(num_epochs):
    # Get an iterator for the training data loader
    train_iter = iter(train_data_loader)

    # Loop over the batches in the training data
    for i, (input_data, target_data) in enumerate(train_iter):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(input_data)

        # Compute the loss
        loss = criterion(output, target_data)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Print the loss
         # Print the loss every 100 batches
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_data_loader), loss.item()))
            
    print("Epoch " + " "+ str(epoch +1) + " "+ "Completed")
print("Finished Training")



# In[9]:


torch.save(model.state_dict(), "Face_Mask_ML.pt")

# n_total_steps = len(train_data_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_data_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)
torch.save(model.state_dict(), "face_mask_model.pt")
# In[10]:


batch_size = 32
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(2)]
    n_class_samples = [0 for i in range(2)]
    for images, labels in test_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(2):
        if n_class_samples[i] != 0:
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')
        else: print ("error")
            


# In[11]:


torch.save(model.state_dict(), 'Face_Mask_ML.pt')

#Later to restore:
model.load_state_dict(torch.load('Face_Mask_ML.pt'))
model.eval()


# In[ ]:





# In[ ]:





# In[12]:


import torchvision
import cv2

# Capture the video
video = cv2.VideoCapture(0)

# Loop through the frames of the video
while True:
    # Capture the frame
    ret, frame = video.read()

    # If the frame was not captured, break the loop
    if not ret:
        break

    # Pre-process the frame
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)

    # Classify the frame using the trained CNN model
    class_label = model(frame)

    # Display the classified frame
    cv2.imshow("Classified Frame", frame)

    # Display the class label
    cv2.putText(frame, class_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Classified Frame with Label", frame)

    # Break the loop if the user presses the "q" key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture device
video.release()

# Close all windows
cv2.destroyAllWindows()


# In[ ]:


mode = CNN()


# In[ ]:


print(classes
     )


# In[ ]:


model = 

# Save the model
torch.save(model.state_dict(), "face_mask_model.pt")


# In[ ]:


# Load the model and its parameters from a state_dict
state_dict = torch.load("face_mask_model.pt", map_location='cpu')

# Create a DNN module
net = cv2.dnn.readNetFromTorch(state_dict)

# Set the target device to CPU
f_model = cv2.dnn.readNetFromCaffe("model.prototxt", "model.caffemodel")


# In[23]:


import cv2


# In[ ]:




